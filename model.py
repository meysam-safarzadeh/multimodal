import torch
import torch.nn as nn
import torchviz
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import math


class DownSample(nn.Module):
    def __init__(self, output_dim, method=None, input_dim=None):
        super(DownSample, self).__init__()
        if method is not None:
            self.pool1 = nn.AdaptiveMaxPool1d(output_dim)
            self.pool2 = nn.Linear(input_dim, output_dim)
            self.method = method
        else:
            pass

    def forward(self, x):
        if self.method == 'MaxPool':
            reduced = self.pool1(x)
        elif self.method == 'Linear':
            # x shape: (batch, 7, 512)
            batch_size, seq_len, input_dim = x.size()

            # Reshape x to (-1, 512) to apply the linear layer
            x = x.view(-1, input_dim)
            x = self.pool2(x)

            # Reshape x back to (batch, 7, 23)
            reduced = x.view(batch_size, seq_len, -1)
        else:
            raise ValueError("Invalid method for down sampling. Choose 'MaxPool' or 'Linear'.")
        return reduced


# Define a class that applies the transformer L times
class MultiLayerTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim=None, downsmaple_method=None):
        super(MultiLayerTransformer, self).__init__()

        # Define a single transformer encoder layer with batch_first=True
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                        batch_first=True)

        # Stack num_layers of these layers to form the complete transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.down_sample = DownSample(output_dim, downsmaple_method, input_dim)
        self.downsample_method = downsmaple_method

    def forward(self, z):
        z = self.transformer_encoder(z)

        # Reduce the output dimension if specified
        if self.downsample_method is not None:
            z = self.down_sample(z)
        return z


# Define a class that applies the MultiLayerModalityTransformer to two modalities
# and concatenates the results with T additional tokens in between
class ModalitySpecificTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, T, downsmaple_method):
        super(ModalitySpecificTransformer, self).__init__()
        self.modality1_transformer = MultiLayerTransformer(input_dim[0], hidden_dim[0], num_heads[0], num_layers[0])
        self.modality2_transformer = MultiLayerTransformer(input_dim[1], hidden_dim[1], num_heads[1], num_layers[1],
                                                           input_dim[0], downsmaple_method)
        if len(input_dim) == 3:
            self.modality3_transformer = MultiLayerTransformer(input_dim[2], hidden_dim[2], num_heads[2], num_layers[2],
                                                               input_dim[0], downsmaple_method)

    def forward(self, z1, z2, z3):
        z1_final = self.modality1_transformer(z1)
        z2_final = self.modality2_transformer(z2)
        if z3 is not None:
            z3_final = self.modality3_transformer(z3)
            return z1_final, z2_final, z3_final

        return z1_final, z2_final, None


class FusionTransformers(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, Lf, B):
        super(FusionTransformers, self).__init__()
        self.Lf = Lf
        self.T = B
        # Adjusting the bottleneck tokens shape for batch_first=True
        self.bottleneck_tokens = nn.Parameter(torch.empty(1, B, input_dim[0]), requires_grad=True)
        init.normal_(self.bottleneck_tokens, mean=0, std=0.02)

        self.layers_modality1 = self._get_layers(input_dim[0], num_heads[-1], hidden_dim[-1], Lf)
        self.layers_modality2 = self._get_layers(input_dim[0], num_heads[-1], hidden_dim[-1], Lf)
        self.layers_modality3 = self._get_layers(input_dim[0], num_heads[-1], hidden_dim[-1], Lf)

    @staticmethod
    def _get_layers(input_dim, num_heads, hidden_dim, Lf):
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   batch_first=True)
        return nn.ModuleList([encoder_layer for _ in range(Lf)])

    def forward(self, z1, z2, z3=None):
        # Adjusting concatenation for batch_first=True
        # Repeat Bottleneck tokens for the batch size
        final_tokens = self.bottleneck_tokens.repeat(z1.size(0), 1, 1)

        for i in range(self.Lf):
            z1 = torch.cat((z1, final_tokens), dim=1)
            z2 = torch.cat((z2, final_tokens), dim=1)
            if z3 is not None:
                z3 = torch.cat((z3, final_tokens), dim=1)

            z1 = self.layers_modality1[i](z1)
            z2 = self.layers_modality2[i](z2)
            if z3 is not None:
                z3 = self.layers_modality3[i](z3)

            # Separate the output into z1, temp_tokens1, z2, and temp_tokens2
            z1, temp_tokens1 = z1[:, :-self.T, :], z1[:, -self.T:, :]
            z2, temp_tokens2 = z2[:, :-self.T, :], z2[:, -self.T:, :]
            if z3 is not None:
                z3, temp_tokens3 = z3[:, :-self.T, :], z3[:, -self.T:, :]
                final_tokens = 0.333333 * (temp_tokens1 + temp_tokens2 + temp_tokens3)
                continue

            # Average the two temporary tokens
            final_tokens = 0.5 * (temp_tokens1 + temp_tokens2)

        return z1, z2, z3


def positional_encoding(sequence_length, d_model, device):
    """Compute the sinusoidal positional encoding for a batch of sequences."""
    # Initialize a matrix to store the positional encodings
    pos_enc = torch.zeros(sequence_length, d_model)

    # Compute the positional encodings
    for pos in range(sequence_length):
        for i in range(0, d_model, 2):
            div_term = torch.exp(torch.tensor(-math.log(10000.0) * (i // 2) / d_model))
            pos_enc[pos, i] = torch.sin(pos * div_term)
            pos_enc[pos, i + 1] = torch.cos(pos * div_term)

    # Add an extra dimension to match the batch size in input
    pos_enc = pos_enc.unsqueeze(0).to(device)
    return pos_enc


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, seq_len, dropout_rate, head_layer_sizes, n_classes: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.seq = nn.Sequential(nn.Flatten(), nn.Linear(embedding_dim * seq_len, head_layer_sizes[0]), nn.ReLU(),
                                 nn.Dropout(dropout_rate), nn.Linear(head_layer_sizes[0], head_layer_sizes[1]),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate), nn.Linear(head_layer_sizes[1], head_layer_sizes[2]),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_rate), nn.Linear(head_layer_sizes[2], n_classes))

    def forward(self, x):
        x = self.norm(x)
        x = self.seq(x)
        return x


class ClassificationProcessor(nn.Module):
    def __init__(self, input_dim, dropout_rate, mode,
                 classification_head, max_seq_length, head_layer_sizes, num_classes, modalities):
        super(ClassificationProcessor, self).__init__()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate) if len(modalities) == 3 else None

        # Classification heads or layers
        if classification_head:
            self.combined_classifier = ClassificationHead(input_dim[0], max_seq_length * len(modalities)
                                                          , dropout_rate, head_layer_sizes)
            self.classifier1 = ClassificationHead(input_dim[0], max_seq_length, dropout_rate, head_layer_sizes)
            self.classifier2 = ClassificationHead(input_dim[0], max_seq_length, dropout_rate, head_layer_sizes)
            self.classifier3 = ClassificationHead(input_dim[0], max_seq_length, dropout_rate, head_layer_sizes)
        elif not classification_head:
            self.combined_classifier = nn.Linear(2 * input_dim[0], num_classes)  # Combined classifier
            self.classifier1 = nn.Linear(input_dim[0], num_classes)  # Separate classifier for modality 1
            self.classifier2 = nn.Linear(input_dim[0], num_classes)  # Separate classifier for modality 2
            self.classifier3 = nn.Linear(input_dim[0], num_classes)  # Separate classifier for modality 3

        self.mode = mode
        self.classification_head = classification_head

    def forward(self, z1_out, z2_out, z3_out=None):
        if self.classification_head:
            if self.mode == 'concat':
                representations = [z1_out, z2_out]
                if z3_out is not None:
                    representations.append(z3_out)
                combined_cls = torch.cat(representations, dim=1)
                final_output = self.combined_classifier(combined_cls)
                return final_output

            elif self.mode == 'separate':
                logits_output_1 = self.classifier1(z1_out)
                logits_output_2 = self.classifier2(z2_out)
                logits = [logits_output_1, logits_output_2]

                if z3_out is not None and self.classifier3:
                    logits_output_3 = self.classifier3(z3_out)
                    logits.append(logits_output_3)

                final_output = sum(logits) / len(logits)
                return final_output

            else:
                raise ValueError("Invalid mode. Choose 'concat' or 'separate'.")

        elif not self.classification_head:
            cls_representation1 = self.dropout1(z1_out[:, 0, :])
            cls_representation2 = self.dropout2(z2_out[:, 0, :])
            cls_representation3 = self.dropout3(z3_out[:, 0, :]) if z3_out is not None else None

            if self.mode == 'concat':
                representations = [cls_representation1, cls_representation2]
                if cls_representation3 is not None:
                    representations.append(cls_representation3)
                combined_cls = torch.cat(representations, dim=1)
                final_output = self.combined_classifier(combined_cls)

            elif self.mode == 'separate':
                logits_output_1 = self.classifier1(cls_representation1)
                logits_output_2 = self.classifier2(cls_representation2)
                logits = [logits_output_1, logits_output_2]

                if cls_representation3 is not None:
                    logits.append(self.classifier3(cls_representation3))

                final_output = sum(logits) / len(logits)

            else:
                raise ValueError("Invalid mode. Choose 'concat' or 'separate'.")

        else:
            raise ValueError("Invalid classification head. Choose True or False.")

        return final_output


class AttentionBottleneckFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, Lf, T, num_classes, device, max_seq_length,
                 mode, dropout_rate, downsmaple_method, classification_head, head_layer_sizes, modalities):
        super(AttentionBottleneckFusion, self).__init__()

        # CLS tokens for each modality
        self.cls_token1 = nn.Parameter(2 * torch.rand(1, 1, input_dim[0]) - 1, requires_grad=True)
        self.cls_token2 = nn.Parameter(2 * torch.rand(1, 1, input_dim[1]) - 1, requires_grad=True)
        self.cls_token3 = nn.Parameter(2 * torch.rand(1, 1, input_dim[2]) - 1, requires_grad=True) if len(modalities) == 3 else None

        # Positional encodings
        self.positional_encodings1 = positional_encoding(100, input_dim[0], device)
        self.positional_encodings2 = positional_encoding(100, input_dim[1], device)
        self.positional_encodings3 = positional_encoding(100, input_dim[2], device) if len(modalities) == 3 else None

        # Initialize ModalitySpecificTransformer
        self.modality_specific_transformer = ModalitySpecificTransformer(input_dim, hidden_dim, num_heads, num_layers,
                                                                         T, downsmaple_method)

        # Initialize FusionTransformers
        self.fusion_transformer = FusionTransformers(input_dim, num_heads, hidden_dim, Lf, T)

        self.classification_processor = ClassificationProcessor(input_dim, dropout_rate, mode, classification_head,
                                                                max_seq_length, head_layer_sizes, num_classes,
                                                                modalities)

    def forward(self, z1, z2, z3=None):
        # Concat the CLS tokens for each modality
        cls_token1_embed = self.cls_token1.repeat(z1.size(0), 1, 1)
        cls_token2_embed = self.cls_token2.repeat(z2.size(0), 1, 1)
        cls_token3_embed = self.cls_token3.repeat(z3.size(0), 1, 1) if z3 is not None else None

        z1 = torch.cat([cls_token1_embed, z1], dim=1)
        z2 = torch.cat([cls_token2_embed, z2], dim=1)
        z3 = torch.cat([cls_token3_embed, z3], dim=1) if z3 is not None else None

        # Add positional encodings
        z1 = z1 + self.positional_encodings1[:, z1.size(1), :]
        z2 = z2 + self.positional_encodings2[:, z2.size(1), :]
        z3 = z3 + self.positional_encodings3[:, z3.size(1), :] if z3 is not None else None

        # Get the outputs from the modality-specific transformers
        z1, z2, z3 = self.modality_specific_transformer(z1, z2, z3)

        # Feed the outputs to the FusionTransformers
        z1_out, z2_out, z3_out = self.fusion_transformer(z1, z2, z3)

        # Classification using the classification head
        final_output = self.classification_processor(z1_out, z2_out, z3_out)

        return final_output


class SingleModalityTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, Lf, T, num_classes, device, max_seq_length,
                 mode, dropout_rate, downsmaple_method, classification_head, head_layer_sizes, output_dim):
        super(SingleModalityTransformer, self).__init__()
        """
        - output_dim: The output dimension of the transformer will be downsampled to this dimension.
        - downsmaple_method: The method used to downsample the output of the transformer. Choose 'MaxPool' or 'Linear'
         or None for no downsampling.
        """

        # CLS tokens for each modality
        self.cls_token1 = nn.Parameter(2 * torch.rand(1, 1, input_dim) - 1, requires_grad=True)

        # Positional encodings
        self.positional_encodings1 = positional_encoding(100, input_dim, device)

        # Initialize ModalitySpecificTransformer
        self.multi_layer_transformer = MultiLayerTransformer(input_dim, hidden_dim, num_heads, num_layers,
                                                             output_dim, downsmaple_method)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)

        # Classification heads or layers
        if classification_head:
            self.classifier1 = ClassificationHead(output_dim, max_seq_length, dropout_rate, head_layer_sizes)

        elif not classification_head:
            self.classifier1 = nn.Linear(input_dim, num_classes)  # Separate classifier for modality 1

        self.classification_head = classification_head

    def forward(self, z1):
        # Concat the CLS tokens for each modality
        cls_token1_embed = self.cls_token1.repeat(z1.size(0), 1, 1)
        z1 = torch.cat([cls_token1_embed, z1], dim=1)

        # Add positional encodings
        z1 = z1 + self.positional_encodings1[:, z1.size(1), :]

        # Get the outputs from the modality-specific transformers
        z1_out = self.multi_layer_transformer(z1)

        # Classification using the classification head
        if self.classification_head:
            final_output = self.classifier1(z1_out)
            return final_output

        # Classification without classification head
        elif not self.classification_head:
            # Extracting the CLS token's representation post transformation
            cls_representation1 = self.dropout1(z1_out[:, 0, :])
            final_output = self.classifier1(cls_representation1)

        else:
            raise ValueError("Invalid classification head. Choose True or False.")

        return final_output
