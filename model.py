
import torch
import torch.nn as nn
import torchviz
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import math


class DownSample(nn.Module):
    def __init__(self, output_dim, method=None):
        super(DownSample, self).__init__()
        if method is not None:
            self.pool1 = nn.AdaptiveMaxPool1d(output_dim)
            self.pool2 = nn.Linear(512, output_dim)
            self.method = method
        else:
            pass

    def forward(self, x):
        if self.method == 'MaxPool':
            reduced = self.pool1(x)
        elif self.method == 'Linear':
            # x shape: (batch, 7, 512)
            batch_size, seq_len, _ = x.size()

            # Reshape x to (-1, 512) to apply the linear layer
            x = x.view(-1, 512)
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
        self.down_sample = DownSample(output_dim, downsmaple_method)
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

    def forward(self, z1, z2):
        z1_final = self.modality1_transformer(z1)
        z2_final = self.modality2_transformer(z2)

        return z1_final, z2_final


class FusionTransformers(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, Lf, B):
        super(FusionTransformers, self).__init__()
        self.Lf = Lf
        self.T = B
        # Adjusting the bottleneck tokens shape for batch_first=True
        self.bottleneck_tokens = nn.Parameter(torch.empty(1, B, input_dim[0]), requires_grad=True)
        init.normal_(self.bottleneck_tokens, mean=0, std=0.02)

        self.layers_modality1 = self._get_layers(input_dim[0], num_heads[2], hidden_dim[2], Lf)
        self.layers_modality2 = self._get_layers(input_dim[0], num_heads[2], hidden_dim[2], Lf)

    def _get_layers(self, input_dim, num_heads, hidden_dim, Lf):
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   batch_first=True)
        return nn.ModuleList([encoder_layer for _ in range(Lf)])

    def forward(self, z1, z2):
        # Adjusting concatenation for batch_first=True
        # Repeat Bottleneck tokens for the batch size
        temp_tokens1 = temp_tokens2 = self.bottleneck_tokens.repeat(z1.size(0), 1, 1)

        for i in range(self.Lf):
            z1 = torch.cat((z1, temp_tokens1), dim=1)
            z2 = torch.cat((z2, temp_tokens2), dim=1)

            z1 = self.layers_modality1[i](z1)
            z2 = self.layers_modality2[i](z2)

            # Separate the output into z1, temp_tokens1, z2, and temp_tokens2
            z1, temp_tokens1 = z1[:, :-self.T, :], z1[:, -self.T:, :]
            z2, temp_tokens2 = z2[:, :-self.T, :], z2[:, -self.T:, :]

            # Average the two temporary tokens
            final_tokens = 0.5 * (temp_tokens1 + temp_tokens2)

        return z1, final_tokens, z2


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


class AttentionBottleneckFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, Lf, T, num_classes, device, max_seq_length=10,
                 mode='concat', dropout_rate=0.0, downsmaple_method='MaxPool'):
        super(AttentionBottleneckFusion, self).__init__()

        # CLS tokens for each modality
        self.cls_token1 = nn.Parameter(2 * torch.rand(1, 1, input_dim[0]) - 1, requires_grad=True)
        self.cls_token2 = nn.Parameter(2 * torch.rand(1, 1, input_dim[1]) - 1, requires_grad=True)

        # Positional encodings
        self.positional_encodings1 = positional_encoding(max_seq_length, input_dim[0], device)
        self.positional_encodings2 = positional_encoding(max_seq_length, input_dim[1], device)

        # Initialize ModalitySpecificTransformer
        self.modality_specific_transformer = ModalitySpecificTransformer(input_dim, hidden_dim, num_heads, num_layers,
                                                                         T, downsmaple_method)

        # Initialize FusionTransformers
        self.fusion_transformer = FusionTransformers(input_dim, num_heads, hidden_dim, Lf, T)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Combining the CLS representations from both modalities for classification
        self.combined_classifier = nn.Linear(2*input_dim[0], num_classes)  # Combined classifier
        self.classifier1 = nn.Linear(input_dim[0], num_classes)  # Separate classifier for modality 1
        self.classifier2 = nn.Linear(input_dim[0], num_classes)  # Separate classifier for modality 2
        self.mode = mode  # Mode for classification

    def forward(self, z1, z2):
        # Concat the CLS tokens for each modality
        cls_token1_embed = self.cls_token1.repeat(z1.size(0), 1, 1)
        cls_token2_embed = self.cls_token2.repeat(z2.size(0), 1, 1)
        z1 = torch.cat([cls_token1_embed, z1], dim=1)
        z2 = torch.cat([cls_token2_embed, z2], dim=1)

        # Add positional encodings
        z1 = z1 + self.positional_encodings1[:, z1.size(1), :]
        z2 = z2 + self.positional_encodings2[:, z2.size(1), :]

        # Get the outputs from the modality-specific transformers
        z1, z2 = self.modality_specific_transformer(z1, z2)

        # Feed the outputs to the FusionTransformers
        z1_out, final_tokens, z2_out = self.fusion_transformer(z1, z2)

        # Extracting the CLS token's representation post transformation
        cls_representation1 = self.dropout1(z1_out[:, 0, :])
        cls_representation2 = self.dropout2(z2_out[:, 0, :])

        if self.mode == 'concat':
            # Combining the two CLS representations
            combined_cls = torch.cat([cls_representation1, cls_representation2], dim=1)

            # Classification using combined classifier
            output = self.combined_classifier(combined_cls)
            final_class = F.softmax(output, dim=1)

        elif self.mode == 'separate':
            # Classification using separate classifiers
            output1 = self.classifier1(cls_representation1)
            output2 = self.classifier2(cls_representation2)

            # Averaging the logits from both classifiers
            averaged_output = (output1 + output2) / 2
            final_class = F.softmax(averaged_output, dim=1)

        else:
            raise ValueError("Invalid mode. Choose 'concat' or 'separate'.")

        return z1_out, final_tokens, z2_out, final_class

# 
# # Example usage
# if __name__ == "__main__":
#     import torch
#     import torch.nn as nn
# 
#     # [ ... Insert your class definitions here ... ]
# 
#     # Initialize parameters
#     input_dim = 512  # Token embedding dimension
#     hidden_dim = 2048  # Dimension of the inner feedforward network
#     num_heads = 8  # Number of attention heads
#     num_layers = [6, 4]  # Number of transformer encoder layers for modality 1 and modality 2 respectively
#     T = 5  # Number of bottleneck tokens
#     Lf = 3  # Number of iterations for the TempTokensFusion
#     batch_size = 32
#     sequence_length = 10
#     num_classes = 5
# 
#     # Generate sample inputs
#     z1 = torch.randn(batch_size, sequence_length, input_dim)  # Input sequence in batch-first format for modality 1
#     z2 = torch.randn(batch_size, sequence_length, input_dim)  # Input sequence in batch-first format for modality 2
# 
#     # Make them variables to compute the graph
#     z1 = Variable(z1, requires_grad=True)
#     z2 = Variable(z2, requires_grad=True)
# 
#     # Instantiate model
#     model = AttentionBottleneckFusion(input_dim, hidden_dim, num_heads, num_layers, Lf, T, num_classes)
# 
#     # Get outputs
#     z1_out, final_tokens, z2_out , predicted_class = model(z1, z2)
# 
#     # Combine the outputs into a single tuple
#     combined_outputs = (z1_out, final_tokens, z2_out)
# 
#     # Create the graph using torchviz
#     # dot = torchviz.make_dot(combined_outputs, params=dict(list(model.named_parameters()) + [('z1', z1), ('z2', z2)]))
# 
#     # Display the graph
#     # dot.view()
# 
# 
#     # Print shapes to check
#     print(z1_out.shape)  # Expected: (batch_size, sequence_length, input_dim)
#     print(final_tokens.shape)  # Expected: (batch_size, T, input_dim)
#     print(z2_out.shape)  # Expected: (batch_size, sequence_length, input_dim)
#     print(predicted_class.shape)  #     Expected: (batch_size, num_classes)