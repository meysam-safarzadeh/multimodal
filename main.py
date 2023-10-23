
import torch
import torch.nn as nn
import torchviz
from torch.autograd import Variable
import torch.nn.functional as F


# Define a class that applies the transformer L times
class MultiLayerTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(MultiLayerTransformer, self).__init__()

        # Define a single transformer encoder layer with batch_first=True
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                        batch_first=True)

        # Stack num_layers of these layers to form the complete transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, z):
        return self.transformer_encoder(z)


# Define a class that applies the MultiLayerModalityTransformer to two modalities
# and concatenates the results with T additional tokens in between
class ModailitySpecificTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, T):
        super(ModailitySpecificTransformer, self).__init__()
        self.modality1_transformer = MultiLayerTransformer(input_dim, hidden_dim, num_heads, num_layers[0])
        self.modality2_transformer = MultiLayerTransformer(input_dim, hidden_dim, num_heads, num_layers[1])

    def forward(self, z1, z2):
        z1_final = self.modality1_transformer(z1)
        z2_final = self.modality2_transformer(z2)

        return z1_final, z2_final


class FusionTransformers(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, Lf, T):
        super(FusionTransformers, self).__init__()
        self.Lf = Lf
        self.T = T
        # Adjusting the extra_tokens shape for batch_first=True
        self.extra_tokens = nn.Parameter(torch.randn(1, T, input_dim), requires_grad=True)

        self.layers_modality1 = self._get_layers(input_dim, num_heads, hidden_dim, Lf)
        self.layers_modality2 = self._get_layers(input_dim, num_heads, hidden_dim, Lf)

    def _get_layers(self, input_dim, num_heads, hidden_dim, Lf):
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   batch_first=True)
        return nn.ModuleList([encoder_layer for _ in range(Lf)])

    def forward(self, z1, z2):
        # Adjusting concatenation for batch_first=True
        # Repeat extra tokens for the batch size
        temp_tokens1 = temp_tokens2 = self.extra_tokens.repeat(z1.size(0), 1, 1)

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


class AttentionBottleneckFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, Lf, T, num_classes):
        super(AttentionBottleneckFusion, self).__init__()

        # CLS tokens for each modality
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, input_dim), requires_grad=True)
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, input_dim), requires_grad=True)

        # Initialize ModalitySpecificTransformer
        self.modality_specific_transformer = ModailitySpecificTransformer(input_dim, hidden_dim, num_heads, num_layers,
                                                                          T)

        # Initialize FusionTransformers
        self.fusion_transformer = FusionTransformers(input_dim, num_heads, hidden_dim, Lf, T)

        # Combining the CLS representations from both modalities for classification
        self.classifier = nn.Linear(2*input_dim, num_classes)

    def forward(self, z1, z2):
        # Concat the CLS tokens for each modality
        cls_token1_embed = self.cls_token1.repeat(z1.size(0), 1, 1)
        cls_token2_embed = self.cls_token2.repeat(z2.size(0), 1, 1)
        z1 = torch.cat([cls_token1_embed, z1], dim=1)
        z2 = torch.cat([cls_token2_embed, z2], dim=1)

        # Get the outputs from the modality-specific transformers
        z1, z2 = self.modality_specific_transformer(z1, z2)

        # Feed the outputs to the FusionTransformers
        z1_out, final_tokens, z2_out = self.fusion_transformer(z1, z2)

        # Extracting the CLS token's representation post transformation
        cls_representation1 = z1_out[:, 0, :]
        cls_representation2 = z2_out[:, 0, :]

        # Combining the two CLS representations
        combined_cls = torch.cat([cls_representation1, cls_representation2], dim=1)

        # Classification
        output = self.classifier(combined_cls)
        final_class = F.softmax(output, dim=1)

        return z1_out, final_tokens, z2_out, final_class


# Example usage
if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # [ ... Insert your class definitions here ... ]

    # Initialize parameters
    input_dim = 512  # Token embedding dimension
    hidden_dim = 2048  # Dimension of the inner feedforward network
    num_heads = 8  # Number of attention heads
    num_layers = [6, 4]  # Number of transformer encoder layers for modality 1 and modality 2 respectively
    T = 5  # Number of extra tokens
    Lf = 3  # Number of iterations for the TempTokensFusion
    batch_size = 32
    sequence_length = 10
    num_classes = 5

    # Generate sample inputs
    z1 = torch.randn(batch_size, sequence_length, input_dim)  # Input sequence in batch-first format for modality 1
    z2 = torch.randn(batch_size, sequence_length, input_dim)  # Input sequence in batch-first format for modality 2

    # Make them variables to compute the graph
    z1 = Variable(z1, requires_grad=True)
    z2 = Variable(z2, requires_grad=True)

    # Instantiate model
    model = AttentionBottleneckFusion(input_dim, hidden_dim, num_heads, num_layers, Lf, T, num_classes)

    # Get outputs
    z1_out, final_tokens, z2_out , predicted_class = model(z1, z2)

    # Combine the outputs into a single tuple
    combined_outputs = (z1_out, final_tokens, z2_out)

    # Create the graph using torchviz
    # dot = torchviz.make_dot(combined_outputs, params=dict(list(model.named_parameters()) + [('z1', z1), ('z2', z2)]))

    # Display the graph
    # dot.view()


    # Print shapes to check
    print(z1_out.shape)  # Expected: (batch_size, sequence_length, input_dim)
    print(final_tokens.shape)  # Expected: (batch_size, T, input_dim)
    print(z2_out.shape)  # Expected: (batch_size, sequence_length, input_dim)
    print(predicted_class.shape)  # Expected: (batch_size, num_classes)