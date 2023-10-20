
import torch
import torch.nn as nn

# Define the Transformer model
class ModalityTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(ModalityTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
        )

    def forward(self, zl):
        return self.transformer(zl)


# Define a class that applies the transformer L times
class MultiLayerModalityTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, L):
        super(MultiLayerModalityTransformer, self).__init__()
        self.L = L
        self.transformers = nn.ModuleList([ModalityTransformer(input_dim, hidden_dim, num_heads, num_layers) for _ in range(L)])

    def forward(self, z):
        for i in range(self.L):
            z = self.transformers[i](z)
        return z


# Define a class that applies the MultiLayerModalityTransformer to two modalities
# and concatenates the results with T additional tokens in between
class FusionTwoModalities(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, L, T):
        super(FusionTwoModalities, self).__init()
        self.modality1_transformer = MultiLayerModalityTransformer(input_dim, hidden_dim, num_heads, num_layers, L)
        self.modality2_transformer = MultiLayerModalityTransformer(input_dim, hidden_dim, num_heads, num_layers, L)
        self.T = T
        self.extra_tokens = nn.Parameter(torch.randn(1, input_dim, T), requires_grad=True)

    def forward(self, z1, z2):
        z1_final = self.modality1_transformer(z1)
        z2_final = self.modality2_transformer(z2)

        # Concatenate T additional tokens in between z1_final and z2_final
        concatenated_output = torch.cat((z1_final, self.extra_tokens, z2_final), dim=-1)

        return concatenated_output


class TempTokensFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, Lf):
        super(TempTokensFusion, self).__init()
        self.Lf = Lf
        self.z1_transformer = MultiLayerModalityTransformer(input_dim, hidden_dim, num_heads, num_layers, Lf)
        self.z2_transformer = MultiLayerModalityTransformer(input_dim, hidden_dim, num_heads, num_layers, Lf)

    def forward(self, z1, z2, some_tokens):
        temp_tokens1 = temp_tokens2 = some_tokens

        for _ in range(self.Lf):
            z1_final = self.z1_transformer(torch.cat((z1, temp_tokens2), dim=-1))
            z2_final = self.z2_transformer(torch.cat((z2, temp_tokens1), dim=-1))

            # Separate the output into z1, temp_tokens1, z2, and temp_tokens2
            z1, temp_tokens1 = z1_final[:, :, :z1.shape[-1]], z1_final[:, :, z1.shape[-1]:]
            z2, temp_tokens2 = z2_final[:, :, :z2.shape[-1]], z2_final[:, :, z2.shape[-1]:]

            # Average the two temporary tokens
            final_tokens = 0.5 * (temp_tokens1 + temp_tokens2)

        return z1_final, final_tokens, z2_final


# Example usage
if __name__ == "__main__":
    input_dim = 7  # Adjust to your input dimension
    hidden_dim = 256  # Adjust to your hidden dimension
    num_heads = 4
    num_layers = 3
    num_modalities = 3  # RGB, Spec, and more modalities
    B = 4  # Number of bottleneck tokens

    fusion_model = FusionModel(input_dim, hidden_dim, num_heads, num_layers, num_modalities, B)

    # Input data for each modality
    z_rgb = torch.randn(32, input_dim, 22)  # Adjust the shape and values
    z_spec = torch.randn(32, input_dim, 10)  # Adjust the shape and values

    # Combine the modalities
    x = [z_rgb, z_spec]

    # Forward pass to compute fusion tokens
    zlplus1_fsn = fusion_model(x)

    print("Fusion Tokens Shape:", zlplus1_fsn.shape)
