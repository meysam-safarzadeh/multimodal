import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import sys
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn

# Append the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(parent_dir)

from model import AttentionBottleneckFusion
from mint_pain_dataset_creator import create_dataset

torch.manual_seed(123)
np.random.seed(123)


def plot_interpretation(attr_z1, attr_z2):
    # Sum attributions across batch and sequence dimensions for z1 and z2
    feature_importance_z1 = attr_z1.sum(dim=0).sum(dim=0)
    feature_importance_z2 = attr_z2.sum(dim=0).sum(dim=0)

    # Normalize attributions
    feature_importance_z1 /= feature_importance_z1.abs().max()
    feature_importance_z2 /= feature_importance_z2.abs().max()

    # Plot for z1
    # Feature labels
    features = [
        "pose_Rx", "pose_Ry", "pose_Rz",
        "Inner Brow Raiser (AU01_r)", "Outer Brow Raiser (AU02_r)", "Brow Lowerer (AU04_r)",
        "Upper Lid Raiser (AU05_r)", "Cheek Raiser (AU06_r)", "Lid Tightener (AU07_r)",
        "Nose Wrinkler (AU09_r)", "Upper Lip Raiser (AU10_r)", "Lip Corner Puller (AU12_r)",
        "Dimpler (AU14_r)", "Lip Corner Depressor (AU15_r)", "Chin Raiser (AU17_r)",
        "Lip Stretcher (AU20_r)", "Lip Tightener (AU23_r)", "Lips Part (AU25_r)",
        "Jaw Drop (AU26_r)", "Blink (AU45_r)", "gaze_angle_x", "gaze_angle_y"
    ]

    # Assuming feature_importance_z1 is a tensor with the importance values
    plt.bar(range(len(feature_importance_z1)), feature_importance_z1.detach().cpu().numpy())
    plt.xticks(range(len(features)), features, rotation='vertical')

    # Set the title and labels
    plt.title('Feature Importance in z1')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()  # This will adjust the layout to fit the labels
    plt.show()

    # Plot for z2
    plt.bar(range(len(feature_importance_z2)), feature_importance_z2.detach().cpu().numpy())
    plt.title('Feature Importance in z2')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()


# Function to register a hook on multihead attention layers
def get_attention_hook(name, attention_dict):
    def hook(module, input, output):
        # Access the attention weights from the module if possible
        if hasattr(module, 'attention_weights'):
            attention_weights = module.attention_weights
            attention_dict[module] = attention_weights.detach()
    return hook


# Function to plot an attention map
def plot_attention_map(attention, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, annot=False, cmap='viridis')
    plt.title(title)
    plt.ylabel('Query Index')
    plt.xlabel('Key Index')
    plt.show()


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return forward_orig(*args, **kwargs)

    m.forward = wrap



def interpret_model(model, data_loader, device):
    model.eval()
    integrated_gradients = IntegratedGradients(model)

    # Assuming the first batch in the loader for demonstration
    data = next(iter(data_loader))
    z1, z2, labels = data
    z1, z2, labels = z1.to(device), z2.to(device), labels.to(device)

    # Register hooks to capture the attention weights
    # Register hooks
    save_output = SaveOutput()
    hook_handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            patch_attention(module)
            handle = module.register_forward_hook(save_output)
            hook_handles.append(handle)

    # Forward pass
    # with torch.no_grad():
    output = model(z1, z2)

    print(f"Number of hook handles: {len(hook_handles)}")
    print(f"Number of saved outputs: {len(save_output.outputs)}")
    for i, output in enumerate(save_output.outputs):
        print(f"Output {i + 1} shape: {output.shape}")

    # Visualize the attention maps
    # loop through the keys in hook_handles or specify layer names
    for i, attention in enumerate(save_output.outputs):
        plot_attention_map(attention[23, 0].cpu().detach().numpy(), f"Attention Map {i + 1}")

    baseline1 = torch.zeros_like(z1, device=device)
    baseline2 = torch.zeros_like(z2, device=device)

    # Calculate Integrated Gradients
    attributes, delta = integrated_gradients.attribute(inputs=(z1, z2), baselines=(baseline1, baseline2),
                                                       target=labels, return_convergence_delta=True)

    return attributes, delta


def main(hidden_dim, num_heads, num_layers, learning_rate, dropout_rate, weight_decay, downsample_method, mode,
         fusion_layers, n_bottlenecks, batch_size, num_epochs, verbose, fold, device, save_model, max_seq_len,
         classification_head, plot, head_layer_sizes):
    # Initialize parameters and data
    input_dim = [22, 512]
    num_classes = 5

    # Initialize datasets and dataloaders
    # Paths to your files
    fau_file_path = 'FAU_embedding/FAU_embeddings_with_labels.csv'
    thermal_file_path = 'thermal_embedding/Thermal_embeddings_and_filenames_new.npz'
    split_file_path = 'cross_validation_split_2.csv'

    # Create the DataLoader
    train_dataset, val_dataset, test_dataset = create_dataset(fau_file_path, thermal_file_path, split_file_path,
                                                              fold, batch_size=batch_size, max_seq_len=max_seq_len)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Assuming you have a data_loader for your dataset
    model = AttentionBottleneckFusion(input_dim, hidden_dim, num_heads, num_layers, fusion_layers, n_bottlenecks,
                                      num_classes, device, max_seq_len + 1, mode, dropout_rate,
                                      downsample_method, classification_head, head_layer_sizes).to(device)
    # Load the model
    best_model_path = 'checkpoints/model_best.pth.tar'
    model.load_state_dict(torch.load(best_model_path)['state_dict'])

    attributes, delta = interpret_model(model, train_loader, device)

    return attributes, delta


if __name__ == '__main__':
    attributes, delta = main(hidden_dim=[96, 512, 384], num_heads=[2, 64, 2], num_layers=[2, 3], learning_rate=3e-4,
                             dropout_rate=0.0, weight_decay=0.0, downsample_method='Linear', mode='separate',
                             fusion_layers=2, n_bottlenecks=4, batch_size=32, num_epochs=150, verbose=True, fold=1,
                             device='cuda:1', save_model=True, max_seq_len=40, classification_head=True, plot=True,
                             head_layer_sizes=[352, 112, 48])

    print(attributes[0].shape, attributes[1].shape, delta.shape)

    print(delta)

    plot_interpretation(attributes[0], attributes[1])