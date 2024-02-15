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
import shap
from utils import prepare_z

# Append the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(parent_dir)

from model import AttentionBottleneckFusion
from mint_pain_dataset_creator import create_dataset

torch.manual_seed(123)
np.random.seed(123)


def get_average_importance(attr):
    # Get the average attributions across the sequence dimension and then across the batch dimension while ignoring the
    # padding tokens
    mask = attr != 0
    sum_across_samples = torch.sum(attr * mask, dim=1)
    non_zero_counts_samples = torch.sum(mask, dim=1)
    avg_across_samples = sum_across_samples / non_zero_counts_samples.clamp(min=1)
    feature_importance = avg_across_samples.mean(dim=0)

    return feature_importance


def plot_feature_importance(attr_z1, attr_z2):
    # Get the average attributions across the sequence dimension and then across the batch dimension while ignoring the
    # padding tokens
    feature_importance_z1 = get_average_importance(attr_z1)
    feature_importance_z2 = get_average_importance(attr_z2)

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


# Function to register a hook on multi head attention layers
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


def attention_map_extraction(model, data_loader, device, modalities):
    model.eval()
    # Assuming the first batch in the loader for demonstration
    data = next(iter(data_loader))
    z1, z2, z3, labels = data
    z1, z2, z3, labels = prepare_z(z1, z2, z3, labels, device, modalities)
    z1, z2, z3, labels = z1.to(device), z2.to(device), z3.to(device), labels.to(device)

    # Register hooks to capture the attention weights
    save_output = SaveOutput()
    hook_handles = []
    module_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            patch_attention(module)
            handle = module.register_forward_hook(save_output)
            hook_handles.append(handle)
            module_name.append(name)

    # Forward pass to get the model outputs through the forward hooks
    output = model(z1, z2, z3)

    # Print the shapes of the attention maps
    print(f"Number of hook handles: {len(hook_handles)}")
    print(f"Number of saved outputs: {len(save_output.outputs)}")
    for i, output in enumerate(save_output.outputs):
        print(f"Output {i + 1} shape: {output.shape}")

    # Visualize the attention maps
    # loop through the keys in hook_handles or specify layer names
    for i, attention in enumerate(save_output.outputs):
        plot_attention_map(attention[37, 0].cpu().detach().numpy(), f"Attention Map {i + 1}")

    return


def compute_feature_importances(model, data_loader, device, target, modalities):
    """
    Compute the feature importances using Integrated Gradients. The average importance for each feature is computed over
    all samples in the dataset.
    :param model: model with loaded weights
    :param data_loader: DataLoader object
    :param device: can be 'cpu' or 'cuda:0' etc.
    :param target: which class to compute the feature importances for
    :return: average feature importances for z1 and z2 and all the attributes for z1 and z2
    """
    model.eval()
    integrated_gradients = IntegratedGradients(model)
    total_importance_1 = None
    total_importance_2 = None
    all_attributes_1 = []
    all_attributes_2 = []
    all_attributes_3 = [] if len(modalities) == 3 else None
    count = 0

    for data in data_loader:
        z1, z2, z3, labels = data
        z1, z2, z3, labels = prepare_z(z1, z2, z3, labels, device, modalities)

        # Filter out the samples with the target class
        z1 = z1[labels == target]
        z2 = z2[labels == target]
        z3 = z3[labels == target]
        labels = labels[labels == target]

        z1, z2, z3, labels = z1.to(device), z2.to(device), z3.to(device), labels.to(device)

        baseline1 = torch.zeros_like(z1, device=device)
        baseline2 = torch.zeros_like(z2, device=device)
        baseline3 = torch.zeros_like(z3, device=device)

        # Calculate Integrated Gradients
        attributes, _ = integrated_gradients.attribute(inputs=(z1, z2, z3), baselines=(baseline1, baseline2, baseline3),
                                                       target=target, return_convergence_delta=True) if len(modalities) == 3 else \
            integrated_gradients.attribute(inputs=(z1, z2), baselines=(baseline1, baseline2), target=target,
                                             return_convergence_delta=True)

        # Compute average importance for the current batch
        feature_importance_1 = get_average_importance(attributes[0])
        feature_importance_2 = get_average_importance(attributes[1])
        feature_importance_3 = get_average_importance(attributes[2]) if len(modalities) == 3 else None

        # Accumulate feature importance
        if total_importance_1 is None:
            total_importance_1 = feature_importance_1
            total_importance_2 = feature_importance_2
            total_importance_3 = feature_importance_3 if len(modalities) == 3 else None
        else:
            total_importance_1 += feature_importance_1
            total_importance_2 += feature_importance_2
            total_importance_3 += feature_importance_3 if len(modalities) == 3 else None

        # Store attributes for each batch
        all_attributes_1.append(attributes[0])
        all_attributes_2.append(attributes[1])
        all_attributes_3.append(attributes[2]) if len(modalities) == 3 else None

        count += 1

    # Concatenate attributes from all batches
    all_attributes_1 = torch.cat(all_attributes_1, dim=0)
    all_attributes_2 = torch.cat(all_attributes_2, dim=0)
    all_attributes_3 = torch.cat(all_attributes_3, dim=0) if len(modalities) == 3 else None

    # Compute the average over all batches
    avg_importance_1 = total_importance_1 / count
    avg_importance_2 = total_importance_2 / count
    avg_importance_3 = total_importance_3 / count if len(modalities) == 3 else None

    # Print the average feature importances
    print("Average Feature Importance for z1:", avg_importance_1)
    print("Average Feature Importance for z2:", avg_importance_2)
    print("Average Feature Importance for z3:", avg_importance_3) if len(modalities) == 3 else None

    return [avg_importance_1, avg_importance_2, avg_importance_3], [all_attributes_1, all_attributes_2, all_attributes_3]


def compute_gradient_explainer(model, data_loader, device, target):
    """
    Compute the feature importances using SHAP. The average importance for each feature is computed over
    all samples in the dataset.
    :param model:
    :param data_loader:
    :param device:
    :param target:
    :return:
    """
    model.eval()
    shap_value_z1 = []
    batch = 0
    for data in data_loader:
        z1, z2, labels = data
        z1, z2, labels = z1.to(device), z2.to(device), labels.to(device)

        explainer = shap.GradientExplainer(model, [z1, z2])
        shap_values = explainer.shap_values([z1, z2])

        print(shap_values)
        print(len(shap_values))
        shap_values_z1 = shap_values[target][0]
        shap_value_z1.append(shap_values_z1)
        batch += 1
        print("SHAP calculation batch ", batch, " done")

    shap_value_z1 = torch.cat(shap_value_z1, dim=0)

    return shap_value_z1


def load_model(hidden_dim, num_heads, num_layers, learning_rate, dropout_rate, weight_decay, downsample_method, mode,
               fusion_layers, n_bottlenecks, batch_size, num_epochs, verbose, fold, device, save_model, max_seq_len,
               classification_head, plot, head_layer_sizes, modalities, fusion_dim, sub_independent, best_model_path):

    # Initialize parameters and data
    input_dim_dic = {'fau': 22, 'thermal': 512, 'depth': 128}
    input_dim = [input_dim_dic[modality] for modality in modalities]
    num_classes = 5

    # Initialize datasets and dataloaders
    # Paths to your files
    fau_file_path = 'embeddings_fau/FAU_embeddings_with_labels.csv'
    thermal_file_path = 'embeddings_thermal/Thermal_embeddings_and_filenames_new.npz'
    depth_file_path = 'embeddings_depth/Depth_embeddings_and_filenames_new.npz'
    split_file_path = 'cross_validation_split_2.csv'

    # Create the DataLoader
    train_dataset, val_dataset, test_dataset = create_dataset(fau_file_path, thermal_file_path, split_file_path,
                                                              fold, batch_size=batch_size, max_seq_len=max_seq_len,
                                                              depth_file_path=depth_file_path,
                                                              sub_independent=sub_independent)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Assuming you have a data_loader for your dataset
    model = AttentionBottleneckFusion(input_dim, hidden_dim, num_heads, num_layers, fusion_layers, n_bottlenecks,
                                      num_classes, device, max_seq_len, mode, dropout_rate,
                                      downsample_method, classification_head, head_layer_sizes,
                                      modalities, fusion_dim).to(device)

    # Load the model
    best_model_path = 'checkpoints/model_best.pth.tar' if best_model_path is None else best_model_path
    model.load_state_dict(torch.load(best_model_path)['state_dict'])

    return model, train_loader


if __name__ == '__main__':
    features = ["pose_Rx", "pose_Ry", "pose_Rz",
                "Inner Brow Raiser (AU01_r)", "Outer Brow Raiser (AU02_r)", "Brow Lowerer (AU04_r)",
                "Upper Lid Raiser (AU05_r)", "Cheek Raiser (AU06_r)", "Lid Tightener (AU07_r)",
                "Nose Wrinkler (AU09_r)", "Upper Lip Raiser (AU10_r)", "Lip Corner Puller (AU12_r)",
                "Dimpler (AU14_r)", "Lip Corner Depressor (AU15_r)", "Chin Raiser (AU17_r)",
                "Lip Stretcher (AU20_r)", "Lip Tightener (AU23_r)", "Lips Part (AU25_r)",
                "Jaw Drop (AU26_r)", "Blink (AU45_r)", "gaze_angle_x", "gaze_angle_y"]
    attributes_all_classes = []
    attributes_all_classes_abs = []

    for i in range(5):
        # Set the target class for computing the feature importances
        target = i
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        modalities = ['fau', 'depth', 'thermal']

        # Load the model and data loader
        model, data_loader = load_model(hidden_dim=[320, 768, 640, 352], num_heads=[2, 4, 64, 16], num_layers=[2, 3, 1],
                             learning_rate=0.0004,
                             dropout_rate=0.03, weight_decay=0.0008, downsample_method='Linear', mode='separate',
                             fusion_layers=7, n_bottlenecks=5, batch_size=128, num_epochs=200, verbose=True, fold=1,
                             device='cuda:1', save_model=True, max_seq_len=36, classification_head=True, plot=True,
                             head_layer_sizes=[64, 32, 16], modalities=modalities, fusion_dim=64,
                             sub_independent=False, best_model_path='/home/meysam/NursingSchool/code/checkpoints/model_best_fau_depth_thermal.pth.tar')

        # Compute the feature importances using Integrated Gradients
        _, attributes = compute_feature_importances(model, data_loader, device, target, modalities=modalities)
        print(attributes[0].shape, attributes[1].shape)
        plot_feature_importance(attributes[0], attributes[1])

        # Using list comprehension to collect all z1 tensors from the data loader and concatenate them
        all_z1 = [z1[labels == target] for z1, _, _, labels in data_loader]
        all_z1 = torch.cat(all_z1, dim=0)

        # Plot the feature importance using SHAP
        reshaped_z1 = all_z1.view(-1, 22)
        reshaped_attributes = attributes[0].view(-1, 22).cpu()

        reshaped_z1 = reshaped_z1[reshaped_attributes.sum(dim=1) != 0]
        reshaped_attributes = reshaped_attributes[reshaped_attributes.sum(dim=1) != 0]

        # Calculate the global feature importance, shape (22,)
        global_attributes = np.array(reshaped_attributes).mean(0)
        global_attributes_abs = np.abs(reshaped_attributes).mean(0)

        # Plot the global feature importance
        shap.summary_plot(reshaped_attributes.cpu().numpy(), reshaped_z1.cpu().numpy(),
                          feature_names=features,
                          max_display=22, plot_type='layered_violin')

        # compute_gradient_explainer(model, data_loader, device, target)
        # attention_map_extraction(model, data_loader, device, modalities)

        # Store the global feature importance for all classes
        attributes_all_classes.append(global_attributes)
        attributes_all_classes_abs.append(global_attributes_abs)

    # Plot the global feature importance matrix for all classes
    attributes_all_classes = np.array(attributes_all_classes)
    sum_across_classes_abs = np.sum(attributes_all_classes_abs, axis=0)
    plt.figure(figsize=(12, 8), dpi=300)
    sns.heatmap(attributes_all_classes, annot=True, cmap='coolwarm', center=0)
    plt.title('Global Feature Importance per Class')
    plt.ylabel('Class Index')
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Bar Plot the global feature importance matrix for all classes
    # Sort the features based on their importance
    sorted_indices = sorted(range(len(sum_across_classes_abs)), key=lambda i: sum_across_classes_abs[i], reverse=True)
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importance = [sum_across_classes_abs[i] for i in sorted_indices]
    plt.figure(figsize=(12, 8), dpi=300)
    # Plot the sorted bars
    bars = plt.bar(range(len(sorted_importance)), sorted_importance)
    # Optionally highlight the top 5 bars
    for i in range(5):
        bars[i].set_color('C1')  # Change color, or use any other method to highlight
    plt.title('Overall Feature Importance (Sorted)')
    plt.ylabel('Importance')
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
