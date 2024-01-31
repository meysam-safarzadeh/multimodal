import torch
import os
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from pytorch_captum import load_model
import umap
import matplotlib.pyplot as plt

# Append the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(parent_dir)

from utils import prepare_z

torch.manual_seed(123)
np.random.seed(123)


def extract_features_from_layers(model, data_loader, layer_names, device, modalities):
    extracted_features = {}

    # Assuming the first batch in the loader for demonstration
    data = next(iter(data_loader))
    z1, z2, z3, labels = data
    z1, z2, z3, labels = z1.to(device), z2.to(device), z3.to(device), labels.to(device)
    z1, z2, z3, labels = prepare_z(z1, z2, z3, labels, device, modalities)

    def hook_fn(module, input, output):
        extracted_features[module] = output.detach()

    hooks = []
    for name, layer in model.named_modules():
        if name in layer_names:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Forward pass
    _ = model(z1, z2, z3)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Collecting and returning the features
    features = [extracted_features[module] for name, module in model.named_modules() if name in layer_names]
    return features, labels


def plot_projection(title, labels, X_reduced, save_dir, xlabel, ylabel):
    plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.axes()  # Creating a 3D plot
    color_map = ['#0000FF', '#5555FF', '#AAAAFF', '#FF5555', '#FF0000']  # Blue to Red
    for i, label in enumerate(np.unique(labels)):
        indices = np.where(labels == label)
        ax.scatter(X_reduced[indices, 0], X_reduced[indices, 1], label=label, alpha=0.5, color=color_map[i])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title='Label')
    plt.savefig(save_dir, format='png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # Load the model and data loader
    device = torch.device('cuda:0')
    best_model_path = 'checkpoints/model_best_fau_depth_thermal.pth.tar'
    modalities = ['fau', 'depth', 'thermal']
    model, data_loader = load_model(hidden_dim=[320, 768, 640, 352], num_heads=[2, 4, 64, 16], num_layers=[2, 3, 1],
                                    learning_rate=0.0004,
                                    dropout_rate=0.03, weight_decay=0.0008, downsample_method='Linear', mode='separate',
                                    fusion_layers=7, n_bottlenecks=5, batch_size=1024, num_epochs=150, verbose=True,
                                    fold=0,
                                    device=device, save_model=True, max_seq_len=36, classification_head=True,
                                    plot=True,
                                    head_layer_sizes=[64, 32, 16], modalities=modalities,
                                    fusion_dim=64, sub_independent=True, best_model_path=best_model_path)

    print(model)

    # Choose the layers to extract features from
    layer_names = ['classification_processor.classifier1.seq.7',
                   'classification_processor.classifier2.seq.7',
                   'classification_processor.classifier3.seq.7']
    features, labels = extract_features_from_layers(model, data_loader, layer_names, device, modalities)

    # Concatenate the tensors along a specific dimension (e.g., 0)
    concatenated_tensor = torch.cat((features[0], features[1], features[2]), dim=1)

    # Move the tensor to CPU and convert to a NumPy array
    X = concatenated_tensor.cpu().numpy()
    y = np.int64(np.array(labels.cpu()))

    # UMAP reduction to two dimensions
    umap_model = umap.UMAP(n_components=3, n_jobs=-1)
    X_reduced = umap_model.fit_transform(X)

    plot_projection('3D UMAP projection of the final embeddings (FAU, Depth, and Thermal)', y, X_reduced,
                '/home/meysam/Pictures/fau_depth_thermal_embeddings_UMAP.png', 'UMAP 1', 'UMAP 2')

    # Assuming X_reduced_pca contains three components and y contains the labels

    fig = go.Figure()

    # Calculate the range for each axis
    x_range = [np.min(X_reduced[:, 0]), np.max(X_reduced[:, 0])]
    y_range = [np.min(X_reduced[:, 1]), np.max(X_reduced[:, 1])]
    z_range = [np.min(X_reduced[:, 2]), np.max(X_reduced[:, 2])]

    for label in np.unique(y):
        indices = np.where(y == label)[0]
        fig.add_trace(go.Scatter3d(
            x=X_reduced[indices, 0],
            y=X_reduced[indices, 1],
            z=X_reduced[indices, 2],
            mode='markers',
            marker=dict(size=5, opacity=0.5),
            name=str(label)
        ))

    fig.update_layout(
        title='3D UMAP projection of Depth embeddings',
        scene=dict(
            xaxis=dict(range=x_range, title='UMAP 1'),
            yaxis=dict(range=y_range, title='UMAP 2'),
            zaxis=dict(range=z_range, title='UMAP 3')
        ),
        legend_title="Label"
    )

    fig.show()

    # Applying PCA for dimensionality reduction
    pca_model = PCA(n_components=2)
    X_reduced_pca = pca_model.fit_transform(X)

    # Plotting PCA results
    plot_projection('3D PCA projection of the final embeddings (FAU, Depth, and Thermal)', y, X_reduced_pca,
                '/home/meysam/Pictures/fau_depth_thermal_embeddings_PCA.png', 'PC1', 'PC2')
