import pandas as pd
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Load the dataset
file_path = 'Depth_embeddings_and_filenames_new.npz'
depth_data = np.load(file_path, allow_pickle=True)
file_names = depth_data['filenames']
depth_embeddings = depth_data['embeddings']
labels = [file_name[-8:-7] for file_name in file_names]
color_map = ['#0000FF', '#5555FF', '#AAAAFF', '#FF5555', '#FF0000']  # Blue to Red

# Prepare data for UMAP
X = depth_embeddings
y = np.int64(np.array(labels))

# UMAP reduction to two dimensions
umap_model = umap.UMAP(n_components=2, n_jobs=-1)
X_reduced = umap_model.fit_transform(X)

# Plotting
plt.figure(figsize=(10, 8), dpi=300)
for i, label in enumerate(np.unique(y)):
    indices = np.where(y == label)
    plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], label=label, alpha=0.5, color=color_map[i])

plt.title('UMAP projection of Depth embeddings')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(title='Label')
plt.savefig('/home/meysam/Pictures/Depth_embeddings_UMAP.png', format='png')
plt.show()
plt.close()


# Applying PCA for dimensionality reduction
pca_model = PCA(n_components=2)
X_reduced_pca = pca_model.fit_transform(X)

# Plotting PCA results
plt.figure(figsize=(10, 8), dpi=300)
for i, label in enumerate(np.unique(y)):
    indices = np.where(y == label)
    plt.scatter(X_reduced_pca[indices, 0], X_reduced_pca[indices, 1], label=label, alpha=0.5, color=color_map[i])

plt.title('PCA projection of Depth embeddings')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Label')
plt.savefig('/home/meysam/Pictures/Depth_embeddings_PCA.png', format='png')
plt.show()
