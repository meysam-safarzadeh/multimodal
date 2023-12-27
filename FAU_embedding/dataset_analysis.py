import pandas as pd
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Load the dataset
file_path = 'FAU_embeddings_with_labels.csv'
data = pd.read_csv(file_path)

# Keeping columns from index 1 to 23 and the 'label' column
columns_to_keep = data.columns[1:23].tolist() + ['label']
modified_data = data[columns_to_keep]

# Display the first few rows of the modified dataset
print(modified_data.head())

# Prepare data for UMAP
X = modified_data.drop(columns=['label']).values
y = modified_data['label'].values

# UMAP reduction to two dimensions
umap_model = umap.UMAP(n_components=2, n_jobs=-1)
X_reduced = umap_model.fit_transform(X)

# Plotting
plt.figure(figsize=(10, 8))
for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], label=label, alpha=0.5)

plt.title('UMAP projection of the dataset')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(title='Label')
plt.show()

# Applying PCA for dimensionality reduction
pca_model = PCA(n_components=2)
X_reduced_pca = pca_model.fit_transform(X)

# Plotting PCA results
plt.figure(figsize=(10, 8))
for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X_reduced_pca[indices, 0], X_reduced_pca[indices, 1], label=label, alpha=0.5)

plt.title('PCA projection of the dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Label')
plt.show()