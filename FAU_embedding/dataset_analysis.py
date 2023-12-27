import pandas as pd

# Load the dataset
file_path = 'FAU_embeddings_with_labels.csv'
data = pd.read_csv(file_path)

# Keeping columns from index 1 to 23 and the 'label' column
columns_to_keep = data.columns[1:23].tolist() + ['label']
modified_data = data[columns_to_keep]

# Display the first few rows of the modified dataset
print(modified_data.head())
