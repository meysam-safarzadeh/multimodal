from torch.utils.data import Dataset, DataLoader, Subset
import torch
import numpy as np
import pandas as pd


class MintPainDataset(Dataset):
    def __init__(self, fau_dataframe, thermal_file_path, fau_min_max_vals, thermal_min_max_vals):
        """
        Initialize the dataset with FAU FAU_dataframe and thermal embeddings.

        Args:
            fau_dataframe (DataFrame): DataFrame containing FAU embeddings.
            thermal_file_path (str): Path to .npz file with Thermal embeddings.
        """
        # FAU embeddings
        self.FAU_dataframe = fau_dataframe
        self.sequences = fau_dataframe.groupby(['sub', 'trial', 'sweep', 'label']).groups

        # Thermal embeddings
        thermal_data = np.load(thermal_file_path)
        self.thermal_embeddings = {filename: embedding for filename, embedding in zip(thermal_data['filenames'], thermal_data['embeddings'])}

        # Min and max values for each modality
        self.fau_min_vals, self.fau_max_vals = fau_min_max_vals
        self.thermal_min_vals, self.thermal_max_vals = thermal_min_max_vals

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        indices = self.sequences[list(self.sequences)[idx]]
        fau_data = self.FAU_dataframe.iloc[indices]

        fau_embeddings = self._get_fau_embeddings(fau_data)
        thermal_embeddings = self._get_thermal_embeddings(fau_data['file name'])

        label = torch.tensor(fau_data.iloc[0]['label'], dtype=torch.long)

        return fau_embeddings, thermal_embeddings, label

    def _get_fau_embeddings(self, data):
        fau_features = 22
        fau_embeddings = data.iloc[:, 1:fau_features+1].values
        # Apply min-max normalization and scale to -1 to 1
        fau_embeddings = 2 * ((fau_embeddings - self.fau_min_vals) / (self.fau_max_vals - self.fau_min_vals)) - 1
        fau_embeddings = self._pad_embeddings(fau_embeddings, fau_features)
        return torch.tensor(fau_embeddings, dtype=torch.float32)

    def _get_thermal_embeddings(self, filenames):
        thermal_embedding_size = 512
        thermal_embeddings = [self.thermal_embeddings.get(filename, np.zeros(thermal_embedding_size)) for filename in filenames]
        thermal_embeddings = np.array(thermal_embeddings)
        # Apply min-max normalization and scale to -1 to 1
        thermal_embeddings = 2 * ((thermal_embeddings - self.thermal_min_vals) / (
                    self.thermal_max_vals - self.thermal_min_vals)) - 1
        thermal_embeddings = self._pad_embeddings(thermal_embeddings, thermal_embedding_size, axis=1)
        return torch.tensor(thermal_embeddings, dtype=torch.float32)

    def _pad_embeddings(self, embeddings, num_features, axis=1):
        max_samples = 7
        if embeddings.shape[axis] < max_samples:
            print(f"Padding embeddings from shape {embeddings.shape} to {max_samples} samples")
            padding_shape = list(embeddings.shape)
            padding_shape[axis] = max_samples - embeddings.shape[axis]
            padding = np.zeros(padding_shape)
            embeddings = np.concatenate((embeddings, padding), axis=axis)
            print(f"New embeddings shape: {embeddings.shape}")
        return embeddings


def create_dataset(fau_file_path, thermal_file_path, split_file_path, iteration, batch_size=64):
    """
    Create dataset for the FAU and thermal embeddings' dataset, split into train, validation, and test sets for
    the given iteration.

    Args:
    fau_file_path (str): Path to the CSV file containing FAU embeddings.
    thermal_file_path (str): Path to the NPZ file containing thermal embeddings.
    split_file_path (str): Path to the CSV file containing split information.
    iteration (int): The iteration number to select the split.
    batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.

    Returns:
    tuple: A tuple containing the train, validation, and test Datasets.
    """
    # Read the datasets
    df = pd.read_csv(fau_file_path)
    split_df = pd.read_csv(split_file_path)

    # Extract the split information for the given iteration
    train_subjects = np.int16(split_df.loc[split_df['Iteration'] == iteration, 'Training'].values[0].split(','))
    val_subjects = np.int16(split_df.loc[split_df['Iteration'] == iteration, 'Validation'].values[0].split(','))
    test_subjects = np.int16(split_df.loc[split_df['Iteration'] == iteration, 'Test'].values[0].split(','))

    # Split the dataset
    train_df = df[df['sub'].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df['sub'].isin(val_subjects)].reset_index(drop=True)
    test_df = df[df['sub'].isin(test_subjects)].reset_index(drop=True)

    # Get min and max values for each modality
    fau_min_max_vals, thermal_min_max_vals = get_min_max_for_each_modality(train_df, thermal_file_path)

    # Create subsets
    train_dataset = MintPainDataset(train_df, thermal_file_path, fau_min_max_vals, thermal_min_max_vals)
    val_dataset = MintPainDataset(val_df, thermal_file_path, fau_min_max_vals, thermal_min_max_vals)
    test_dataset = MintPainDataset(test_df, thermal_file_path, fau_min_max_vals, thermal_min_max_vals)

    return train_dataset, val_dataset, test_dataset


def get_min_max_for_each_modality(train_df, thermal_file_path):
    """
    Extract corresponding thermal samples based on 'file name' and calculate min-max for each modality.

    Args:
        train_df (DataFrame): DataFrame containing FAU embeddings and 'file name' column.
        thermal_file_path (str): Path to the NPZ file containing thermal embeddings.

    Returns:
        tuple of tuples: A tuple containing two tuples, first with min and max values for FAU data,
                         and second with min and max values for thermal data.
    """
    # Load thermal data
    thermal_data = np.load(thermal_file_path)
    thermal_embeddings_dict = {filename: embedding for filename, embedding in zip(thermal_data['filenames'], thermal_data['embeddings'])}

    # Extract corresponding thermal samples
    thermal_samples = np.array([thermal_embeddings_dict[fname] for fname in train_df['file name'] if fname in thermal_embeddings_dict])

    # Assuming you want to include columns from 1 to 23 (excluding the first column at index 0)
    fau_min_vals = train_df.iloc[:, 1:23].min()
    fau_max_vals = train_df.iloc[:, 1:23].max()
    # Convert to numpy arrays and reshape to (1, 22)
    fau_min_vals = np.array(fau_min_vals).reshape(1, -1)
    fau_max_vals = np.array(fau_max_vals).reshape(1, -1)

    # Calculate min and max for Thermal data
    thermal_min_vals = thermal_samples.min(axis=0)
    thermal_max_vals = thermal_samples.max(axis=0)

    return (fau_min_vals, fau_max_vals), (thermal_min_vals, thermal_max_vals)


# for i, (fau_emb, thermal_emb, labels) in enumerate(dataloader):
#     print(f"Batch {i}: Shapes - FAU: {fau_emb.shape}, Thermal: {thermal_emb.shape}, Labels: {labels.shape}")
#     if i == 10:
#         break
