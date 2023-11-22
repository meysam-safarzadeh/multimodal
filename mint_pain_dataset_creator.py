from torch.utils.data import Dataset, DataLoader, Subset
import torch
import numpy as np
import pandas as pd


class MintPainDataset(Dataset):
    def __init__(self, fau_dataframe, thermal_file_path):
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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        indices = self.sequences[list(self.sequences)[idx]]
        data = self.FAU_dataframe.iloc[indices]

        fau_embeddings = self._get_fau_embeddings(data)
        thermal_embeddings = self._get_thermal_embeddings(data['file name'])

        label = torch.tensor(data.iloc[0]['label'], dtype=torch.long)

        return fau_embeddings, thermal_embeddings, label

    def _get_fau_embeddings(self, data):
        fau_features = 22
        fau_embeddings = data.iloc[:, 1:fau_features+1].values
        fau_embeddings = self._pad_embeddings(fau_embeddings, fau_features)
        return torch.tensor(fau_embeddings, dtype=torch.float32)

    def _get_thermal_embeddings(self, filenames):
        thermal_embedding_size = 512
        thermal_embeddings = [self.thermal_embeddings.get(filename, np.zeros(thermal_embedding_size)) for filename in filenames]
        thermal_embeddings = np.array(thermal_embeddings)
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
    tuple: A tuple containing the train, validation, and test DataLoaders.
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

    # Create subsets
    train_dataset = MintPainDataset(train_df, thermal_file_path)
    val_dataset = MintPainDataset(val_df, thermal_file_path)
    test_dataset = MintPainDataset(test_df, thermal_file_path)

    return train_dataset, val_dataset, test_dataset

# for i, (fau_emb, thermal_emb, labels) in enumerate(dataloader):
#     print(f"Batch {i}: Shapes - FAU: {fau_emb.shape}, Thermal: {thermal_emb.shape}, Labels: {labels.shape}")
#     if i == 10:
#         break
