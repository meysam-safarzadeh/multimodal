from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd


class MintPainDataset(Dataset):
    def __init__(self, dataframe, npz_file_path):
        """
        Initialize the dataset with FAU dataframe and thermal embeddings.

        Args:
            dataframe (DataFrame): DataFrame containing FAU data.
            npz_file_path (str): Path to .npz file with thermal embeddings.
        """
        self.dataframe = dataframe
        self.groups = dataframe.groupby(['sub', 'trial', 'sweep', 'label']).groups

        npz_data = np.load(npz_file_path)
        self.thermal_embeddings = {filename: embedding for filename, embedding in zip(npz_data['filenames'], npz_data['embeddings'])}

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indices = self.groups[list(self.groups)[idx]]
        data = self.dataframe.iloc[indices]

        fau_embeddings = self._get_fau_embeddings(data)
        thermal_embeddings = self._get_thermal_embeddings(data['file name'])

        label = torch.tensor(data.iloc[0]['label'], dtype=torch.long)

        return fau_embeddings, thermal_embeddings, label

    def _get_fau_embeddings(self, data):
        FAU_FEATURES = 22
        fau_embeddings = data.iloc[:, 1:FAU_FEATURES+1].values
        fau_embeddings = self._pad_embeddings(fau_embeddings, FAU_FEATURES)
        return torch.tensor(fau_embeddings, dtype=torch.float32)

    def _get_thermal_embeddings(self, filenames):
        THERMAL_EMBEDDING_SIZE = 512
        thermal_embeddings = [self.thermal_embeddings.get(filename, np.zeros(THERMAL_EMBEDDING_SIZE)) for filename in filenames]
        thermal_embeddings = np.array(thermal_embeddings)
        thermal_embeddings = self._pad_embeddings(thermal_embeddings, THERMAL_EMBEDDING_SIZE, axis=1)
        return torch.tensor(thermal_embeddings, dtype=torch.float32)

    def _pad_embeddings(self, embeddings, num_features, axis=1):
        MAX_SAMPLES = 7
        if embeddings.shape[axis] < MAX_SAMPLES:
            print(f"Padding embeddings from shape {embeddings.shape} to {MAX_SAMPLES} samples")
            padding_shape = list(embeddings.shape)
            padding_shape[axis] = MAX_SAMPLES - embeddings.shape[axis]
            padding = np.zeros(padding_shape)
            embeddings = np.concatenate((embeddings, padding), axis=axis)
            print(f"New embeddings shape: {embeddings.shape}")
        return embeddings


def create_dataloader(fau_file_path, npz_file_path, batch_size=64, shuffle=True):
    """
    Create a DataLoader for the FAU and thermal embeddings' dataset.

    Args:
    fau_file_path (str): Path to the CSV file containing FAU embeddings.
    npz_file_path (str): Path to the NPZ file containing thermal embeddings.
    batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.
    shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
    DataLoader: The DataLoader for the combined dataset.
    """
    df = pd.read_csv(fau_file_path)
    dataset = MintPainDataset(df, npz_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# for i, (fau_emb, thermal_emb, labels) in enumerate(dataloader):
#     print(f"Batch {i}: Shapes - FAU: {fau_emb.shape}, Thermal: {thermal_emb.shape}, Labels: {labels.shape}")
#     if i == 10:
#         break
