from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from utils import under_sampling


class MintPainDataset(Dataset):
    def __init__(self, fau_dataframe, thermal_file_path, fau_min_max_vals, thermal_min_max_vals, max_seq_len,
                 depth_file_path, depth_min_max_vals):
        """
        Initialize the dataset with FAU FAU_dataframe and thermal embeddings.

        Args:
            fau_dataframe (DataFrame): DataFrame containing FAU embeddings.
            thermal_file_path (str): Path to .npz file with Thermal embeddings.
            fau_min_max_vals (tuple): Tuple containing a tuple, min and max values for FAU data,
                                        and second with min and max values for thermal data.
            thermal_min_max_vals (tuple): Tuple containing a tuple, min and max values for thermal data.
            max_seq_len (int): Maximum sequence length for the FAU embeddings.
            depth_file_path (str): Path to .npz file with Depth embeddings.
            depth_min_max_vals (tuple): Tuple containing a tuple, min and max values for depth data.
        """

        self.max_seq_len = max_seq_len

        # FAU embeddings
        self.FAU_dataframe = fau_dataframe
        self.sequences = fau_dataframe.groupby(['sub', 'trial', 'sweep', 'label']).groups

        # Thermal embeddings
        thermal_data = np.load(thermal_file_path)
        self.thermal_embeddings = {filename: embedding for filename, embedding in zip(thermal_data['filenames'], thermal_data['embeddings'])}

        # Depth embeddings
        depth_data = np.load(depth_file_path)
        self.depth_embeddings = {filename: embedding for filename, embedding in zip(depth_data['filenames'], depth_data['embeddings'])}

        # Min and max values for each modality
        self.fau_min_vals, self.fau_max_vals = fau_min_max_vals
        self.thermal_min_vals, self.thermal_max_vals = thermal_min_max_vals
        self.depth_min_vals, self.depth_max_vals = depth_min_max_vals

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        indices = self.sequences[list(self.sequences)[idx]]
        fau_data = self.FAU_dataframe.iloc[indices]

        fau_embeddings = self._get_fau_embeddings(fau_data)
        thermal_embeddings = self._get_thermal_embeddings(fau_data['file name'])
        depth_embeddings = self._get_depth_embeddings(fau_data['file name'])

        label = torch.tensor(fau_data.iloc[0]['label'], dtype=torch.long)

        return fau_embeddings, thermal_embeddings, depth_embeddings, label

    def _get_fau_embeddings(self, data):
        fau_features = 22
        fau_embeddings = data.iloc[:, 1:fau_features+1].values
        # Apply min-max normalization and scale to -1 to 1
        fau_embeddings = 2 * ((fau_embeddings - self.fau_min_vals) / (self.fau_max_vals - self.fau_min_vals)) - 1
        fau_embeddings = self._pad_embeddings(fau_embeddings)
        return torch.tensor(fau_embeddings, dtype=torch.float32)

    def _get_thermal_embeddings(self, filenames):
        thermal_embedding_size = 512
        thermal_embeddings = [self.thermal_embeddings.get(filename, np.zeros(thermal_embedding_size)) for filename in filenames]
        thermal_embeddings = np.array(thermal_embeddings)
        # Apply min-max normalization and scale to -1 to 1
        thermal_embeddings = 2 * ((thermal_embeddings - self.thermal_min_vals) / (
                    self.thermal_max_vals - self.thermal_min_vals)) - 1
        thermal_embeddings = self._pad_embeddings(thermal_embeddings, axis=0)
        return torch.tensor(thermal_embeddings, dtype=torch.float32)

    def _get_depth_embeddings(self, filenames):
        depth_embedding_size = 128
        depth_embeddings = [self.depth_embeddings.get(filename, np.zeros(depth_embedding_size)) for filename in filenames]
        depth_embeddings = np.array(depth_embeddings)
        # Apply min-max normalization and scale to -1 to 1
        depth_embeddings = 2 * ((depth_embeddings - self.depth_min_vals) / (
                    self.depth_max_vals - self.depth_min_vals)) - 1
        depth_embeddings = self._pad_embeddings(depth_embeddings, axis=0)
        return torch.tensor(depth_embeddings, dtype=torch.float32)

    def _pad_embeddings(self, embeddings, axis=0):
        # Pad or truncate embeddings to max_seq_len
        max_samples = self.max_seq_len
        if embeddings.shape[axis] < max_samples:
            # print(f"Padding embeddings from shape {embeddings.shape} to {max_samples} samples")
            padding_shape = list(embeddings.shape)
            padding_shape[axis] = max_samples - embeddings.shape[axis]
            padding = np.zeros(padding_shape)
            embeddings = np.concatenate((embeddings, padding), axis=axis)
            # print(f"New embeddings shape: {embeddings.shape}")
        elif embeddings.shape[axis] > max_samples:
            # print(f"Truncating embeddings from shape {embeddings.shape} to {max_samples} samples")
            embeddings = embeddings[:max_samples, :]
            # print(f"New embeddings shape: {embeddings.shape}")
        return embeddings


def create_dataset(fau_file_path, thermal_file_path, split_file_path, iteration, batch_size, max_seq_len,
                   depth_file_path, sub_independent):
    """
    Create dataset for the FAU and thermal embeddings' dataset, split into train, validation, and test sets for
    the given iteration.

    Args:
    fau_file_path (str): Path to the CSV file containing FAU embeddings.
    thermal_file_path (str): Path to the NPZ file containing thermal embeddings.
    split_file_path (str): Path to the CSV file containing split information.
    iteration (int): The iteration number to select the split.
    batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.
    max_seq_len (int): Maximum sequence length for the FAU embeddings. Defaults to 100.
    depth_file_path (str): Path to the NPZ file containing depth embeddings.

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

    # split the dataset into train and validation sets separately for each subject
    train_df, val_df= split_dataframe(df) if sub_independent else (train_df, val_df)

    # Get min and max values for each modality
    fau_min_max_vals, thermal_min_max_vals , depth_min_max_vals = get_min_max_for_each_modality(train_df,
                                                                                                thermal_file_path,
                                                                                                depth_file_path)

    # Under-sample the training set to balance the classes
    train_df_undersampled = under_sampling(train_df)

    # Create subsets
    train_dataset = MintPainDataset(train_df_undersampled, thermal_file_path, fau_min_max_vals,
                                    thermal_min_max_vals, max_seq_len, depth_file_path, depth_min_max_vals)
    val_dataset = MintPainDataset(val_df, thermal_file_path, fau_min_max_vals,
                                  thermal_min_max_vals, max_seq_len, depth_file_path, depth_min_max_vals)
    test_dataset = MintPainDataset(test_df, thermal_file_path, fau_min_max_vals,
                                   thermal_min_max_vals, max_seq_len, depth_file_path, depth_min_max_vals)

    return train_dataset, val_dataset, test_dataset


def get_min_max_for_each_modality(train_df, thermal_file_path, depth_file_path):
    """
    Extract corresponding thermal samples based on 'file name' and calculate min-max for each modality.

    Args:
        train_df (DataFrame): DataFrame containing FAU embeddings and 'file name' column.
        thermal_file_path (str): Path to the NPZ file containing thermal embeddings.
        depth_file_path (str): Path to the NPZ file containing depth embeddings.
    Returns:
        tuple of tuples: A tuple containing two tuples, first with min and max values for FAU data,
                         and second with min and max values for thermal data.
    """
    # Load thermal data
    thermal_data = np.load(thermal_file_path)
    thermal_embeddings_dict = {filename: embedding for filename, embedding in zip(thermal_data['filenames'], thermal_data['embeddings'])}
    # Extract corresponding thermal samples
    thermal_samples = np.array([thermal_embeddings_dict[fname] for fname in train_df['file name'] if fname in thermal_embeddings_dict])

    # Load depth data
    depth_data = np.load(depth_file_path)
    depth_embeddings_dict = {filename: embedding for filename, embedding in zip(depth_data['filenames'], depth_data['embeddings'])}
    # Extract corresponding depth samples
    depth_samples = np.array([depth_embeddings_dict[fname] for fname in train_df['file name'] if fname in depth_embeddings_dict])

    # Assuming you want to include columns from 1 to 23 (excluding the first column at index 0)
    fau_min_vals = train_df.iloc[:, 1:23].min()
    fau_max_vals = train_df.iloc[:, 1:23].max()
    # Convert to numpy arrays and reshape to (1, 22)
    fau_min_vals = np.array(fau_min_vals).reshape(1, -1)
    fau_max_vals = np.array(fau_max_vals).reshape(1, -1)

    # Calculate min and max for Thermal data
    thermal_min_vals = thermal_samples.min(axis=0)
    thermal_max_vals = thermal_samples.max(axis=0)

    # Calculate min and max for Depth data
    depth_min_vals = depth_samples.min(axis=0)
    depth_max_vals = depth_samples.max(axis=0)

    return (fau_min_vals, fau_max_vals), (thermal_min_vals, thermal_max_vals), (depth_min_vals, depth_max_vals)


def split_dataframe(df):
    """
    Split the DataFrame into 90% training and 10% validation.
    :param df: The input DataFrame.
    :return: df_val, df_train: The validation and training DataFrames.
    """
    df_val = pd.DataFrame()
    df_train = pd.DataFrame()

    # Iterate over each 'sub'
    for sub, sub_group in df.groupby('sub'):
        # Iterate over each label from 0 to 4
        for label in range(5):
            # Identify groups where the current label is present
            label_group_indices = sub_group[sub_group['label'] == label].groupby(['trial', 'sweep', 'label']).ngroup()

            # Calculate 10% of these groups.
            ten_percent_size = int(np.ceil(len(label_group_indices.unique()) * 0.1))

            # Randomly select 10% of these groups
            groups_10_percent = np.random.choice(label_group_indices.unique(), size=ten_percent_size, replace=False)

            # Separate 10% and 90% groups
            group_10_percent_df = sub_group[(sub_group['label'] == label) & label_group_indices.isin(groups_10_percent)]
            group_90_percent_df = sub_group[(sub_group['label'] == label) & ~label_group_indices.isin(groups_10_percent)]

            # Append to the respective DataFrames
            df_val = pd.concat([df_val, group_10_percent_df])
            df_train = pd.concat([df_train, group_90_percent_df])

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)
