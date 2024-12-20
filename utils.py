from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


def class_wise_accuracy(outputs, labels, num_classes):
    """
    Calculate class-wise accuracy using confusion matrix.

    Args:
        outputs (torch.Tensor): The output logits from the model.
        labels (torch.Tensor): The ground-truth labels.
        num_classes (int): The number of classes.

    Returns:
        list: An array containing accuracy for each class.
    """
    _, predicted = torch.max(outputs, 1)
    cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=np.arange(num_classes))
    class_accuracies = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return np.array(class_accuracies)


def under_sampling(df):
    """
    Perform under-sampling on the DataFrame by removing 3/4 of the groups where label is 0
    for each 'sub'.

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The filtered DataFrame after under-sampling.
    """
    # Container for filtered groups
    filtered_df = pd.DataFrame()

    # Group by 'sub' and iterate through each subgroup
    for sub, sub_group in df.groupby('sub'):
        # Identify groups where label is 0
        label_0_group_indices = sub_group[sub_group['label'] == 0].groupby(['trial', 'sweep', 'label']).ngroup()

        # Sample 1/4 of these groups to keep
        groups_to_keep = np.random.choice(label_0_group_indices.unique(),
                                          size=int(len(label_0_group_indices.unique()) * 0.27), replace=False)

        # Filter out the selected groups and groups where label is not 0
        filtered_sub_group = sub_group[(sub_group['label'] != 0) | label_0_group_indices.isin(groups_to_keep)]

        # Append the result to the filtered DataFrame
        filtered_df = pd.concat([filtered_df, filtered_sub_group])

    return filtered_df.reset_index(drop=True)


def plot_accuracy(train_acc, val_acc, save_path):
    """
    Plot training and validation accuracy and save the plot to a file.

    Args:
    train_acc (list): List of training accuracies over epochs.
    val_acc (list): List of validation accuracies over epochs.
    save_path (str): Path to save the plot image.
    """
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, format='jpg')
    plt.close()


def plot_loss(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load a model and optimizer from a checkpoint file.

    Parameters:
    model (torch.nn.Module): The model to load the state into.
    optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    checkpoint_path (str): Path to the checkpoint file.

    Returns:
    tuple: Returns the updated model and optimizer, and the epoch and best validation loss from the checkpoint.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load model and optimizer states
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Extracting additional information if available
    epoch = checkpoint.get('epoch', None)
    best_val_acc = checkpoint.get('best_val_acc', None)

    print(f"Model and optimizer states have been loaded from {checkpoint_path}")
    return model, optimizer, epoch, best_val_acc


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', eps=1e-8):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        logp_target = logp.gather(1, targets.unsqueeze(1)).view(-1)
        pt = logp_target.exp()

        # Adding epsilon for numerical stability
        F_loss = -1 * self.alpha * ((1 - pt) ** self.gamma) * (logp_target + self.eps)

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def prepare_z(z1=None, z2=None, z3=None, labels=None, device=None, modalities=None):
    """
    Prepare the embeddings and labels for training. based on the modalities list order, it returns the embeddings and
    labels in the same order. e.g. if modalities = ['thermal', 'depth'], it returns z1, z2, z3, labels in the same
    order. It assumes z1 is FAU, z2 is thermal, and z3 is depth.
    :param z1: embeddings of FAU
    :param z2: embeddings of thermal
    :param z3: embeddings of depth
    :param labels: labels
    :param device: device to be used
    :param modalities: list of modalities to be used. e.g. ['thermal', 'fau', 'depth']
    :return: prepared embeddings and labels
    """

    modalities_dic = {'fau': None, 'thermal': None, 'depth': None}
    for mod in modalities:
        if mod == 'fau':
            z1 = z1.to(device)
            modalities_dic['fau'] = z1
        if mod == 'thermal':
            z2 = z2.to(device)
            modalities_dic['thermal'] = z2
        if mod == 'depth':
            z3 = z3.to(device)
            modalities_dic['depth'] = z3

    labels = labels.to(device).long()

    if len(modalities) == 2:
        return modalities_dic[modalities[0]], modalities_dic[modalities[1]], None, labels

    elif len(modalities) == 3:
        return modalities_dic[modalities[0]], modalities_dic[modalities[1]], modalities_dic[modalities[2]], labels

    else:
        raise ValueError('modalities should be a list of 2 or 3 modalities')
