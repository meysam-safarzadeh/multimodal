from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


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