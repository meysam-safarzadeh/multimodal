from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import pandas as pd


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
                                          size=int(len(label_0_group_indices.unique()) * 0.25), replace=False)

        # Filter out the selected groups and groups where label is not 0
        filtered_sub_group = sub_group[(sub_group['label'] != 0) | label_0_group_indices.isin(groups_to_keep)]

        # Append the result to the filtered DataFrame
        filtered_df = pd.concat([filtered_df, filtered_sub_group])

    return filtered_df.reset_index(drop=True)
