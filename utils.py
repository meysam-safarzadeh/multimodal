from sklearn.metrics import confusion_matrix
import numpy as np
import torch


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
