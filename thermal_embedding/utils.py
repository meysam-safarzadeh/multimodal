import matplotlib.pyplot as plt
import torch
from datetime import datetime


def plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(f'models/training_validation_metrics_epoch_{current_time}.png')
    plt.close()


def save_best_model(model, current_val_loss=None, current_val_acc=None, best_val_loss=float('inf'), best_val_acc=0.0,
                    model_path_loss='best_model_loss.pth', model_path_acc='best_model_acc.pth'):
    """
    Saves the model if the current validation loss is lower than the best seen so far,
    or if the current validation accuracy is higher than the best seen so far.
    Either the validation loss or accuracy can be used for saving the best model.

    Parameters:
    model (torch.nn.Module): The model to save.
    current_val_loss (float, optional): The current epoch's validation loss.
    current_val_acc (float, optional): The current epoch's validation accuracy.
    best_val_loss (float): The best validation loss seen so far.
    best_val_acc (float): The best validation accuracy seen so far.
    model_path_loss (str): Path to save the model with the best validation loss.
    model_path_acc (str): Path to save the model with the best validation accuracy.

    Returns:
    tuple: Updated best validation loss and best validation accuracy.
    """
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path_loss = f'models/best_model_loss_{current_time}.pth'
    model_path_acc = f'models/best_model_acc_{current_time}.pth'

    # Save the model if it has the best validation loss so far
    if current_val_loss is not None and current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        torch.save(model.state_dict(), model_path_loss)
        print(f"Model saved with lower validation loss: {current_val_loss:.4f}")

    # Save the model if it has the best validation accuracy so far
    if current_val_acc is not None and current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        torch.save(model.state_dict(), model_path_acc)
        print(f"Model saved with higher validation accuracy: {current_val_acc:.2f}%")

    return best_val_loss, best_val_acc
