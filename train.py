import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from model import AttentionBottleneckFusion
from torch.utils.data import DataLoader, Dataset
from mint_pain_dataset_creator import create_dataset
from utils import class_wise_accuracy, plot_accuracy, plot_loss, load_checkpoint, FocalLoss


# Set random seed for reproducibility
random_seed = 41
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


def save_checkpoint(state, is_best, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    torch.save(state, os.path.join(checkpoint_folder, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_folder, 'model_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, device, verbose, epoch, numEpochs, batch_size, train_size):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_classes = 5
    all_class_accuracies = []
    for i, (z1, z2, _, labels) in enumerate(train_loader, 0):
        z1, z2, labels = z1.to(device), z2.to(device), labels.to(device).long()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = model(z1, z2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Calculate class-wise accuracy
        class_accuracies = class_wise_accuracy(outputs, labels, num_classes)
        all_class_accuracies.append(class_accuracies)

    train_loss = running_loss / len(train_loader)

    # Average across all classes and batches for class-wise accuracy
    avg_class_accuracy = np.nanmean(np.array(all_class_accuracies), axis=0)
    overall_avg_accuracy = np.mean(avg_class_accuracy) * 100

    # Print average loss and class-wise accuracy for the epoch
    if verbose:
        print('Epoch [%d/%d], Train Loss: %.4f, Average Class Accuracy: %.3f %%' %
              (epoch + 1, numEpochs, train_loss, overall_avg_accuracy))

    return train_loss, overall_avg_accuracy


def val(val_loader, model, criterion, device, verbose, epoch, numEpochs, batch_size, val_size):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_classes = 5
    all_class_accuracies = []

    with torch.no_grad():
        for i, (z1, z2, _,labels) in enumerate(val_loader, 0):
            z1, z2, labels = z1.to(device), z2.to(device), labels.to(device).long()
            outputs = model(z1, z2)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate class-wise accuracy
            class_accuracies = class_wise_accuracy(outputs, labels, num_classes)
            all_class_accuracies.append(class_accuracies)

    val_loss = running_loss / len(val_loader)

    # Average across all classes and batches for class-wise accuracy
    avg_class_accuracy = np.nanmean(np.array(all_class_accuracies), axis=0) * 100
    overall_avg_accuracy = np.mean(avg_class_accuracy)

    if verbose:
        print('Epoch [%d/%d], Validation Loss: %.3f' % (epoch + 1, numEpochs, val_loss))
        print('Validation Class-wise Accuracy:', np.round(avg_class_accuracy, 2))

    return val_loss, overall_avg_accuracy,  avg_class_accuracy


def test(test_loader, model, criterion, device, verbose):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_classes = 5
    all_class_accuracies = []
    with torch.no_grad():
        for z1, z2, labels in test_loader:
            z1, z2, labels = z1.to(device), z2.to(device), labels.to(device)
            outputs = model(z1, z2)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Calculate class-wise accuracy
            class_accuracies = class_wise_accuracy(outputs, labels, num_classes)
            all_class_accuracies.append(class_accuracies)

    test_loss = running_loss / len(test_loader)

    # Calculate class-wise accuracy
    avg_class_accuracy = np.nanmean(np.array(all_class_accuracies), axis=0) * 100
    overall_avg_accuracy = np.mean(avg_class_accuracy)

    if verbose:
        print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss, overall_avg_accuracy))
        print('Tes Class-wise Accuracy:', np.round(avg_class_accuracy, 2))

    return test_loss, overall_avg_accuracy


def main(hidden_dim, num_heads, num_layers, learning_rate, dropout_rate, weight_decay, downsample_method, mode,
         fusion_layers, n_bottlenecks, batch_size, num_epochs, verbose, fold, device, save_model, max_seq_len,
         classification_head, plot, head_layer_sizes):
    """
        Main function for training an Attention-based Bottleneck Fusion model.

        Parameters:
        - hidden_dim: List of hidden dimensions for each modality and after fusion.
        - num_heads: Number of attention heads for each modality and after fusion.
        - num_layers: Number of transformer encoder layers for each modality.
        - learning_rate: Learning rate for the optimizer.
        - dropout_rate: Dropout rate used in the model.
        - weight_decay: Weight decay factor for the optimizer.
        - downsample_method: Method for downsampling (e.g., 'Linear', 'MaxPool').
        - mode: Mode of operation for the final classification layer ('concat' or 'separate').
        - fusion_layers: Number of layers after modality fusion.
        - n_bottlenecks: Number of bottleneck tokens in the model.
        - batch_size: Batch size for training and validation.
        - num_epochs: Number of epochs for training.
        - verbose: Verbosity mode.
        - fold: Fold number for cross-validation.
        - device: Device to use for training and validation.
        - save_model: Whether to save the model or not. If True, the model will be saved in the 'checkpoints' folder.
        - max_seq_len: Maximum sequence length for the input sequences. The length of the sequences + 1 CLS token
        - classification_head: Whether to use a classification head or not. If True, a classification head will be added.
        - plot: Whether to plot the loss and accuracy curves or not. bool = True or False
        - head_layer_sizes: List of hidden layer sizes for the classification head. 3 layers are used by default.
    """
    # Initialize parameters and data
    input_dim = [22, 512]
    num_classes = 5

    # Initialize datasets and dataloaders
    fau_file_path = 'embeddings_fau/FAU_embeddings_with_labels.csv'
    thermal_file_path = 'embeddings_thermal/Thermal_embeddings_and_filenames_new.npz'
    depth_file_path = 'embeddings_depth/Depth_embeddings_and_filenames_new.npz'
    split_file_path = 'cross_validation_split_2.csv'

    # Create the DataLoader
    train_dataset, val_dataset, test_dataset = create_dataset(fau_file_path, thermal_file_path, split_file_path,
                                                              fold, batch_size=batch_size, max_seq_len=max_seq_len,
                                                              depth_file_path=depth_file_path)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = AttentionBottleneckFusion(input_dim, hidden_dim, num_heads, num_layers, fusion_layers, n_bottlenecks,
                                      num_classes, device, max_seq_len+1, mode, dropout_rate,
                                      downsample_method, classification_head, head_layer_sizes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = float(0.0)
    for epoch in range(num_epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device, False,
                                      epoch, num_epochs, batch_size, len(train_dataset))
        val_loss, val_acc, class_wise_acc = val(val_loader, model, criterion, device, False,
                                               epoch, num_epochs, batch_size, len(val_dataset))
        if verbose:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}',
                  f'Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')
            print('Validation Class-wise Accuracy:', np.round(class_wise_acc, 2))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save checkpoints
        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            if save_model:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best)
                print("Checkpoint saved: Epoch {}, Validation Accuracy {}".format(epoch + 1, best_val_acc))

    # Load the best model and test the model based on that
    # model, _, _, _ = load_checkpoint(model, optimizer, 'checkpoints/model_best.pth.tar')
    # test(test_loader, model, criterion, device, True)

    if plot:
        plot_loss(train_losses, val_losses, 'loss_curve.png')
        plot_accuracy(train_accuracies, val_accuracies, 'accuracy_curve.png')

    return train_losses, val_losses, train_accuracies, val_accuracies, best_val_acc


if __name__ == '__main__':
    _, _, _, _, _ = main(hidden_dim=[96, 512, 384], num_heads=[2, 64, 2], num_layers=[2, 3], learning_rate=3e-4,
                         dropout_rate=0.0, weight_decay=0.0, downsample_method='Linear', mode='separate',
                         fusion_layers=2, n_bottlenecks=4, batch_size=64, num_epochs=150, verbose=True, fold=1,
                         device='cuda:1', save_model=False, max_seq_len=40, classification_head=True, plot=True,
                         head_layer_sizes=[352, 112, 48])
