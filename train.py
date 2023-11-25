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
from utils import class_wise_accuracy, plot_accuracy, plot_loss


class RandomDataset(Dataset):
    def __init__(self, size, sequence_length, input_dim, num_classes=5):
        self.len = size
        self.data1 = torch.randn(size, sequence_length, input_dim[0])
        self.data2 = torch.randn(size, sequence_length, input_dim[1])
        self.labels = torch.randint(0, num_classes, (size,))

    def __getitem__(self, index):
        return self.data1[index], self.data2[index], self.labels[index]

    def __len__(self):
        return self.len


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
    for i, (z1, z2, labels) in enumerate(train_loader, 0):
        z1, z2, labels = z1.to(device), z2.to(device), labels.to(device).long()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        _, _, _, outputs = model(z1, z2)
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
        for i, (z1, z2, labels) in enumerate(val_loader, 0):
            z1, z2, labels = z1.to(device), z2.to(device), labels.to(device).long()
            _, _, _, outputs = model(z1, z2)
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
    with torch.no_grad():
        for z1, z2, labels in test_loader:
            z1, z2, labels = z1.to(device), z2.to(device), labels.to(device)
            _, _, _, outputs = model(z1, z2)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    if verbose:
        print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss, accuracy))

    return test_loss, accuracy


def main():
    # Initialize parameters and data
    verbose = False # Set to True to print inside the train and val functions
    input_dim = [22, 512]
    hidden_dim = [88, 1024, 256]  # Hidden dimension for modality 1 and modality 2 and after fusion respectively
    num_heads = [2, 16, 2]  # Number of heads for modality 1 and modality 2 and after fusion respectively
    num_layers = [2, 4]  # Number of transformer encoder layers for modality 1 and modality 2 respectively
    B = 3  # Number of bottleneck tokens
    Lf = 4  # Number of layers after fusion
    num_classes = 5
    batch_size = 64
    sequence_length = 7
    learning_rate = 1e-4
    num_epochs = 100
    iteration = 1
    mode = 'concat'  # Choose 'concat' or 'separate' for the last classification layer
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    # Initialize datasets and dataloaders
    # Paths to your files
    fau_file_path = 'FAU_embedding/uniform_sampled_FAU_embeddings.csv'
    thermal_file_path = 'thermal_embedding/Thermal_embeddings_and_filenames.npz'
    split_file_path = 'cross_validation_split.csv'

    # Create the DataLoader
    train_dataset, val_dataset, test_dataset = create_dataset(fau_file_path, thermal_file_path, split_file_path, iteration, batch_size=batch_size)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = AttentionBottleneckFusion(input_dim, hidden_dim, num_heads, num_layers, Lf, B, num_classes, device,
                                      mode=mode).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device, verbose, epoch, num_epochs, batch_size,
                           len(train_dataset))
        val_loss, val_acc, class_wise_acc = val(val_loader, model, criterion, device, verbose, epoch, num_epochs, batch_size, len(val_dataset))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}',
              f'Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')
        print('Validation Class-wise Accuracy:', np.round(class_wise_acc, 2))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save checkpoints
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    # Test the model
    # test(test_loader, model, criterion, device, True)

    # Plot loss & acc curves
    plot_loss(train_losses, val_losses, 'loss_curve.png')
    plot_accuracy(train_accuracies, val_accuracies, 'accuracy_curve.png')


if __name__ == '__main__':
    main()