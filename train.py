import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from model import AttentionBottleneckFusion
# from your_dataset import YourDataset  # Replace with your actual dataset import

def plot_loss(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)

def save_checkpoint(state, is_best, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    torch.save(state, os.path.join(checkpoint_folder, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_folder, 'model_best.pth.tar'))

def train(train_loader, model, criterion, optimizer, device, verbose, epoch, numEpochs, batch_size, train_size):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if verbose and i % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    return running_loss / (train_size // batch_size)

def val(val_loader, model, criterion, device, verbose, epoch, numEpochs, batch_size, val_size):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / (val_size // batch_size)

def test(test_loader, model, criterion, device, verbose):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))

def main():
    # Initialize parameters and data
    input_dim = 512
    hidden_dim = 2048
    num_heads = 8
    num_layers = [6, 4]
    T = 5
    Lf = 3
    num_classes = 5
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize datasets and data loaders
    train_dataset = YourDataset('train')  # Replace with actual initialization
    val_dataset = YourDataset('val')  # Replace with actual initialization
    test_dataset = YourDataset('test')  # Replace with actual initialization

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model, loss function, and optimizer
    model = AttentionBottleneckFusion(input_dim, hidden_dim, num_heads, num_layers, Lf, T, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(train_loader, model, criterion, optimizer, device, True, epoch, num_epochs, batch_size, len(train_dataset))
        val_loss = val(val_loader, model, criterion, device, True, epoch, num_epochs, batch_size, len(val_dataset))
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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
    test(test_loader, model, criterion, device, True)

    # Plot loss curves
    plot_loss(train_losses, val_losses, 'loss_curve.png')

if __name__ == '__main__':
    main()
