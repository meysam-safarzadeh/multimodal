import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from tqdm import tqdm
from split_dataset import split_dataset_by_iteration


def initialize_device():
    # Check if CUDA is available
    selected_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {selected_device}')
    return selected_device


def prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    # Create DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs, eval_interval):
    # Training Loop with periodic validation
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        progress_bar = tqdm(train_loader, unit='batch')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy = 100 * correct_predictions / total_samples
            progress_bar.set_postfix(loss=running_loss / total_samples, accuracy=f'{accuracy:.2f}%')

        # Evaluate on the validation set after certain number of epochs
        if (epoch + 1) % eval_interval == 0:
            val_loss, val_accuracy = evaluate_model(model, device, val_loader, criterion)
            print(f'Validation after Epoch {epoch + 1}: Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.2f}%')

    # Final evaluation after all epochs
    val_loss, val_accuracy = evaluate_model(model, device, val_loader, criterion)
    print(f'Final Validation: Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.2f}%')


def evaluate_model(model, device, data_loader, criterion):
    # Validation or Test Evaluation
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    accuracy = 100 * total_correct / total_samples
    return total_loss / total_samples, accuracy


def main():
    device = initialize_device()

    # 1. Prepare and split Dataset based on iteration
    iteration = 0
    train_dataset, val_dataset, test_dataset = split_dataset_by_iteration(iteration)

    # 2. Prepare DataLoaders
    batch_size = 64
    train_loader, val_loader, test_loader = prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    # 3. Load Pre-trained Model
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=5).to(device)

    # 4. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. Train the Model
    num_epochs = 1
    eval_interval = 1  # This means evaluation after every 2 epochs
    train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs, eval_interval)

    # 7. Evaluate on Test Set
    test_loss, test_accuracy = evaluate_model(model, device, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


if __name__ == '__main__':
    main()
