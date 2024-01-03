import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from split_dataset import split_dataset_by_iteration
from utils import plot_and_save_metrics, save_best_model


def initialize_device():
    # Check if CUDA is available
    selected_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {selected_device}')
    return selected_device


def prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    # Retrieve labels from the original dataset
    original_labels = train_dataset.dataset.targets
    # Now you need to get the labels for the subset
    subset_indices = train_dataset.indices
    subset_labels = torch.tensor([original_labels[idx] for idx in subset_indices], dtype=torch.long)

    # Count the number of occurrences of each class in the target vector
    class_count = torch.bincount(subset_labels)
    # Create weights for each sample based on the class count
    weights = 1. / class_count[subset_labels]
    # Create a sampler with the weights
    sampler = WeightedRandomSampler(weights, len(weights))

    # Create DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs, eval_interval):
    # Initialize lists to keep track of metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0

    num_classes = 5  # Adjust this if you have a different number of classes
    class_correct = [0 for i in range(num_classes)]
    class_total = [0 for i in range(num_classes)]

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

            # Calculate per-class correct predictions
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

            # Calculate running accuracy
            accuracy = 100 * correct_predictions / total_samples
            progress_bar.set_postfix(loss=running_loss / total_samples, accuracy=f'{accuracy:.2f}%')

        # Calculate average accuracy per class and overall average
        class_accuracies = [round((class_correct[i] / class_total[i]) * 100, 2) if class_total[i] > 0 else 0 for i in
                            range(num_classes)]
        average_accuracy = sum(class_accuracies) / num_classes

        # Append the average loss and accuracy for this epoch to the lists
        train_losses.append(running_loss / total_samples)
        train_accuracies.append(average_accuracy)

        print(f'Training Overall Accuracy: {average_accuracy:.2f}%, '
              f'Class Accuracies: {class_accuracies}')

        # Evaluate on the validation set after certain number of epochs
        if (epoch + 1) % eval_interval == 0:
            val_loss, val_accuracy, val_class_accuracies = evaluate_model(model, device, val_loader, criterion)
            print(f'Validation Loss: {val_loss:.4f}, '
                  f'Overall Accuracy: {val_accuracy:.2f}%, '
                  f'Class Accuracies: {val_class_accuracies}')
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Save the model if it has the best validation accuracy so far
            best_val_loss, best_val_acc = save_best_model(
                model,
                current_val_loss=None,
                current_val_acc=val_accuracy,
                best_val_loss=None,
                best_val_acc=best_val_acc
            )

        # Plot and save the metrics after each epoch
        plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch + 1)

    # Final evaluation after all epochs
    val_loss, val_accuracy, val_class_accuracies = evaluate_model(model, device, val_loader, criterion)
    print(f'Validation Loss: {val_loss:.4f}, '
          f'Overall Accuracy: {val_accuracy:.2f}%, '
          f'Class Accuracies: {val_class_accuracies}')


def evaluate_model(model, device, data_loader, criterion):
    # Validation or Test Evaluation
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    num_classes = 5  # Adjust if you have a different number of classes
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate per-class correct predictions
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # Calculate per-class accuracy and round to two decimals
    class_accuracies = [round((class_correct[i] / class_total[i]) * 100, 2) if class_total[i] > 0 else 0 for i in
                        range(num_classes)]
    # Calculate the average accuracy across all classes and round to two decimals
    average_accuracy = round(sum(class_accuracies) / num_classes, 2)
    overall_accuracy = 100 * total_correct / total_samples  # Overall accuracy

    # Return the total loss, overall accuracy, and class accuracies
    return total_loss / total_samples, average_accuracy, class_accuracies


def main():
    device = initialize_device()

    # 1. Prepare and split Dataset based on iteration
    iteration = 0
    train_dataset, val_dataset, test_dataset = split_dataset_by_iteration(iteration)

    # 2. Prepare DataLoaders
    batch_size = 128
    train_loader, val_loader, test_loader = prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    # 3. Load Pre-trained Model
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=5, dropout_prob=0.6).to(device)

    # 4. Define Loss Function and Optimizer
    weights = torch.tensor([0.5, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. Train the Model
    num_epochs = 25
    eval_interval = 1  # This means evaluation after every 2 epochs
    train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs, eval_interval)

    # 7. Evaluate on Test Set
    # test_loss, test_accuracy = evaluate_model(model, device, test_loader, criterion)
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


if __name__ == '__main__':
    main()
