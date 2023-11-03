import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Check if CUDA is available
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# 1. Prepare Dataset
dataset_path = "/media/meysam/NewVolume/MintPain_dataset/cropped_face/thermal_classified"
dataset = datasets.ImageFolder(dataset_path, transform=transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]))

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader for each set
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# 2. Load Pre-trained Model
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(dataset.class_to_idx))
model = model.to(device)

# 3. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4. Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    print(f'Epoch [{epoch + 1}/{num_epochs}')
    progress_bar = tqdm(train_loader, unit='batch')
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        accuracy = 100 * correct / total
        progress_bar.set_postfix(loss=running_loss / total, accuracy=f'{accuracy:.2f}%')

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_accuracy = 100 * val_correct / val_total
    print(f'Validation Loss: {val_loss / val_total}, Validation Accuracy: {val_accuracy}%\n')

print('Finished Training')

# 6. (Optional) Test Evaluation
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        test_loss += loss.item()
test_accuracy = 100 * test_correct / test_total
print(f'Test Loss: {test_loss / test_total}, Test Accuracy: {test_accuracy}%')
