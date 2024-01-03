from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def create_train_loader(image_directory, batch_size=64, shuffle=True, num_workers=0):
    # Define the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(image_directory, transform=transform)

    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader