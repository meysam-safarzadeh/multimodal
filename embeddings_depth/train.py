import os
import shutil
import torch
from torch import optim
from model import Autoencoder
import torch.nn as nn
from utils import MyAutoencoderDataset, save_images
from torch.utils.data import DataLoader


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        # Assuming your dataset returns images and targets
        images, _ = batch

        # Move data to device
        images = images.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, images)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader), outputs, images


def main(use_batch_norm=True, device="cuda:1", plot_loss=False, num_epochs=10, save_checkpoint_flag=True,
         lr=None, im_directory=None, batch_size=None, save_output_images=None):
    # Set up device
    device = torch.device(device)

    # Initialize model
    model = Autoencoder(use_batch_norm=use_batch_norm).to(device)

    # Other setup (data loader, loss function, optimizer) remains the same
    train_set = MyAutoencoderDataset(directory=im_directory)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)

    # the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    epoch_losses = []
    for epoch in range(num_epochs):
        train_loss, outputs, inputs = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')
        epoch_losses.append(train_loss)

        # Save some of the input images and generated images from the last batch
        if save_output_images:
            save_images(outputs.cpu().data, name=f'./Saved_Images/inputs_epoch_{epoch}.png')
            save_images(outputs.cpu().data, name=f'./Saved_Images/generated_epoch_{epoch}.png')

        # Save checkpoint
        if save_checkpoint_flag:
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict()}, False, checkpoint_dir='checkpoints')

    # Plotting the training losses
    if plot_loss:
        import matplotlib.pyplot as plt
        plt.plot(range(1, num_epochs+1), epoch_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.show()


# Example usage
if __name__ == '__main__':
    main(use_batch_norm=True, device="cuda:1", plot_loss=True, num_epochs=5, save_checkpoint_flag=False,
         lr=1e-4, im_directory='/media/meysam/NewVolume/MintPain_dataset/cropped_face/D',
         batch_size=8, save_output_images=True)

