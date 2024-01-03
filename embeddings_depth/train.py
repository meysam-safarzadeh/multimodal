import os
import shutil
import torch
from model import Autoencoder


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
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def main(use_batch_norm=True, use_cuda=True, plot_loss=False, num_epochs=10, save_checkpoint_flag=True):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # Initialize model
    model = Autoencoder(use_batch_norm=use_batch_norm).to(device)

    # Other setup (data loader, loss function, optimizer) remains the same

    # Training loop
    epoch_losses = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')
        epoch_losses.append(train_loss)

        # Save checkpoint
        if save_checkpoint_flag:
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict()}, False, 'your_checkpoint_dir')

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
    main(use_batch_norm=True, use_cuda=True, plot_loss=True, num_epochs=5, save_checkpoint_flag=True)
