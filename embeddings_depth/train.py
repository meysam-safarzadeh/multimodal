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


def main():
    pass


if __name__ == '__main__':
    main()