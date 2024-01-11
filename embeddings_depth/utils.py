from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.utils import save_image


def save_images(img, name):
    """
    Save the decoded/generated image
    Args:
        img (tensor): The image tensors to save.
        name (str): Path and name of the file to save.
    """
    # Extract directory name from the file path
    directory = os.path.dirname(name)

    # Create directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_image(img, name)


class AutoencoderDataset(Dataset):
    def __init__(self, directory, target_size=(85, 85)):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.target_size = target_size
        self.directory = directory
        self.transform = transforms.Compose([
            transforms.CenterCrop(size=self.target_size),
            transforms.ToTensor()]) # ToTensor() scales the image pixel values to [0, 1]
        self.image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        # In an autoencoder, the input and output are usually the same
        return image, image


# Example usage
# dataset = MyAutoencoderDataset(directory='/media/meysam/NewVolume/MintPain_dataset/cropped_face/D')
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
