from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
import os


class MyAutoencoderDataset(Dataset):
    def __init__(self, directory):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
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
