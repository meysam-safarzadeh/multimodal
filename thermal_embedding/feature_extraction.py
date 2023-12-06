import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np


def main():
    # Paths and Device Configuration
    model_path = 'model/best_model_acc_2023-11-09_23-31-44.pth'
    image_folder = '/media/meysam/NewVolume/MintPain_dataset/cropped_face/thermal_classified'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print("Loading model...")
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=5, dropout_prob=0.25).to(device)
    # model.load_state_dict(torch.load(model_path))
    # print(model)
    model.eval()

    # Modified Model for Embedding Extraction
    modified_model = ModifiedModel(model)
    print("Model modified to extract embeddings.")

    # Data Preparation
    val_image_transforms = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = datasets.ImageFolder(image_folder, transform=val_image_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12)
    print(f"Loaded {len(dataset)} images from {image_folder}")

    # Extract and Save Embeddings along with filenames
    embeddings = []
    filenames = []
    print("Extracting embeddings...")
    with torch.no_grad():
        for idx, (images, _) in enumerate(dataloader):
            # Get the file path
            filepath = dataset.imgs[idx][0]  # or dataset.samples[idx][0]
            filename = os.path.basename(filepath)
            # print(filename)

            # Process the image
            images = images.to(device)
            embedding = modified_model(images)
            # print(embedding.shape)

            # Store the results
            embeddings.append(embedding.cpu().numpy())
            filenames.append(filename)

            if idx % 100 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} images")

    # Save the embeddings and filenames in a single NumPy .npz file
    np.savez_compressed('Thermal_embeddings_and_filenames_new.npz', embeddings=np.concatenate(embeddings, axis=0),
                        filenames=filenames)

    print("Embeddings and filenames saved in a single NumPy file.")


class ModifiedModel(torch.nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        x = self.original_model.conv2d_1a(x)
        x = self.original_model.conv2d_2a(x)
        x = self.original_model.conv2d_2b(x)
        x = self.original_model.maxpool_3a(x)
        x = self.original_model.conv2d_3b(x)
        x = self.original_model.conv2d_4a(x)
        x = self.original_model.conv2d_4b(x)
        x = self.original_model.repeat_1(x)
        x = self.original_model.mixed_6a(x)
        x = self.original_model.repeat_2(x)
        x = self.original_model.mixed_7a(x)
        x = self.original_model.repeat_3(x)
        x = self.original_model.block8(x)
        x = self.original_model.avgpool_1a(x)
        # x = self.original_model.dropout(x)
        embeddings = self.original_model.last_linear(x.view(x.shape[0], -1))
        return embeddings


class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = os.listdir(image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == "__main__":
    main()