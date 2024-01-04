import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from model import Autoencoder
from utils import AutoencoderDataset


def main():
    # Paths and Device Configuration
    model_path = 'checkpoints/best_model.pth.tar'
    image_folder = '/media/meysam/NewVolume/MintPain_dataset/cropped_face/D'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print("Loading model...")
    model = Autoencoder(use_batch_norm=True).to(device)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    # print(model)
    model.eval()

    # Data Preparation
    # Note: Keep the batch size to 1 to extract embeddings for each image separately
    dataset = AutoencoderDataset(image_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, drop_last=False)
    print(f"Loaded {len(dataset)} images from {image_folder}")

    # Extract and Save Embeddings along with filenames
    embeddings = []
    filenames = []
    print("Extracting embeddings...")
    with torch.no_grad():
        for idx, (images, _) in enumerate(dataloader):
            # Get the file path
            filename = dataset.image_files[idx]  # or dataset.samples[idx][0]

            # Process the image
            images = images.to(device)
            _, embedding = model(images)
            # print(embedding.shape)

            # Store the results
            embeddings.append(embedding.cpu().numpy())
            filenames.append(filename)

            if idx % 100 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} images")

    # Save the embeddings and filenames in a single NumPy .npz file
    np.savez_compressed('Depth_embeddings_and_filenames_new.npz', embeddings=np.concatenate(embeddings, axis=0),
                        filenames=filenames)

    print("Embeddings and filenames saved in a single NumPy file.")


if __name__ == '__main__':
    main()