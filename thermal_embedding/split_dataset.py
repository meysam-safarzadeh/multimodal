import csv
import os
from torchvision import datasets, transforms
from torch.utils.data import Subset

# Constants
DATASET_PATH = '/media/meysam/NewVolume/MintPain_dataset/cropped_face/thermal_classified'
CSV_FILE_PATH = '/home/meysam/NursingSchool/code/cross_validation_split.csv'

# Image transformation
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def read_indices_from_csv(csv_file, iteration):
    with open(csv_file, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for i, row in enumerate(csv_reader):
            if i == iteration:
                training_indices = list(map(int, row['Training'].split(', ')))
                validation_indices = list(map(int, row['Validation'].split(', ')))
                test_indices = list(map(int, row['Test'].split(', ')))
                return training_indices, validation_indices, test_indices
    raise ValueError(f"No data found for iteration {iteration}")


def get_indices(imgs, number_list):
    return [idx for idx, (file_path, _) in enumerate(imgs)
            if any(f"Sub{str(number).zfill(2)}" in os.path.basename(file_path) for number in number_list)]


def split_dataset_by_indices(dataset_path, transform, training_indices, validation_indices, test_indices):
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    train_indices = get_indices(dataset.imgs, training_indices)
    valid_indices = get_indices(dataset.imgs, validation_indices)
    test_indices = get_indices(dataset.imgs, test_indices)
    return Subset(dataset, train_indices), Subset(dataset, valid_indices), Subset(dataset, test_indices)


def split_dataset_by_iteration(iteration):
    training_indices, validation_indices, test_indices = read_indices_from_csv(CSV_FILE_PATH, iteration)
    return split_dataset_by_indices(DATASET_PATH, IMAGE_TRANSFORMS, training_indices, validation_indices, test_indices)