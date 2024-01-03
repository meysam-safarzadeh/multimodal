import os
import glob
import numpy as np


def count_files_in_folder(folder_path):
    file_count = 0
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count


def find_and_count_files(base_path, target_folder):
    folder_paths = glob.glob(os.path.join(base_path, '**', target_folder), recursive=True)
    num_files_list = []
    for folder_path in folder_paths:
        num_files = count_files_in_folder(folder_path)
        num_files_list.append(num_files)
        print(f'Number of files in "{folder_path}": {num_files}')

    if num_files_list:
        print(f'Maximum number of files in a folder: {np.max(num_files_list)}')
        print(f'Minimum number of files in a folder: {np.min(num_files_list)}')
        print(f'Standard deviation of number of files: {np.std(num_files_list):.2f}')
        print(f'Mean number of files: {np.mean(num_files_list):.2f}')
        print(f'Median number of files: {np.median(num_files_list):.2f}')
        print(np.sort(num_files_list)[-10:])
    else:
        print('No "T" folders found.')


if __name__ == '__main__':
    base_path = '/media/meysam/NewVolume/MintPain_dataset/data'
    target_folder = 'RGB'
    find_and_count_files(base_path, target_folder)
