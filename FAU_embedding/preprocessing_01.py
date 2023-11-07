import os
import pandas as pd


def sort_key(filename):
    # This assumes the filename pattern is always consistent as per your example
    parts = filename.split('_')
    # Extract the parts of the filename we want to sort by.
    # For example: Sub01, Trial01, Sweep01, Label0, 01.jpg
    sub = parts[2]  # Sub01
    trial = parts[3]  # Trial01
    sweep = parts[4]  # Sweep01
    label = parts[5]  # Label0
    number = parts[6]  # 01.jpg

    # Convert the numeric parts to integers for proper sorting
    # We strip non-numeric parts ('Sub', 'Trial', 'Sweep', 'Label') and file extension
    sub_num = int(sub[3:])
    trial_num = int(trial[5:])
    sweep_num = int(sweep[5:])
    label_num = int(label[5:])
    num = int(number.split('.')[0])

    # Return a tuple that will be used as the sorting key
    return (sub_num, trial_num, sweep_num, label_num, num)


# Replace 'your_directory_path' with the path of your directory
directory_path = '/media/meysam/NewVolume/MintPain_dataset/cropped_face/sub20'

# List to hold file paths
file_paths = []

# Walk through the directory
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.jpg'):  # Only add .jpg files
            # file_path = os.path.join(root, file)
            file_paths.append(file)

# Sort the file paths using the custom sort key
file_paths_sorted = sorted(file_paths, key=lambda f: sort_key(os.path.basename(f)))

# Specify the full path to the CSV file
csv_file_path = '/media/meysam/NewVolume/MintPain_dataset/processed_openface/sub20.csv'  # Replace with your actual file path

# List of columns to keep
columns_to_keep = [
    ' pose_Rx', ' pose_Ry', ' pose_Rz',
    ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
    ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r',
    ' gaze_angle_x', ' gaze_angle_y'
]

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Keep only the specified columns
df = df[columns_to_keep]

# Add a new column at the beginning with the title 'file name'
df.insert(0, 'file name', file_paths_sorted)

# Extract the directory, base file name, and extension
file_dir, file_name = os.path.split(csv_file_path)
base_file_name, file_extension = os.path.splitext(file_name)

# Create the new file name by adding "_new" before the file extension
new_file_name = f"{base_file_name}_new{file_extension}"

# Combine the directory and the new file name
new_csv_file_path = os.path.join(file_dir, new_file_name)

# Save the modified DataFrame to a new CSV file
df.to_csv(new_csv_file_path, index=False)

print(f'The modified CSV has been saved as {new_csv_file_path}')
