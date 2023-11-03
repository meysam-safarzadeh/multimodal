import os
import shutil

# The directory containing the JPG images
source_dir = "/media/meysam/NewVolume/MintPain_dataset/cropped_face/T"

# The base directory to create the folders 0, 1, 2, 3, and 4
base_dir = "/media/meysam/NewVolume/MintPain_dataset/cropped_face/thermal_classified"

# Ensure the destination directories exist, create if not
for i in range(5):
    dest_dir = os.path.join(base_dir, str(i))
    os.makedirs(dest_dir, exist_ok=True)

# Loop through all files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".jpg") and "Label" in filename:
        # Extract the label number from the filename
        label = filename.split("Label")[-1][0]
        if label.isdigit() and 0 <= int(label) <= 4:
            # Construct the source and destination file paths
            src_file = os.path.join(source_dir, filename)
            dest_dir = os.path.join(base_dir, label)
            dest_file = os.path.join(dest_dir, filename)

            # Move the file
            shutil.move(src_file, dest_file)
            print(f"Moved {filename} to folder {label}")

print("Image sorting complete.")
