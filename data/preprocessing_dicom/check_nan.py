import os
import nibabel as nib
import numpy as np

def check_nans_in_nii(folder_path):
    nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii') or f.endswith('.nii.gz')]

    if not nii_files:
        print("No .nii or .nii.gz files found in the directory.")
        return

    for filename in nii_files:
        file_path = os.path.join(folder_path, filename)
        try:
            img = nib.load(file_path)
            data = img.get_fdata()

            if np.isnan(data).any():
                print(f"[NaNs FOUND] {filename}")
            else:
                print(f"[OK] {filename}")
        except Exception as e:
            print(f"[ERROR] {filename} - {e}")

if __name__ == "__main__":
    folder = input("Enter the path to the folder containing .nii files: ").strip()
    if os.path.isdir(folder):
        check_nans_in_nii(folder)
    else:
        print("Invalid folder path.")
