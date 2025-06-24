import os
import nibabel as nib
import torchio as tio
import pandas as pd
from tqdm import tqdm

# Load metadata CSV
csv_path = '/home/abhat/ADNI/adni_matched_nii_with_labels.csv'
df = pd.read_csv(csv_path)

# Define preprocessing pipeline
transform = tio.Compose([
    tio.Resample((1, 1, 1)),  # Ensure isotropic spacing if necessary
    tio.Resize((128, 128, 128)),
    tio.ZNormalization()
])

# Process and save
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        original_path = row['filepath']
        image = tio.ScalarImage(original_path)
        preprocessed = transform(image)

        # Save to same folder with 'preprocessed_' prefix
        folder = os.path.dirname(original_path)
        filename = os.path.basename(original_path)
        output_path = os.path.join(folder, f'preprocessed_{filename}')
        preprocessed.save(output_path)

    except Exception as e:
        print(f"❌ Failed to preprocess {row['filepath']}: {e}")

print("\n✅ All matched MRI files preprocessed and saved next to originals.")
