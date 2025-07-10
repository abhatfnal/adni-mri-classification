#!/usr/bin/env python
import os
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom, binary_fill_holes
import SimpleITK as sitk
import matplotlib.pyplot as plt
import subprocess
import tempfile

# ————————————————————————————————
# Parse CLI
parser = argparse.ArgumentParser(description="Preprocess one ADNI scan by index")
parser.add_argument('--idx', type=int, required=True,
                    help="Row index in adni_matched_nii_with_labels.csv to process")
args = parser.parse_args()

# Get FSL template
FSLDIR = os.environ.get('FSLDIR')
if not FSLDIR:
    raise RuntimeError("FSLDIR is not set – please `module load fsl` before running.")
template_path = os.path.join(FSLDIR, 'data', 'standard', 'MNI152_T1_1mm_brain.nii.gz')
# ————————————————————————————————

# Configuration
csv_path    = 'adni_matched_nii_with_labels.csv'
output_dir  = './preprocessed_npy'
qa_dir      = './qa_slices'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(qa_dir, exist_ok=True)

df           = pd.read_csv(csv_path)
output_shape = (128, 128, 128)

def bias_correct(nifti_path):
    img  = sitk.ReadImage(nifti_path, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corr = sitk.N4BiasFieldCorrectionImageFilter().Execute(img, mask)
    return sitk.GetArrayFromImage(corr)

def simple_skull_strip(data):
    thresh = data > (0.1 * np.max(data))
    mask   = np.stack([binary_fill_holes(thresh[z]) for z in range(thresh.shape[0])])
    return data * mask

def skull_strip_hd_bet(data_nib, idx):
    with tempfile.TemporaryDirectory() as tmpdir:
        inp       = os.path.join(tmpdir, f'tmp_in_{idx}.nii.gz')
        out_brain = os.path.join(tmpdir, f'brain_{idx}.nii.gz')

        nib.save(data_nib, inp)
        cmd = ['hd-bet', '-i', inp, '-o', out_brain, '-device', 'cuda', '--verbose']
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"\nHD-BET failed idx={idx}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")
            return None
        return nib.load(out_brain).get_fdata().astype(np.float32)

def register_to_template(volume, affine, idx):
    with tempfile.TemporaryDirectory() as tmpdir:
        in_  = os.path.join(tmpdir, f'reg_in_{idx}.nii.gz')
        out_ = os.path.join(tmpdir, f'reg_out_{idx}.nii.gz')

        nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), in_)
        cmd = ['flirt', '-in', in_, '-ref', template_path,
               '-out', out_, '-omat', os.path.join(tmpdir, f'trans_{idx}.mat'),
               '-interp', 'trilinear']
        subprocess.run(cmd, check=True)
        return nib.load(out_).get_fdata().astype(np.float32)

def normalize_and_resize(vol):
    m, s = vol.mean(), vol.std()
    vol  = (vol - m) / s if s > 0 else vol - m
    f    = [o/i for o, i in zip(output_shape, vol.shape)]
    return zoom(vol, f, order=1)

def save_qa_slice(vol, path):
    z = vol.shape[2] // 2
    plt.imshow(vol[:, :, z], cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ————————————————————————————————
# Process exactly one row
i   = args.idx
row = df.iloc[i]

nii_path = row['filepath']
try:
    # 1) Bias correction
    arr     = bias_correct(nii_path)
    orig    = nib.load(nii_path)
    affine  = orig.affine
    nib_img = nib.Nifti1Image(arr.astype(np.float32), affine)

    # 2) Skull strip
    brain = skull_strip_hd_bet(nib_img, i)
    if brain is None:
        brain = simple_skull_strip(arr)

    # 3) Register to MNI
    brain_reg = register_to_template(brain, affine, i)

    # 4) Normalize & resize
    proc = normalize_and_resize(brain_reg)

    # 5) Save .npy
    rid   = row['rid']
    date  = pd.to_datetime(row['scan_date']).strftime('%Y%m%d')
    fname = f"sub-{rid}_date-{date}_i-{i}.npy"
    np.save(os.path.join(output_dir, fname), proc)

    # 6) QA slice
    save_qa_slice(proc, os.path.join(qa_dir, fname.replace('.npy', '.png')))

    print(f"[{i}] Done")

except Exception as e:
    print(f"❌ Failed idx={i} on {nii_path}: {e}")
