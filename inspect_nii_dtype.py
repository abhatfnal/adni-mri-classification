import nibabel as nib
import numpy as np

nii_path = "/project/aereditato/abhat/ADNI/ADNI/002_S_0559/MPR____N3__Scaled/2006-06-27_18_28_33.0/I45126/ADNI_002_S_0559_MR_MPR____N3__Scaled_Br_20070319121214158_S15922_I45126.nii"

try:
    img = nib.load(nii_path)
    data = img.get_fdata()
    print("Shape:", data.shape)
    print("Original dtype:", data.dtype)
    print("Min/Max:", np.min(data), "/", np.max(data))
except Exception as e:
    print("❌ Error:", e)
