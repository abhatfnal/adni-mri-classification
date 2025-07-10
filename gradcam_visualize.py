import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from adni_dataset import ADNIDataset
from model_3dcnn_gradcam import GradCAM3DCNN

# ---- Configuration ----
model_path       = "best_3dcnn_gradcam.pth"
csv_file         = "adni_preprocessed_npy_metadata.csv"
idx_to_visualize = 316   # pick any held-out index
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names      = ['MCI', 'AD', 'CN']

# ---- Reproduce train/val split to ensure “unseen” ----
df       = pd.read_csv(csv_file)
labels   = df['diagnosis'].astype(int).tolist()
all_idx  = list(range(len(df)))
train_idx, val_idx = train_test_split(
    all_idx,
    test_size=0.2,
    stratify=labels,
    random_state=42
)
if idx_to_visualize in train_idx:
    raise ValueError(
        f"Index {idx_to_visualize} was in the TRAINING set!  "
        "Please pick an index from the held‐out set."
    )
print(f"✓ Index {idx_to_visualize} is held‐out (not used for training).")

# ---- Load model ----
model = GradCAM3DCNN(num_classes=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# ---- Load sample ----
dataset = ADNIDataset(csv_file)
print(f"Total samples in dataset: {len(dataset)}")
img, true_label = dataset[idx_to_visualize]
img = img.unsqueeze(0).to(device)  # [1,1,D,H,W]

# ---- Forward / probabilities ----
output = model(img)
probs  = torch.softmax(output, dim=1).squeeze().detach().cpu().numpy()
pred_class = int(np.argmax(probs))

print(f"Predicted class: {class_names[pred_class]} ({pred_class}), probabilities:")
for i, p in enumerate(probs):
    print(f"  {class_names[i]:<3} ({i}): {p:.4f}")
print(f"True label: {class_names[true_label]} ({true_label})")

# ---- Grad-CAM computation ----
model.zero_grad()
output[0, pred_class].backward()

# pull out the feature maps & gradients as NumPy
feature_maps = model.feature_maps.detach().cpu().squeeze(0).numpy()  # [C,D,H,W]
grads        = model.feature_maps.grad.detach().cpu().squeeze(0).numpy()   # [C,D,H,W]

# channel‐wise weights: mean gradient over spatial dims
weights = grads.mean(axis=(1,2,3))  # [C]

# build the CAM
cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)  # [D,H,W]
for c, w in enumerate(weights):
    cam += w * feature_maps[c]

# normalize to [0,1]
cam = np.maximum(cam, 0)
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# ---- Plot middle ±2 slices ----
original = img.cpu().squeeze().numpy()  # [D,H,W]
mid      = cam.shape[0] // 2
slices   = range(mid-2, mid+3)

fig, axes = plt.subplots(len(slices), 2, figsize=(8, 12))
for i, z in enumerate(slices):
    orig_slice = original[z, :, :]
    heat_slice = cam[z, :, :]

    axes[i,0].imshow(orig_slice, cmap='gray')
    axes[i,0].set_title(f"Orig Slice {z}")
    axes[i,0].axis('off')

    axes[i,1].imshow(orig_slice, cmap='gray')
    axes[i,1].imshow(heat_slice,  cmap='jet', alpha=0.5)
    axes[i,1].set_title(f"Grad-CAM {z}")
    axes[i,1].axis('off')

plt.tight_layout()
out_fname = f"gradcam_idx{idx_to_visualize}_true{class_names[true_label]}_pred{class_names[pred_class]}.pdf"
plt.savefig(out_fname)
print(f"✅ Grad-CAM visualization saved to {out_fname}")
