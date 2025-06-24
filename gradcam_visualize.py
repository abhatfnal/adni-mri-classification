import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from adni_dataset import ADNIDataset
from model_3dcnn_gradcam import GradCAM3DCNN  # model with conv3 hook

# ---- Configuration ----
model_path = "baseline_3dcnn.pth"
csv_file = "adni_preprocessed_metadata.csv"
idx_to_visualize = 0  # index of the sample in test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Model ----
model = GradCAM3DCNN(num_classes=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ---- Load Dataset and Sample ----
dataset = ADNIDataset(csv_file)
img, label = dataset[idx_to_visualize]
img = img.unsqueeze(0).to(device)  # add batch dimension

# ---- Forward Pass ----
output = model(img)
pred_class = torch.argmax(output, dim=1).item()

# ---- Grad-CAM Calculation ----
model.zero_grad()
output[0, pred_class].backward()

# Feature maps and gradients
feature_maps = model.feature_maps.detach().cpu().squeeze(0)  # shape: [C, D, H, W]
grads = model.conv3.weight.grad.cpu().mean(dim=[1, 2, 3, 4])  # [C]

# Weighted combination of feature maps
cam = torch.zeros_like(feature_maps[0])
for i in range(len(grads)):
    cam += grads[i] * feature_maps[i]

# ---- Normalize and Extract Slice ----
cam = cam.numpy()
cam = np.maximum(cam, 0)  # ReLU
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

mid_slice = cam.shape[1] // 2
heatmap = cam[:, mid_slice, :]  # Axial view (Z = middle slice)

# Get original MRI slice
original_img = img.cpu().squeeze().numpy()  # shape: [128, 128, 128]
original_slice = original_img[:, mid_slice, :]  # same view as CAM

# ---- Plot and Save ----
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_slice, cmap='gray')
plt.title("Original MRI Slice")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(original_slice, cmap='gray')
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title("Grad-CAM Overlay")
plt.axis('off')

plt.tight_layout()
plt.savefig("gradcam_output.pdf")
print("✅ Grad-CAM visualization saved to gradcam_output.pdf")
