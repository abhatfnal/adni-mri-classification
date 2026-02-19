import torch
import argparse
import sys
import numpy as np
import os

from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from data.datasets import ADNIDataset

from pytorch_grad_cam import GradCAM, ShapleyCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def compute_gradcam(model, sample, target_layer_name, target_class=1):
    
    model.eval()
    out = model(sample)
    
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    
    with ShapleyCAM(model=model, target_layers=[ target_layer ]) as cam:
       gcam_overlay = cam(sample, targets=[ ClassifierOutputTarget(target_class)])
       
    return gcam_overlay.squeeze()

if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Average Grad-CAM")
    
    parser.add_argument("--exp-dir", type=str, required=True, help="Path to the experiment directory")
    parser.add_argument("--fold", type=int, required=True, help="Fold number")
    parser.add_argument("--output-dir", type=str, required=False, default=".")
    
    args = parser.parse_args(sys.argv[1:])
    
    # Load config
    cfg = OmegaConf.load(args.exp_dir + '/config.yaml')
    
    # Load test dataset
    test_dataset = ADNIDataset(cfg.data.test_csv)
    loader = DataLoader(test_dataset,1,shuffle=False)
    
    # Load model
    model = torch.load(os.path.join(args.exp_dir, f'fold_{args.fold}', 'best_model.pth'), map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    
    samples = []
    labels = []
    for imgs,lbls in loader:
        samples.append(imgs)
        labels.append(lbls)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Save labels to file
    np.save(os.path.join(args.output_dir, 'labels.npy'), np.array(labels))
    
    # Initialize average Grad-CAM
    avg_cn_cam = 0
    avg_mci_cam = 0
    avg_ad_cam = 0
    
    # Initialize average volumes
    avg_cn_vol = 0
    avg_mci_vol = 0
    avg_ad_vol = 0
    
    # Predictions
    pred = []
    
    # Compute Grad-CAM for each sample and save to file
    for i, sample in enumerate(samples):
        
        print("Processing sample", i+1, "/", len(samples), end='\r')
        
        cam_overlay = compute_gradcam(model, sample, 'cbam_final', labels[i])
        
        # compile prediction
        pred.append(model(sample).max(1, keepdim=True)[1].detach().numpy())
        
        # Save sample
        np.save(os.path.join(args.output_dir, f'sample_{i}.npy'), sample)

        # Save Grad-CAM overlay
        np.save(os.path.join(args.output_dir, f'gradcam_{i}.npy'), cam_overlay)
        
        if labels[i] == 0:
            avg_cn_cam += cam_overlay
            avg_cn_vol += samples[i]
        elif labels[i] == 1:
            avg_mci_cam += cam_overlay
            avg_mci_vol += samples[i]
        elif labels[i] == 2:
            avg_ad_cam += cam_overlay
            avg_ad_vol += samples[i]
    
    # Save predictions to file
    np.save(os.path.join(args.output_dir, 'predictions.npy'), np.array(pred))
     
    # Labels counts
    num_cn = sum([1 for lbl in labels if lbl == 0])
    num_mci = sum([1 for lbl in labels if lbl == 1])
    num_ad = sum([1 for lbl in labels if lbl == 2])

    # Average Grad-CAMs
    if num_cn > 0:
        avg_cn_cam /= num_cn
        avg_cn_vol /= num_cn
        
    if num_mci > 0:
        avg_mci_cam /= num_mci
        avg_mci_vol /= num_mci
        
    if num_ad > 0:
        avg_ad_cam /= num_ad
        avg_ad_vol /= num_ad
        
    # Save average Grad-CAMs
    np.save(os.path.join(args.output_dir, 'avg_cn_gradcam.npy'), avg_cn_cam)
    np.save(os.path.join(args.output_dir, 'avg_mci_gradcam.npy'), avg_mci_cam)
    np.save(os.path.join(args.output_dir, 'avg_ad_gradcam.npy'), avg_ad_cam)

    # Save average volumes
    np.save(os.path.join(args.output_dir, 'avg_cn_volume.npy'), avg_cn_vol)
    np.save(os.path.join(args.output_dir, 'avg_mci_volume.npy'), avg_mci_vol)
    np.save(os.path.join(args.output_dir, 'avg_ad_volume.npy'), avg_ad_vol)