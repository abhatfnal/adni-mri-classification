# # analysis.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from omegaconf import OmegaConf
from torchcam.methods import GradCAM
import os

from skimage import measure

from data.datasets import ADNIDataset
# ----------------------
# Helpers
# ----------------------

def load_config_dataset(exp_dir):
    """Load configuration and dataset from experiment directory."""
    cfg_path = os.path.join(exp_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        st.error(f"config.yaml not found in {exp_dir}")
        return None, None
    
    config = OmegaConf.load(cfg_path)
    
    try:
        dataset = ADNIDataset(config.data.test_csv, transform=None)
        return config, dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None

def load_model(exp_dir, config, fold=None):
    """Load trained model from experiment directory."""
    if config.get('cross_validation', {}).get('method') == 'kfold' and fold is not None:
        model_file = os.path.join(exp_dir, f'fold_{fold}', 'best_model.pth')
    else:
        model_file = os.path.join(exp_dir, 'best_model.pth')
    
    if not os.path.exists(model_file):
        st.error(f"Model file not found: {model_file}")
        return None
    
    try:
        # Load with weights_only=False for security
        model = torch.load(model_file, map_location='cpu', weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def compute_predictions(model, dataset):
    """Compute predictions for entire dataset."""
    records = []
    
    for i, (img, lbl) in enumerate(dataset):
        # Add batch dimensions
        img = img.unsqueeze(0)  # [1, 1, D, H, W]
        
        with torch.no_grad():
            out = model(img)
            probs = torch.softmax(out, dim=1).cpu().numpy().ravel()
        
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        records.append({
            'idx': i, 
            'pred': pred, 
            'truth': int(lbl.item()), 
            'confidence': conf
        })
    
    df = pd.DataFrame(records)
    df['correct'] = df['pred'] == df['truth']
    return df

def display_slices(volume, cor, sag, ax):
    """Display coronal, sagittal, and axial slices."""
    fig = make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=['Coronal', 'Sagittal', 'Axial']
    )
    
    # Add slices to plot
    fig.add_trace(
        go.Heatmap(z=volume[cor, :, :].T, showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=volume[:, sag, :].T, showscale=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=volume[:, :, ax].T, showscale=False),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(height=300, margin=dict(l=0, r=0, b=0, t=30))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    st.plotly_chart(fig, use_container_width=True)

def display_3d(volume, downscale_factor=2):
    """Render 3D volume using Plotly Volume with downsampling."""
    # Downsample volume for performance
    ds_vol = volume[::downscale_factor, ::downscale_factor, ::downscale_factor]
    
        # Extract surface at a certain intensity threshold
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)

    # Create Mesh3D plot
    mesh = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightgray',
        opacity=1.0
    )

    fig = go.Figure(data=[mesh])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Directory Navigation
# ----------------------

def update_cwd(new_path):
    """Update current working directory in session state"""
    st.session_state.cwd = new_path

def create_dir_buttons():
    """Create buttons for directory navigation"""
    st.sidebar.write(f"**Current folder:** `{st.session_state.cwd}`")
    
    # Parent directory button
    parent_dir = os.path.dirname(st.session_state.cwd)
    if st.sidebar.button("⬆️ Parent Directory", key="parent_dir"):
        if os.path.exists(parent_dir):
            update_cwd(parent_dir)
    
    # Get subdirectories
    try:
        subdirs = [d for d in os.listdir(st.session_state.cwd) 
                  if os.path.isdir(os.path.join(st.session_state.cwd, d))]
        subdirs.sort()
    except Exception as e:
        st.sidebar.error(f"Error listing directory: {e}")
        subdirs = []
    
    # Create buttons for each subdirectory
    for d in subdirs:
        if st.sidebar.button(f"📁 {d}", key=f"dir_{d}"):
            new_path = os.path.join(st.session_state.cwd, d)
            update_cwd(new_path)

# ----------------------
# Main App
# ----------------------

def main():
    st.set_page_config(layout="wide", page_title="MRI Classification Explorer")
    st.title('3D MRI Classification Explorer')
    
    # Initialize session state
    if 'cwd' not in st.session_state:
        st.session_state.cwd = os.getcwd()
    if 'exp_dir' not in st.session_state:
        st.session_state.exp_dir = ""
    
    # Sidebar setup
    st.sidebar.title("Navigation & Controls")
    
    # Directory navigation
    with st.sidebar.expander("📁 Folder Navigation", expanded=True):
        st.write(f"**Current folder:** `{st.session_state.cwd}`")
        
        col1, col2 = st.columns(2)
        if col1.button('⬆️ Parent Directory'):
            parent = os.path.dirname(st.session_state.cwd)
            if os.path.exists(parent):
                st.session_state.cwd = parent
        
        # Get subdirectories
        try:
            subdirs = [d for d in os.listdir(st.session_state.cwd) 
                      if os.path.isdir(os.path.join(st.session_state.cwd, d))]
        except Exception:
            subdirs = []
        
        # Directory selection
        selected_dir = st.selectbox(
            "Select subdirectory", 
            [""] + subdirs,
            index=0
        )
        
        if selected_dir:
            new_path = os.path.join(st.session_state.cwd, selected_dir)
            if st.button(f"📂 Open {selected_dir}"):
                st.session_state.cwd = new_path
    
    
    # Experiment loader
    with st.sidebar.expander("🔬 Experiment Setup", expanded=True):
        exp_dir = st.text_input(
            "Experiment directory",
            value=st.session_state.cwd,
            key="exp_input"
        )
        
        if st.button("🚀 Load Experiment"):
            if os.path.isdir(exp_dir):
                st.session_state.exp_dir = exp_dir
                st.success("Experiment loaded!")
            else:
                st.error("Invalid directory")
    
    # Early return if no experiment loaded
    if not st.session_state.exp_dir:
        st.info("👉 Please select an experiment directory in the sidebar")
        st.image("https://placeholder.pics/svg/600x400/DEDEDE/555555/3D%20MRI%20Analyzer")
        return
    
    # Load config and dataset
    config, dataset = load_config_dataset(st.session_state.exp_dir)
    if config is None or dataset is None:
        return
    
    # Fold selection
    fold = None
    if config.get('cross_validation', {}).get('method') == 'kfold':
        n_folds = int(config.cross_validation.folds)
        fold = st.sidebar.selectbox("Select Fold", list(range(1, n_folds+1)), index=0)
    
    # Load model with weights_only=False
    model = load_model(st.session_state.exp_dir, config, fold)
    if model is None:
        return
    
    # Compute predictions
    df = compute_predictions(model, dataset)
    
    # Prediction analysis section
    st.header("📊 Prediction Analysis")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Total Samples", len(df))
        st.metric("Accuracy", f"{df['correct'].mean():.1%}")
        st.metric(
            "Avg Confidence (Correct)", 
            f"{df[df['correct']]['confidence'].mean():.1%}"
        )
    
    with col2:
        # Filter controls
        with st.expander("🔍 Filter Predictions", expanded=True):
            show_correct = st.checkbox("Show correct predictions", value=True)
            min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.01)
        
        # Apply filters
        filtered_df = df.copy()
        if not show_correct:
            filtered_df = filtered_df[~filtered_df['correct']]
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
        
        # Display filtered results
        st.dataframe(
            filtered_df.style.background_gradient(
                subset=['confidence'], 
                cmap='RdYlGn'
            ),
            height=300,
            use_container_width=True
        )
    
    # Sample visualization section
    st.header("🔬 Volume Visualization")
    
    # Sample selection
    sample_idx = st.selectbox(
        "Select sample", 
        options=df['idx'].tolist(),
        format_func=lambda x: f"Sample {x} - True: {df.loc[df['idx']==x, 'truth'].iloc[0]} Pred: {df.loc[df['idx']==x, 'pred'].iloc[0]}"
    )
    
    # Get sample data
    rec = df[df['idx'] == sample_idx].iloc[0]
    img, lbl = dataset[sample_idx]
    volume = img.numpy()
    volume = volume.squeeze() 
    
    # Display prediction info
    status = "✅ Correct" if rec.correct else "❌ Incorrect"
    st.subheader(f"Sample {sample_idx} - {status}")
    st.write(f"**True Label:** {rec.truth} | **Prediction:** {rec.pred}")
    st.write(f"**Confidence:** {rec.confidence:.1%}")
    
    # Slice controls in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        cor = st.slider(
            "Coronal Slice", 
            0, volume.shape[0]-1, volume.shape[0]//2,
            key='cor_slider'
        )
    with col2:
        sag = st.slider(
            "Sagittal Slice", 
            0, volume.shape[1]-1, volume.shape[1]//2,
            key='sag_slider'
        )
    with col3:
        ax = st.slider(
            "Axial Slice", 
            0, volume.shape[2]-1, volume.shape[2]//2,
            key='ax_slider'
        )
    
    # Display slices
    display_slices(volume, cor, sag, ax)
    
    # Advanced visualizations
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Show 3D Volume", key='view3d'):
            with st.spinner("Rendering 3D volume..."):
                display_3d(volume)
    
    with col2:
        cam_enabled = st.checkbox("🧠 Show Grad-CAM Activation", key='cam_chk')
        if cam_enabled:
            try:
                # Initialize Grad-CAM
                cam = GradCAM(model, target_layer='backbone.layer4')
                
                # Prepare input
                inp = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
                
                # Forward pass
                with torch.no_grad():
                    out = model(inp)
                
                # Get activation map
                cls = out.argmax(1).item()
                activation = cam(cls, out)[0].squeeze().cpu().numpy()
                
                # Normalize activation
                norm_act = (activation - activation.min()) / (activation.max() - activation.min())
                
                # Create overlay for coronal slice
                img_slice = volume[cor]
                img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
                
                # Create RGB representation
                overlay = np.zeros((*img_slice.shape, 3))
                overlay[..., 0] = img_slice  # Red channel = original
                overlay[..., 1] = norm_act[cor]  # Green channel = activation
                
                # Display result
                st.image(
                    overlay, 
                    caption="Grad-CAM Activation Overlay (Coronal Slice)", 
                    width=400
                )
            except Exception as e:
                st.error(f"Error generating Grad-CAM: {e}")
    
    # Save snapshot
    with st.sidebar.expander("💾 Save Options"):
        save_path = st.text_input("Save path", "slice_snapshot.png")
        if st.button("💾 Save Current Slice"):
            try:
                plt.imsave(save_path, volume[cor], cmap='gray')
                st.success(f"Slice saved to {save_path}")
            except Exception as e:
                st.error(f"Error saving slice: {e}")

if __name__ == '__main__':
    main()