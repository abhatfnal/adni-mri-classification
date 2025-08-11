import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch

from skimage import measure
from data.datasets import ADNIDataset
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from pytorch_grad_cam import GradCAM, EigenCAM, GradCAMPlusPlus, LayerCAM,ScoreCAM, ShapleyCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import os

# Load experiment from folder and sets session state variables
def load_experiment(exp_dir):
    
    # Load config.yaml from the experiment directory
    config_path = os.path.join(exp_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    
    # Set experiment dir in session
    st.session_state["exp_dir"] = exp_dir
    
     # Load config
    config = OmegaConf.load(config_path)
    
    # Set session state variables based on the configuration
    st.session_state['exp_dataset_csv'] = config.data.test_csv
    st.session_state['exp_name'] = os.path.basename(exp_dir)
    
    if config.cross_validation.method == 'kfold':
        st.session_state['exp_method'] = 'kfold'
        st.session_state['exp_nfolds'] = config.cross_validation.folds
        
    else:
        st.session_state['exp_method'] = 'holdout'
        
# Load dataset from CSV file path
@st.cache_data
def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} does not exist.")

    dataset = ADNIDataset(csv_path)
    return dataset

# Load model from the specified path
# @st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    return model

def compute_gradcam(model_path, volume_index, target_class, target_layer_name):
    
    # Load model and dataset
    model = load_model(model_path)
    dataset = load_dataset(st.session_state["exp_dataset_csv"])
    
    # Get volume and prepare input
    input_tensor, _ = dataset[volume_index]
    
    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    
    # Target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    
    with EigenCAM(model=model, target_layers=[ target_layer ]) as cam:
       gcam_overlay = cam(input_tensor, targets=[ ClassifierOutputTarget(target_class)])
    # Compute gradcam
    # with EigenCAM(model=model, target_layers=[ target_layer]) as cam:   
    #     gcam_overlay = cam(input_tensor, targets=[ ClassifierOutputTarget(target_class)])
    
    print(f"Computing EigenCam: Class: {target_class}")
    
    return gcam_overlay.squeeze(0)

# Computes predictions of selected model on dataset
@st.cache_data
def compute_predictions(model_path, dataset_csv_path):

    # Load model
    model = load_model(model_path)
    
    # Load dataset
    dataset = load_dataset(dataset_csv_path)
    
    # Compute predictions, confidence, actual label and save to records
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
            'pred': pred, 
            'truth': int(lbl.item()), 
            'confidence': conf
        })
    
    # Create and return dataframe
    df = pd.DataFrame(records)
    df['correct'] = df['pred'] == df['truth']
    
    return df

def create_sidebar():

    #===|| Directory navigation ||===

    st.sidebar.subheader("Directory Navigation")
    
    # Initialize previous, current and next dir
    if 'prev_dir' not in st.session_state:
        st.session_state.prev_dir = os.getcwd()

    if 'current_dir' not in st.session_state:
        st.session_state.current_dir = os.getcwd()  
        
    if 'next_dir' not in st.session_state:
        st.session_state.next_dir = os.getcwd()

    # Handles manual path change 
    if st.session_state.current_dir != st.session_state.prev_dir:   # Path has been manually changed

        # If current directory does not exist or is not a directory, reset it to previous directory
        if not os.path.exists(st.session_state.current_dir) or not os.path.isdir(st.session_state.current_dir):
            st.session_state.current_dir = st.session_state.prev_dir
        else:
            st.session_state.prev_dir = st.session_state.current_dir
            
        # the current path has been changed manually: set the next directory to the current one
        st.session_state.next_dir = st.session_state.current_dir


    # Handles open directory button
    if st.session_state.current_dir != st.session_state.next_dir:   #Open button has been clicked
        
        if not os.path.exists(st.session_state.next_dir) or not os.path.isdir(st.session_state.next_dir):
            st.session_state.next_dir = st.session_state.current_dir
        else:
            st.session_state.current_dir = st.session_state.next_dir
            
        # Update the previous directory to the current one
        st.session_state.prev_dir = st.session_state.current_dir
        
        
    # Current path text input
    st.sidebar.text_input(
        "Current directory", 
        key='current_dir',
        placeholder="Enter directory path"
    )
    
    # Files and subdirectories in the current directory
    subdir = st.sidebar.selectbox("Files",
        options=sorted(os.listdir(st.session_state.current_dir)),
        key='subdir_select',
    )
    
    col1, col2, col3 = st.sidebar.columns(3)

    # Open button (open subdirectory)
    with col1:
        if st.button("Open"):
            st.session_state.next_dir = os.path.join(st.session_state.current_dir, subdir)
            st.rerun()
    
    # Back button (parent directory)
    with col2:
        if st.button("Back"):
            st.session_state.next_dir = os.path.dirname(st.session_state.current_dir)
            st.rerun()
    
    # Load experiment button 
    with col3:
        if st.button("Load"):
            
            exp_dir = os.path.join(st.session_state.current_dir, subdir)
            
            try:
                # Load experiment configuration
                load_experiment(exp_dir)
                
            except Exception as e:
                st.sidebar.error(f"Error loading configuration: {e}")
                return
            
            
    # Model selection if experiment has multiple folds
    if 'exp_method' in st.session_state:
        if st.session_state['exp_method'] == 'kfold':
            
            st.sidebar.header("Model Selection")
            
            # Set session variable for selected fold
            st.session_state["exp_selected_fold"] = st.sidebar.selectbox("Fold", options=[ i for i in range(1, st.session_state["exp_nfolds"])])
     
def create_gradcam_menu(model_path):
    
    st.text("Gradcam")
    # Enable / disable gradcam overlay
    st.session_state["gradcam_enabled"] = st.checkbox("GradCAM Overlay")
    
    # Load model
    model = load_model(model_path)
    
    # Get layers list
    layers = [name for name, module in model.named_modules() ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select layer to apply gradcam
        st.session_state["gradcam_layer"] = st.selectbox("Layer", options = layers, index=(len(layers)-1))
    
    with col2:
        # Select target class
        st.session_state["gradcam_class"] = st.selectbox("Class", options = [0,1,2])
        
@st.cache_data
def create_slice_plot(volume_index, axis, gradcam_volume=None):
    
    dataset = load_dataset(st.session_state["exp_dataset_csv"])
    volume, _ = dataset[volume_index]
    
    # Convert to numpy and remove batch dimension if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.squeeze().numpy()
    
    depth, height, width = volume.shape
    
    # Determine slices and axis labeling
    if axis == 0:  # Axial (XY plane, slices along Z)
        slices = [volume[i, :, :] for i in range(depth)]
        active_idx = depth // 2
        prefix = 'Axial Slice: '
        
        # Prepare Grad-CAM slices if available
        if gradcam_volume is not None:
            gradcam_slices = [gradcam_volume[i, :, :] for i in range(depth)]
            
    elif axis == 1:  # Coronal (XZ plane, slices along Y)
        slices = [volume[:, i, :] for i in range(height)]
        active_idx = height // 2
        prefix = 'Coronal Slice: '
        
        # Prepare Grad-CAM slices if available
        if gradcam_volume is not None:
            gradcam_slices = [gradcam_volume[:, i, :] for i in range(height)]
            
    elif axis == 2:  # Sagittal (YZ plane, slices along X)
        slices = [volume[:, :, i] for i in range(width)]
        active_idx = width // 2
        prefix = 'Sagittal Slice: '
        
        # Prepare Grad-CAM slices if available
        if gradcam_volume is not None:
            gradcam_slices = [gradcam_volume[:, :, i] for i in range(width)]
            
    else:
        raise ValueError("Axis must be 0 (axial), 1 (coronal), or 2 (sagittal)")
    
    fig = go.Figure()
    
    # Volume slice
    fig.add_trace(go.Heatmap(
        z=slices[active_idx],
        colorscale='gray',
        showscale=False
    ))
    
    # Add Grad-CAM overlay if available
    if gradcam_volume is not None:
        
        # Compute global Grad-CAM global max and min
        gmax = gradcam_volume.max()
        gmin = gradcam_volume.min()
    
        fig.add_trace(go.Heatmap(
            z=gradcam_slices[active_idx],
            colorscale='hot',
            zmin=gmin,
            zmax=gmax,
            opacity=0.5,
            showscale=False,
            name='Grad-CAM'
        ))
    
    # Create slider steps
    slider_steps = []
    for i in range(len(slices)):
        step_args = [{'z': [slices[i]]}]
        
        if gradcam_volume is not None:
            step_args[0]['z'].append(gradcam_slices[i])
        
        slider_steps.append({
            'method': 'restyle',
            'label': str(i),
            'args': step_args
        })
    
    # Create slider
    slider = {
        'active': active_idx,
        'currentvalue': {'prefix': prefix},
        'steps': slider_steps,
        'pad': {'t': 30},
        'len': 1.0,
        'x': 0.0
    }
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=80),
        sliders=[slider],
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_experiment_page():
    
    st.set_page_config(layout="wide")
    
    # Set title if experiment is loaded, otherwise show message and return
    if not 'exp_name' in st.session_state:
        st.title("Load experiment using sidebar")
        return 
    
    st.title(st.session_state.exp_name)
    
    # Load model (or get it from cache)
    if st.session_state["exp_method"] == "kfold":
        st.session_state["model_path"] = os.path.join(st.session_state["exp_dir"], f"fold_{ st.session_state['exp_selected_fold']}/best_model.pth")
    
    else:
        st.session_state["model_path"] = os.path.join(st.session_state["exp_dir"], "fold_1/best_model.pth")
        
    # Compute predictions
    df = compute_predictions(st.session_state["model_path"], st.session_state["exp_dataset_csv"])
    
    # Prediction analysis section
    st.header("📊 Prediction Analysis")
    
    col1, col2 =  st.columns([1,1])
    with col1:
        st.metric("Total Samples", len(df))
        st.metric("Accuracy", f"{df['correct'].mean():.1%}")
        st.metric(
            "Avg Confidence (Correct)", 
            f"{df[df['correct']]['confidence'].mean():.1%}"
        )
    
    with col2:
        # Confusion Matrix
        st.write("### Confusion Matrix")
        
        # Compute confusion matrix
        cm = confusion_matrix(df['truth'], df['pred'])
        classes = sorted(df['truth'].unique())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, 
                    yticklabels=classes,
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        # Display in Streamlit
        st.pyplot(fig)
    
    # Filter controls
    with st.expander("🔍 Filter Predictions", expanded=True):
        show_correct = st.checkbox("Show correct predictions", value=True)
        
        true_filter = st.multiselect(
            "Filter by True Label", options=[0,1,2], default=[]
        )
        pred_filter = st.multiselect(
            "Filter by Predicted Label", options=[0,1,2], default=[]
        )
    
    # Apply filters
    filtered_df = df.copy()
    if not show_correct:
        filtered_df = filtered_df[~filtered_df['correct']]
        
    if len(true_filter) > 0:
        filtered_df = filtered_df[filtered_df["truth"].isin(true_filter)]
        
    if len(pred_filter) > 0:
        filtered_df = filtered_df[filtered_df["pred"].isin(pred_filter)]
    
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
        options=list(range(len(df))),
        format_func=lambda i: f"Sample {i} - True: {df['truth'][i]} Pred: {df['pred'][i]}"
    )
    
    # Gradcam menu
    create_gradcam_menu(st.session_state["model_path"])
    
    # Compute Grad-CAM if enabled
    gradcam_volume = None
    if st.session_state.get("gradcam_enabled", False):
        try:
            gradcam_volume = compute_gradcam(
                st.session_state["model_path"],
                sample_idx,
                st.session_state["gradcam_class"],
                st.session_state["gradcam_layer"]
            )
        except Exception as e:
            st.error(f"Error computing Grad-CAM: {e}")
            gradcam_volume = None
            
    
    col1, col2, col3 = st.columns(3)
    
    # Slices plots
    with col1:
        create_slice_plot(sample_idx, 0, gradcam_volume)
        
    with col2:
        create_slice_plot(sample_idx, 1, gradcam_volume)
        
    with col3:
        create_slice_plot(sample_idx, 2, gradcam_volume)

def main():
    
    # Compose sidebar
    create_sidebar()
    
    # Compose main content page
    create_experiment_page()

if __name__ == "__main__":
    main()