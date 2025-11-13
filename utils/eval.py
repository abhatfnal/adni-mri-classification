# utils/load_model.py
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
import yaml
import torch
import plotly.graph_objects as go
from IPython.display import HTML
import plotly.io as pio


def _ensure_on_sys_path(path: Path):
    path = str(path.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _import_obj(dotted: str):
    *mod, name = dotted.split(".")
    mod = importlib.import_module(".".join(mod))
    return getattr(mod, name)


def _make_pkg_stub_with_base(base_cls: type, pkg_name: str):
    if pkg_name not in sys.modules:
        pkg = ModuleType(pkg_name)
        pkg.__path__ = []  # mark as namespace package
        sys.modules[pkg_name] = pkg
    base_mod_name = f"{pkg_name}.base"
    base_mod = ModuleType(base_mod_name)
    base_mod.BaseModel = base_cls
    sys.modules[base_mod_name] = base_mod


def _load_module_from_path_as(model_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(model_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_subclass_in_module(module: ModuleType, base_cls: type) -> type:
    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is base_cls:
            continue
        if issubclass(obj, base_cls) and obj.__module__ == module.__name__:
            candidates.append(obj)
    if not candidates:
        raise LookupError(f"No subclass of {base_cls.__name__} found in {module.__file__}")
    return candidates[0] if len(candidates) == 1 else candidates[-1]


def load_model(
    experiment_dir: Path,
    fold: int = 1,
    project_root: Path = Path("."),
    base_model_import: str = "models.base.BaseModel",
    map_location: str = "cpu",
    strict: bool = True,
):
    """
    Load a model from an experiment directory.

    - Uses config.yaml to get hparams (config["model"]).
    - Expects model.py in the experiment dir.
    - Expects weights in fold_{i}/best_model.pth.

    Returns: (model, weights_path, ModelCls, config_dict)
    """
    project_root = Path(project_root).resolve()
    experiment_dir = Path(experiment_dir).resolve()

    model_py_path = experiment_dir / "model.py"
    config_path = experiment_dir / "config.yaml"
    weights_path = experiment_dir / f"fold_{fold}" / "best_model.pth"

    if not model_py_path.exists():
        raise FileNotFoundError(model_py_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    # load config
    config = yaml.safe_load(config_path.read_text())
    hparams = config.get("model", {})

    # make repo importable and load BaseModel
    _ensure_on_sys_path(project_root)
    BaseModel = _import_obj(base_model_import)

    # fake package so "from .base import BaseModel" resolves
    stub_pkg = f"_exp_pkg_{abs(hash(str(model_py_path)))}"
    _make_pkg_stub_with_base(BaseModel, stub_pkg)

    # import the copied model.py
    mod = _load_module_from_path_as(model_py_path, f"{stub_pkg}.model")
    ModelCls = _find_subclass_in_module(mod, BaseModel)

    # instantiate + load weights
    model = ModelCls(hparams)
    state_dict = torch.load(weights_path, map_location=map_location, weights_only=False).state_dict()
    #state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    return model, config


def create_slice_plot(volume, axis, gradcam_volume=None):
    
    # Convert to numpy and remove batch dimension if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.squeeze().numpy()
        
    if gradcam_volume is not None and isinstance(gradcam_volume, torch.Tensor):
        gradcam_volume = gradcam_volume.squeeze().numpy()

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
        width=400,
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
    
    return fig

def display_slice_plots(vol, overlay):
    figs = [
        _create_slice_plot(vol, axis=0, gradcam_volume=overlay),
        _create_slice_plot(vol, axis=1, gradcam_volume=overlay),
        _create_slice_plot(vol, axis=2, gradcam_volume=overlay),
    ]

    html_figs = [pio.to_html(fig, full_html=False, include_plotlyjs='cdn') for fig in figs]
    HTML(f"<div style='display:flex;gap:0px;'>{''.join(html_figs)}</div>")
