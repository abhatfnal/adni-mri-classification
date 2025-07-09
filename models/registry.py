# Register models here by importing class and adding it to registry
from .model_3dcnn import Simple3DCNN
from .model_3dcnn_gradcam import GradCAM3DCNN 

_MODEL_REGISTRY = {
                    '3dcnn':Simple3DCNN,
                    '3dcnn_gradcam':GradCAM3DCNN,
                  }

# Returns model's class given the model's name
def get_model(name):
    if name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name]
    else:
        return None