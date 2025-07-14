# Register models here by importing class and adding it to registry
from .model_simple_3dcnn import Simple3DCNN
from .model_custom_3dcnn import Custom3DCNN
from .model_simple_3dcnn_gradcam import GradCAM3DCNN 
from .model_resnet18 import ResNet18

_MODEL_REGISTRY = {
                    'simple_3dcnn':Simple3DCNN,
                    'custom_3dcnn':Custom3DCNN,
                    'custom_3dcnn_gradcam':GradCAM3DCNN,
                    'resnet18':ResNet18
                  }

# Returns model's class given the model's name
def get_model(name):
    if name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name]
    else:
        return None