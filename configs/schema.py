
from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class ModelConfig:
    name:        str = MISSING    # required
    in_channels: int = 1
    num_classes: int = 3          

@dataclass
class OptimizerConfig:
    name:          str = "adam"
    lr:            float = 1e-3
    weight_decay:  float = 0.0

@dataclass
class SchedulerConfig:
    name:   str = "CosineAnnealingLR"
    t_max:  int = 100
    lr_min: float = 1e-6

@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs:     int = 50
    optimizer:  OptimizerConfig = OptimizerConfig()
    scheduler:  SchedulerConfig = SchedulerConfig()

@dataclass
class DataConfig:
    augmentation:        bool = False
    augmentation_params: dict = MISSING    # required subtree if augmentation=True

@dataclass
class FullConfig:
    model:    ModelConfig    = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data:     DataConfig     = DataConfig()
