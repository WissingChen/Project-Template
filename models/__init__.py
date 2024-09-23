# build a model list
from .resnet import ResNet101
from .mlp import MLP
model_fns = {"resnet101": ResNet101, "mlp": MLP}
