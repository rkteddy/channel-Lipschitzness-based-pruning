from .resnet import resnet18
from .mlp import mlp4

def get_model(model, num_classes):
    if model == 'resnet18':
        return resnet18(num_classes)
    elif model == 'mlp4':
        return mlp4(num_classes, True)
    else:
        raise