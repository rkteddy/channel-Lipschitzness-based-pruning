from .resnet import resnet18

def get_model(model, num_classes):
    if model == 'resnet18':
        return resnet18(num_classes)
    else:
        raise