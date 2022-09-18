import torch

class TensorsDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):

        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)

        if self.target_tensor is None:
            return data_tensor

        target_tensor = self.target_tensor[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)

        return data_tensor, target_tensor

    def __len__(self):
        return self.data_tensor.size(0)


def add_trigger(image, trigger, attack_type):
    if attack_type == 'badnets':
        trigger_size = trigger.shape[-1]
        trigger = trigger.reshape(1, 3, trigger_size, trigger_size)
        trigger_image = image.clone()
        trigger_image[:, :, -trigger_size:, -trigger_size:] = trigger
        return trigger_image
    elif attack_type == 'blended':
        trigger_size = trigger.shape[-1]
        trigger = trigger.reshape(1, 3, trigger_size, trigger_size)
        trigger_image = image.clone()
        trigger_image[:, :, -trigger_size:, -trigger_size:] = \
            trigger_image[:, :, -trigger_size:, -trigger_size:] * 0.8 + trigger * 0.2
        return trigger_image
    else:
        raise

