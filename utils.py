import torch
import os

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


def save(model, trigger, args):
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)
    file_name = f'{args.model}_{args.attack_type}_{args.trigger_size}_{args.poisoning_rate}_{args.manual_seed}.pth'
    path = os.path.join(args.checkpoint, file_name)
    torch.save({'state_dict': model.state_dict(),
                'trigger': trigger}, path)
    print(f'Checkpoint saved at {path}')


def load_checkpoint(args):
    file_name = f'{args.model}_{args.attack_type}_{args.trigger_size}_{args.poisoning_rate}_{args.manual_seed}.pth'
    path = os.path.join(args.checkpoint, file_name)
    ckpt = torch.load(path)
    print(f'Checkpoint loaded from {path}')
    return ckpt