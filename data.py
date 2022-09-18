import torch
import torchvision
from torchvision import transforms

from utils import TensorsDataset, add_trigger

def get_data_tensor(dataset):
    # Shuffle the dataset with manual seed
    dataset = torch.utils.data.random_split(dataset, [len(dataset), 0], generator=torch.Generator().manual_seed(0))[0]

    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    img_list = []
    tgt_list = []
    for img, tgt in loader:
        img_list.append(img)
        tgt_list.append(tgt)
    return torch.cat(img_list), torch.cat(tgt_list).long()

def get_dataloader(args, trigger=None):
    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=False, download=True, transform=transforms.ToTensor())
    elif args.dataset == 'stl10':
        train_set = torchvision.datasets.STL10(root=args.dataset_dir, split='train', download=True, transform=transforms.ToTensor())
        test_set = torchvision.datasets.STL10(root=args.dataset_dir, split='test', download=True, transform=transforms.ToTensor())
    else:
        raise

    train_img, train_tgt = get_data_tensor(train_set)
    test_img, test_tgt = get_data_tensor(test_set)

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((train_img.shape[2], train_img.shape[3]), padding=train_img.shape[3]//8),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(train_img.mean([0, 2, 3]), train_img.std([0, 2, 3])),
    ])

    transform_test = transforms.Normalize(train_img.mean([0, 2, 3]), train_img.std([0, 2, 3]))

    if trigger is None:
        trigger = torch.randn(3, args.trigger_size, args.trigger_size)
        
    num_poisoned = int(args.poisoning_rate * 0.95 * len(train_set))
    num_train = int(0.95 * len(train_set))
    num_validation = int(0.04 * len(train_set))
    num_holdout = int(0.01 * len(train_set))

    #                       |----------------------------mixed-----------------------------|
    # Cut training set into |----poisoned----|--------------------clean--------------------|----val----|--holdout--|
    poison_images = add_trigger(train_img[:num_poisoned], trigger, args.attack_type)
    poison_targets = torch.ones_like(train_tgt[:num_poisoned]) * args.target_label
    mixed_images = torch.cat((poison_images, train_img[num_poisoned:num_train]))
    mixed_targets = torch.cat((poison_targets, train_tgt[num_poisoned:num_train]))
    mixed_dataset = TensorsDataset(mixed_images, mixed_targets, transforms=transform_train)
    mixed_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_images = train_img[num_train:len(train_set)-num_holdout]
    val_targets = train_tgt[num_train:len(train_set)-num_holdout]
    val_dataset = TensorsDataset(val_images, val_targets, transforms=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    holdout_images = train_img[len(train_set)-num_holdout:]
    holdout_targets = train_tgt[len(train_set)-num_holdout:]
    holdout_dataset = TensorsDataset(holdout_images, holdout_targets, transforms=transform_train)
    holdout_loader = torch.utils.data.DataLoader(holdout_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    test_poisoned_images = add_trigger(test_img, trigger, args.attack_type)
    test_poisoned_images = test_poisoned_images[test_tgt!=args.target_label]
    test_poisoned_targets = test_tgt[test_tgt!=args.target_label]
    test_poisoned_targets = torch.ones_like(test_poisoned_targets) * args.target_label
    test_poisoned_dataset = TensorsDataset(test_poisoned_images, test_poisoned_targets, transforms=transform_test)
    test_poisoned_loader = torch.utils.data.DataLoader(test_poisoned_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    test_clean_dataset = TensorsDataset(test_img, test_tgt, transforms=transform_test)
    test_clean_loader = torch.utils.data.DataLoader(test_clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_tgt.max()+1, mixed_loader, val_loader, holdout_loader, test_clean_loader, test_poisoned_loader, trigger