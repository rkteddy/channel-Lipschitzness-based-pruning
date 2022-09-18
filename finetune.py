import torch
import torch.nn.functional as F

import os
import argparse
from tqdm import tqdm

from data import get_dataloader
from models import get_model

import torch.nn as nn
class CUFLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.3):
        super(CUFLoss, self).__init__()

    def forward(self, pred, target):
        index = torch.argmin(pred, 1)
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, index.reshape(-1, 1), 0.1)
            true_dist.scatter_(1, target.data.unsqueeze(1), 0.9)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
cuf = CUFLoss()

def train(net, data_loader, optimizer, scheduler):
    net.train()
    sum_loss = 0
    num_samples = 0

    pbar = tqdm(data_loader)
    for images, targets in pbar:
        images, targets = images.to(args.device), targets.to(args.device)
        optimizer.zero_grad()

        logits = net(images)
        # loss = F.cross_entropy(logits, targets)
        loss = cuf(logits, targets)
        # w = net.bottleneck_weight
        # loss += (torch.mm(w, w.T) - torch.eye(w.shape[0]).to(w.device)).norm(2)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item() * images.shape[0]
        num_samples += images.shape[0]
        pbar.set_description('Average Loss: %.2f' % (sum_loss/num_samples))
    pbar.close()
    scheduler.step()

    return sum_loss


def val(net, data_loader):
    net.eval()
    n_correct = 0
    n_total = 0

    for images, targets in data_loader:
        images, targets = images.to(args.device), targets.to(args.device)

        logits = net(images)
        prediction = logits.argmax(-1)

        n_correct += (prediction==targets).sum()
        n_total += targets.shape[0]
        
    acc = n_correct / n_total * 100
    return acc


def save(model, args):
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)
    file_name = f'finetune_{args.lr}_{args.model}_{args.attack_type}_{args.trigger_size}_{args.poisoning_rate}_{args.manual_seed}.pth'
    path = os.path.join(args.checkpoint, file_name)
    torch.save(model.state_dict(), path)
    print(f'Checkpoint saved at {path}')


def load_checkpoint(args):
    file_name = f'{args.model}_{args.attack_type}_{args.trigger_size}_{args.poisoning_rate}_{args.manual_seed}.pth'
    path = os.path.join(args.checkpoint, file_name)
    ckpt = torch.load(path)
    print(f'Checkpoint loaded from {path}')
    return ckpt


def main(args):
    print(args)
    num_classes, train_loader, val_loader, holdout_loader, test_clean_loader, test_poisoned_loader = get_dataloader(args)

    net = get_model(args.model, num_classes).to(args.device)
    net.load_state_dict(load_checkpoint(args))
 
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD([net.bottleneck_weight, net.fc.weight, net.fc.bias], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print('Start Trainig')
    for epoch in range(args.epochs):
        print('Epoch %d:' % epoch)
        loss = train(net, train_loader, optimizer, scheduler)
        acc = val(net, val_loader)
        print('Validation accuracy: %.2f' % acc)
        acc, asr = val(net, test_clean_loader), val(net, test_poisoned_loader)
        print('Test clean accuracy: %.2f' % acc)
        print('Test attack success rate: %.2f' % asr)
        save(net, args)

    print('Training finished')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Backdoor Training') 

    parser.add_argument('--model', default='resnet18', type=str,
                        help='network structure choice')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Optimization options
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')

    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='./ckpt', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    # Miscs
    parser.add_argument('--manual-seed', default=0, type=int, help='manual seed')

    # Device options
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device used for training')

    # data path
    parser.add_argument('--dataset', type=str, default='stl10')
    parser.add_argument('--dataset-dir', type=str, default='./dataset')

    # backdoor setting
    parser.add_argument('--attack-type', type=str, default='badnets')
    parser.add_argument('--target_label', type=int, default=0, help='backdoor target label.')
    parser.add_argument('--poisoning-rate', type=float, default=0.1, help='backdoor training sample ratio.')
    parser.add_argument('--trigger-size', type=int, default=3, help='size of square backdoor trigger.')
    args = parser.parse_args()
    main(args)

