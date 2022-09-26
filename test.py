import torch
import torch.nn.functional as F

import os
import argparse
from tqdm import tqdm

from data import get_dataloader
from models import get_model
from defense import CLP
from utils import save, load_checkpoint

def val(net, data_loader):
    with torch.no_grad():
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


def main(args):
    print(args)

    ckpt = load_checkpoint(args)
    state_dict, trigger = ckpt['state_dict'], ckpt['trigger']
    num_classes, train_loader, val_loader, holdout_loader, test_clean_loader, test_poisoned_loader, _ = get_dataloader(args, trigger)
    net = get_model(args.model, num_classes).to(args.device)
    net.load_state_dict(state_dict)

    print('Before prunning')
    acc = val(net, train_loader)
    print('Training accuracy: %.2f' % acc)
    acc = val(net, val_loader)
    print('Validation accuracy: %.2f' % acc)
    acc, asr = val(net, test_clean_loader), val(net, test_poisoned_loader)
    print('Test clean accuracy: %.2f' % acc)
    print('Test attack success rate: %.2f' % asr)
    
    CLP(net, args.u)
    print('After CLP prunning')
    acc = val(net, train_loader)
    print('Training accuracy: %.2f' % acc)
    acc = val(net, val_loader)
    print('Validation accuracy: %.2f' % acc)
    acc, asr = val(net, test_clean_loader), val(net, test_poisoned_loader)
    print('Test clean accuracy: %.2f' % acc)
    print('Test attack success rate: %.2f' % asr)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Backdoor Training') 

    parser.add_argument('--model', default='resnet18', type=str,
                        help='network structure choice')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Pruning options
    parser.add_argument('--batch-size', default=500, type=int, metavar='N',
                        help='batch size.')
    parser.add_argument('-u', default=3., type=float,
                        help='threshold hyperparameter')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='./ckpt', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    # Miscs
    parser.add_argument('--manual-seed', default=0, type=int, help='manual seed')

    # Device options
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device used for training')

    # data path
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dataset-dir', type=str, default='./dataset')

    # backdoor setting
    parser.add_argument('--attack-type', type=str, default='badnets')
    parser.add_argument('--target_label', type=int, default=0, help='backdoor target label.')
    parser.add_argument('--poisoning-rate', type=float, default=0.1, help='backdoor training sample ratio.')
    parser.add_argument('--trigger-size', type=int, default=3, help='size of square backdoor trigger.')
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True

    main(args)

