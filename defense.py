import torch
import torch.nn as nn

def CLP(net, u):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            print(index)
        
    #     # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)

# def CLP_for_MLP(net, u):
#     for m in net.modules():
#         if isinstance(m, nn.BatchNorm1d):
#             std = m.running_var.sqrt()
#             weight = m.weight

#             W = linear.weight * (weight / std).unsqueeze(-1)
#             channel_lips = W.norm(2, -1)

#             M = nn.Parameter(torch.eye(weight.shape[0]).to(weight.device))
#             M.requries_grad = True
#             eye = torch.eye(weight.shape[0]).to(weight.device)

#             for i in range(100):
#                 loss = 10*(M@M.T - eye).pow(2).sum() + (M@W).norm(2, -1).max()
#                 grad = torch.autograd.grad(loss, M)[0]
#                 M = M - grad * 1e-2
#                 print((M@M.T - eye).pow(2).sum(), (M@W).norm(2, -1).max())
            
#             T = (M@W)
#             idx = T.norm(2, -1).argmax()
#             print(T.norm(2, -1).max())
#             # T[idx] = 0
#             W = M.T@T

#             params = linear.state_dict()
#             params['weight'] = W / (weight / std).unsqueeze(-1)
#             linear.load_state_dict(params)



#             # channel_lips = []
#             # for idx in range(weight.shape[0]):
#             #     w = linear.weight[idx] * (weight[idx]/std[idx]).abs()
#             #     channel_lips.append(w.norm(2))
#             # channel_lips = torch.Tensor(channel_lips)

#             # index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

#             # params = m.state_dict()
#             # for idx in index:
#             #     params['weight'][idx] = 0
#             # m.load_state_dict(params)
        
#         # Convolutional layer should be followed by a BN layer by default
#         elif isinstance(m, nn.Linear):
#             linear = m
#     return net


def batch_entropy(x, step_size=0.1):
    n_bars = int((x.max()-x.min())/step_size)
    entropy = 0
    for n in range(n_bars):
        num = ((x > x.min() + n*step_size) * (x < x.min() + (n+1)*step_size)).sum(0)
        p = num / x.shape[-1]
        entropy += - p * p.log().nan_to_num(0)
    return entropy

import math
import numpy as np

def gaussian(x, std=1):
    return 1/np.sqrt(2*np.pi*std**2) * torch.exp(-x**2/(2*std**2))

def kernel_entropy(x, n_bars=100, bw=1, eps=1e-6):
    min_ = x.min().floor().item()
    max_ = x.max().ceil().item()
    step_size = (max_ - min_) / n_bars

    n, c = x.shape
    Z = torch.zeros(c).to(x.device)
    ps = torch.zeros(c, math.ceil((max_-min_)/step_size)).to(x.device)
    for i, k in enumerate(np.arange(min_, max_, step_size)):
        z = gaussian(x - k, bw).sum(0)
        Z += z
        ps[:, i] = z
    ps = ps / Z.unsqueeze(-1)
    entropy = - (ps * (ps+eps).log()).sum(-1)
    return entropy

def skewness(x):
    return (x - x.mean(0)).pow(3).mean(0) / (x - x.mean(0)).pow(2).mean(0).pow(3).sqrt()

def kurtosis(x):
    return (x - x.mean(0)).pow(4).mean(0) / (x - x.mean(0)).pow(2).mean(0).pow(4).sqrt()

def multi_skewness(x):
    return (x - x.mean()).pow(3).mean() / (x - x.mean()).pow(2).mean().pow(3).sqrt()

def multi_kurtosis(x):
    return (x - x.mean()).pow(4).mean() / (x - x.mean()).pow(2).mean().pow(4).sqrt()


def DDE(net, u, mixture_data_loader, args):
    mixture_data = iter(mixture_data_loader).next()[0].to(args.device)
    params = net.state_dict()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.collect_feats = True

    with torch.no_grad():
        net(mixture_data)
        for name, m in net.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                feats = m.batch_feats
                feats = (feats - feats.mean(0).reshape(1, -1)) / feats.std(0).reshape(1, -1)
                
                entropy = batch_entropy(feats)
                index = (entropy<(entropy.mean() - 2.*entropy.std()))
                
                # skws = skewness(feats)
                # kurt = kurtosis(feats)
                # sbc = (skws**2+1)/kurt
                # index = sbc > (sbc.mean()+2.3*sbc.std())
                # print(index)

                # import diptest
                # import numpy as np

                # dips = []
                # for feat in feats.permute(1, 0):
                #     dip, pval = diptest.diptest(np.array(feat))
                #     dips.append(pval)
                # dips = torch.Tensor(dips)
                # print(dips)
                # # index = dips < (dips.mean()-3*dips.std())
                # index = dips < 0.3

                params[name+'.weight'][index] = 0
                params[name+'.bias'][index] = 0
    net.load_state_dict(params)
