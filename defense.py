import torch
import torch.nn as nn

"""
    Input:
        - net: model to be pruned
        - u: coefficient that determines the pruning threshold
    Output:
        None (in-place modification on the model)
"""

def CLP(net, u):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            print(index)
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)