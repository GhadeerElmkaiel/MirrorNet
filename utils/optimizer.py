import torch
from torch import optim


def get_optim(net, args):
    """
    Function that returns the wanted optimizer using the arguments
    :param net: the network we want to train
    :param args: the arguments of the optimizer
    :return: torch optimizer
    """
    name = args.optim
    optimizer = None
    if name == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=args.betas)
    elif name == 'sgd':
        optimizer = optim.SGD
    elif name == 'adamax':
        optimizer = optim.Adamax
    elif name == 'asgd':
        optimizer = optim.ASGD
    elif name == 'rmsprop':
        optimizer = optim.RMSprop
    elif name == 'rprop':
        optimizer = optim.Rprop
    return optimizer