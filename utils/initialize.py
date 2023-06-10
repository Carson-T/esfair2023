import torch

def xavier(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

def kaiming(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)