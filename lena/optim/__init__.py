import torch.optim as optim


def build_optimizer(name: str, *args, **kargs):
    if name == 'Adam':
        return optim.Adam(*args, **kargs)
    return None
