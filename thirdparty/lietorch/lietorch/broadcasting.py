import torch
import numpy as np

def check_broadcastable(x, y):
    assert len(x.shape) == len(y.shape)
    for (n, m) in zip(x.shape[:-1], y.shape[:-1]):
        assert n==m or n==1 or m==1

def broadcast_inputs(x, y):
    """ Automatic broadcasting of missing dimensions """
    if y is None:
        xs, xd = x.shape[:-1], x.shape[-1] 
        return (x.view(-1, xd).contiguous(), ), x.shape[:-1]

    check_broadcastable(x, y)

    xs, xd = x.shape[:-1], x.shape[-1] 
    ys, yd = y.shape[:-1], y.shape[-1]
    out_shape = [max(n,m) for (n,m) in zip(xs,ys)]

    if x.shape[:-1] == y.shape[-1]:
        x1 = x.view(-1, xd)
        y1 = y.view(-1, yd)

    else:
        x_expand = [m if n==1 else 1 for (n,m) in zip(xs, ys)]
        y_expand = [n if m==1 else 1 for (n,m) in zip(xs, ys)]
        x1 = x.repeat(x_expand + [1]).reshape(-1, xd).contiguous()
        y1 = y.repeat(y_expand + [1]).reshape(-1, yd).contiguous()

    return (x1, y1), tuple(out_shape)
