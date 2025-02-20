import torch
import numpy as np

# group operations implemented in cuda
from .group_ops import Exp, Log, Inv, Mul, Adj, AdjT, Jinv, Act3, Act4, ToMatrix, ToVec, FromVec
from .broadcasting import broadcast_inputs


class LieGroupParameter(torch.Tensor):
    """ Wrapper class for LieGroup """

    from torch._C import _disabled_torch_function_impl
    __torch_function__ = _disabled_torch_function_impl

    def __new__(cls, group, requires_grad=True):
        data = torch.zeros(group.tangent_shape, 
                           device=group.data.device, 
                           dtype=group.data.dtype, 
                           requires_grad=True)

        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self, group):
        self.group = group

    def retr(self):
        return self.group.retr(self)

    def log(self):
        return self.retr().log()

    def inv(self):
        return self.retr().inv()

    def adj(self, a):
        return self.retr().adj(a)

    def __mul__(self, other):
        if isinstance(other, LieGroupParameter):
            return self.retr() * other.retr()
        else:
            return self.retr() * other

    def add_(self, update, alpha):
        self.group = self.group.exp(alpha*update) * self.group

    def __getitem__(self, index):
        return self.retr().__getitem__(index)


class LieGroup:
    """ Base class for Lie Group """

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return "{}: size={}, device={}, dtype={}".format(
            self.group_name, self.shape, self.device, self.dtype)

    @property
    def shape(self):
        return self.data.shape[:-1]

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    def vec(self):
        return self.apply_op(ToVec, self.data)

    @property
    def tangent_shape(self):
        return self.data.shape[:-1] + (self.manifold_dim,)

    @classmethod
    def Identity(cls, *batch_shape, **kwargs):
        """ Construct identity element with batch shape """
        
        if isinstance(batch_shape[0], tuple):
            batch_shape = batch_shape[0]
        
        elif isinstance(batch_shape[0], list):
            batch_shape = tuple(batch_shape[0])

        numel = np.prod(batch_shape)
        data = cls.id_elem.reshape(1,-1)

        if 'device' in kwargs:
            data = data.to(kwargs['device'])

        if 'dtype' in kwargs:
            data = data.type(kwargs['dtype'])

        data = data.repeat(numel, 1)
        return cls(data).view(batch_shape)

    @classmethod
    def IdentityLike(cls, G):
        return cls.Identity(G.shape, device=G.data.device, dtype=G.data.dtype)

    @classmethod
    def InitFromVec(cls, data):
        return cls(cls.apply_op(FromVec, data))

    @classmethod
    def Random(cls, *batch_shape, sigma=1.0, **kwargs):
        """ Construct random element with batch_shape by random sampling in tangent space"""

        if isinstance(batch_shape[0], tuple):
            batch_shape = batch_shape[0]
        
        elif isinstance(batch_shape[0], list):
            batch_shape = tuple(batch_shape[0])
        
        tangent_shape = batch_shape + (cls.manifold_dim,)
        xi = torch.randn(tangent_shape, **kwargs)
        return cls.exp(sigma * xi)

    @classmethod
    def apply_op(cls, op, x, y=None):
        """ Apply group operator """
        inputs, out_shape = broadcast_inputs(x, y)

        data = op.apply(cls.group_id, *inputs)
        return data.view(out_shape + (-1,))

    @classmethod
    def exp(cls, x):
        """ exponential map: x -> X """
        return cls(cls.apply_op(Exp, x))

    def quaternion(self):
        """ extract quaternion """
        return self.apply_op(Quat, self.data)

    def log(self):
        """ logarithm map """
        return self.apply_op(Log, self.data)

    def inv(self):
        """ group inverse """
        return self.__class__(self.apply_op(Inv, self.data))

    def mul(self, other):
        """ group multiplication """
        return self.__class__(self.apply_op(Mul, self.data, other.data))

    def retr(self, a):
        """ retraction: Exp(a) * X """
        dX = self.__class__.apply_op(Exp, a)
        return self.__class__(self.apply_op(Mul, dX, self.data))

    def adj(self, a):
        """ adjoint operator: b = A(X) * a """
        return self.apply_op(Adj, self.data, a)

    def adjT(self, a):
        """ transposed adjoint operator: b = a * A(X) """
        return self.apply_op(AdjT, self.data, a)

    def Jinv(self, a):
        return self.apply_op(Jinv, self.data, a)

    def act(self, p):
        """ action on a point cloud """
        
        # action on point
        if p.shape[-1] == 3:
            return self.apply_op(Act3, self.data, p)
        
        # action on homogeneous point
        elif p.shape[-1] == 4:
            return self.apply_op(Act4, self.data, p)

    def matrix(self):
        """ convert element to 4x4 matrix """
        I = torch.eye(4, dtype=self.dtype, device=self.device)
        I = I.view([1] * (len(self.data.shape) - 1) + [4, 4])
        return self.__class__(self.data[...,None,:]).act(I).transpose(-1,-2)

    def translation(self):
        """ extract translation component """
        p = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        p = p.view([1] * (len(self.data.shape) - 1) + [4,])
        return self.apply_op(Act4, self.data, p)

    def detach(self):
        return self.__class__(self.data.detach())

    def view(self, dims):
        data_reshaped = self.data.view(dims + (self.embedded_dim,))
        return self.__class__(data_reshaped)

    def __mul__(self, other):
        # group multiplication
        if isinstance(other, LieGroup):
            return self.mul(other)

        # action on point
        elif isinstance(other, torch.Tensor):
            return self.act(other)

    def __getitem__(self, index):
        return self.__class__(self.data[index])

    def __setitem__(self, index, item):
        self.data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self.data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self.data.cpu())

    def cuda(self):
        return self.__class__(self.data.cuda())

    def float(self, device):
        return self.__class__(self.data.float())

    def double(self, device):
        return self.__class__(self.data.double())

    def unbind(self, dim=0):
        return [self.__class__(x) for x in self.data.unbind(dim=dim)]
        

class SO3(LieGroup):
    group_name = 'SO3'
    group_id = 1
    manifold_dim = 3
    embedded_dim = 4
    
    # unit quaternion
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 1.0])

    def __init__(self, data):
        if isinstance(data, SE3):
            data = data.data[..., 3:7]

        super(SO3, self).__init__(data)


class RxSO3(LieGroup):
    group_name = 'RxSO3'
    group_id = 2
    manifold_dim = 4
    embedded_dim = 5
    
    # unit quaternion
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 1.0, 1.0])

    def __init__(self, data):
        if isinstance(data, Sim3):
            data = data.data[..., 3:8]

        super(RxSO3, self).__init__(data)


class SE3(LieGroup):
    group_name = 'SE3'
    group_id = 3
    manifold_dim = 6
    embedded_dim = 7

    # translation, unit quaternion
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def __init__(self, data):
        if isinstance(data, SO3):
            translation = torch.zeros_like(data.data[...,:3])
            data = torch.cat([translation, data.data], -1)

        super(SE3, self).__init__(data)

    def scale(self, s):
        t, q = self.data.split([3,4], -1)
        t = t * s.unsqueeze(-1)
        return SE3(torch.cat([t, q], dim=-1))


class Sim3(LieGroup):
    group_name = 'Sim3'
    group_id = 4
    manifold_dim = 7
    embedded_dim = 8

    # translation, unit quaternion, scale
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    def __init__(self, data):

        if isinstance(data, SO3):
            scale = torch.ones_like(SO3.data[...,:1])
            translation = torch.zeros_like(SO3.data[...,:3])
            data = torch.cat([translation, SO3.data, scale], -1)

        elif isinstance(data, SE3):
            scale = torch.ones_like(data.data[...,:1])
            data = torch.cat([data.data, scale], -1)

        elif isinstance(data, Sim3):
            data = data.data

        super(Sim3, self).__init__(data)


def cat(group_list, dim):
    """ Concatenate groups along dimension """
    data = torch.cat([X.data for X in group_list], dim=dim)
    return group_list[0].__class__(data)

def stack(group_list, dim):
    """ Concatenate groups along dimension """
    data = torch.stack([X.data for X in group_list], dim=dim)
    return group_list[0].__class__(data)
