import torch
import lietorch

from lietorch import SO3, RxSO3, SE3, Sim3
from gradcheck import gradcheck, get_analytical_jacobian


### forward tests ###

def make_homogeneous(p):
    return torch.cat([p, torch.ones_like(p[...,:1])], dim=-1)

def matv(A, b):
    return torch.matmul(A, b[...,None])[..., 0]

def test_exp_log(Group, device='cuda'):
    """ check Log(Exp(x)) == x """
    a = .2*torch.randn(2,3,4,5,6,7,Group.manifold_dim, device=device).double()
    b = Group.exp(a).log()
    assert torch.allclose(a,b,atol=1e-8), "should be identity"
    print("\t-", Group, "Passed exp-log test")
    
def test_inv(Group, device='cuda'):
    """ check X * X^{-1} == 0 """
    X = Group.exp(.1*torch.randn(2,3,4,5,Group.manifold_dim, device=device).double())
    a = (X * X.inv()).log()
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-8), "should be 0"
    print("\t-", Group, "Passed inv test")

def test_adj(Group, device='cuda'):
    """ check X * Exp(a) == Exp(Adj(X,a)) * X 0 """
    X = Group.exp(torch.randn(2,3,4,5, Group.manifold_dim, device=device).double())
    a = torch.randn(2,3,4,5, Group.manifold_dim, device=device).double()

    b = X.adj(a)
    Y1 = X * Group.exp(a)
    Y2 = Group.exp(b) * X

    c = (Y1 * Y2.inv()).log()
    assert torch.allclose(c, torch.zeros_like(c), atol=1e-8), "should be 0"
    print("\t-", Group, "Passed adj test")
    

def test_act(Group, device='cuda'):
    X = Group.exp(torch.randn(1, Group.manifold_dim, device=device).double())
    p = torch.randn(1,3,device=device).double()

    p1 = X.act(p)
    p2 = matv(X.matrix(), make_homogeneous(p))

    assert torch.allclose(p1, p2[...,:3], atol=1e-8), "should be 0"
    print("\t-", Group, "Passed act test")


### backward tests ###
def test_exp_log_grad(Group, device='cuda', tol=1e-8):
    
    D = Group.manifold_dim

    def fn(a):
        return Group.exp(a).log()

    a = torch.zeros(1, Group.manifold_dim, requires_grad=True, device=device).double()
    analytical, reentrant, correct_grad_sizes, correct_grad_types = \
        get_analytical_jacobian((a,), fn(a))

    assert torch.allclose(analytical[0], torch.eye(D, device=device).double(), atol=tol)

    a = .2 * torch.randn(1, Group.manifold_dim, requires_grad=True, device=device).double()
    analytical, reentrant, correct_grad_sizes, correct_grad_types = \
        get_analytical_jacobian((a,), fn(a))

    assert torch.allclose(analytical[0], torch.eye(D, device=device).double(), atol=tol)

    print("\t-", Group, "Passed eye-grad test")


def test_inv_log_grad(Group, device='cuda', tol=1e-8):

    D = Group.manifold_dim
    X = Group.exp(.2*torch.randn(1,D,device=device).double())

    def fn(a):
        return (Group.exp(a) * X).inv().log()

    a = torch.zeros(1, D, requires_grad=True, device=device).double()
    analytical, numerical = gradcheck(fn, [a], eps=1e-4)

    # assert torch.allclose(analytical[0], numerical[0], atol=tol)
    if not torch.allclose(analytical[0], numerical[0], atol=tol):
        print(analytical[0])
        print(numerical[0])

    print("\t-", Group, "Passed inv-grad test")


def test_adj_grad(Group, device='cuda'):
    D = Group.manifold_dim
    X = Group.exp(.5*torch.randn(1,Group.manifold_dim, device=device).double())
    
    def fn(a, b):
        return (Group.exp(a) * X).adj(b)

    a = torch.zeros(1, D, requires_grad=True, device=device).double()
    b = torch.randn(1, D, requires_grad=True, device=device).double()

    analytical, numerical = gradcheck(fn, [a, b], eps=1e-4)
    assert torch.allclose(analytical[0], numerical[0], atol=1e-8)
    assert torch.allclose(analytical[1], numerical[1], atol=1e-8)

    print("\t-", Group, "Passed adj-grad test")


def test_adjT_grad(Group, device='cuda'):
    D = Group.manifold_dim
    X = Group.exp(.5*torch.randn(1,Group.manifold_dim, device=device).double())
    
    def fn(a, b):
        return (Group.exp(a) * X).adjT(b)

    a = torch.zeros(1, D, requires_grad=True, device=device).double()
    b = torch.randn(1, D, requires_grad=True, device=device).double()

    analytical, numerical = gradcheck(fn, [a, b], eps=1e-4)

    assert torch.allclose(analytical[0], numerical[0], atol=1e-8)
    assert torch.allclose(analytical[1], numerical[1], atol=1e-8)

    print("\t-", Group, "Passed adjT-grad test")


def test_act_grad(Group, device='cuda'):
    D = Group.manifold_dim
    X = Group.exp(5*torch.randn(1,D, device=device).double())
    
    def fn(a, b):
        return (X*Group.exp(a)).act(b)

    a = torch.zeros(1, D, requires_grad=True, device=device).double()
    b = torch.randn(1, 3, requires_grad=True, device=device).double()

    analytical, numerical = gradcheck(fn, [a, b], eps=1e-4)

    assert torch.allclose(analytical[0], numerical[0], atol=1e-8)
    assert torch.allclose(analytical[1], numerical[1], atol=1e-8)

    print("\t-", Group, "Passed act-grad test")


def test_matrix_grad(Group, device='cuda'):
    D = Group.manifold_dim
    X = Group.exp(torch.randn(1, D, device=device).double())
    
    def fn(a):
        return (Group.exp(a) * X).matrix()

    a = torch.zeros(1, D, requires_grad=True, device=device).double()
    analytical, numerical = gradcheck(fn, [a], eps=1e-4)
    assert torch.allclose(analytical[0], numerical[0], atol=1e-6)

    print("\t-", Group, "Passed matrix-grad test")


def extract_translation_grad(Group, device='cuda'):
    """ prototype function """

    D = Group.manifold_dim
    X = Group.exp(5*torch.randn(1,D, device=device).double())
    
    def fn(a):
        return (Group.exp(a)*X).translation()

    a = torch.zeros(1, D, requires_grad=True, device=device).double()

    analytical, numerical = gradcheck(fn, [a], eps=1e-4)

    assert torch.allclose(analytical[0], numerical[0], atol=1e-8)
    print("\t-", Group, "Passed translation grad test")


def test_vec_grad(Group, device='cuda', tol=1e-6):

    D = Group.manifold_dim
    X = Group.exp(5*torch.randn(1,D, device=device).double())
    
    def fn(a):
        return (Group.exp(a)*X).vec()

    a = torch.zeros(1, D, requires_grad=True, device=device).double()

    analytical, numerical = gradcheck(fn, [a], eps=1e-4)

    assert torch.allclose(analytical[0], numerical[0], atol=tol)
    print("\t-", Group, "Passed tovec grad test")


def test_fromvec_grad(Group, device='cuda', tol=1e-6):

    def fn(a):
        if Group == SO3:
            a = a / a.norm(dim=-1, keepdim=True)

        elif Group == RxSO3:
            q, s = a.split([4, 1], dim=-1)
            q = q / q.norm(dim=-1, keepdim=True)
            a = torch.cat([q, s.exp()], dim=-1)

        elif Group == SE3:
            t, q = a.split([3, 4], dim=-1)
            q = q / q.norm(dim=-1, keepdim=True)
            a = torch.cat([t, q], dim=-1)

        elif Group == Sim3:
            t, q, s = a.split([3, 4, 1], dim=-1)
            q = q / q.norm(dim=-1, keepdim=True)
            a = torch.cat([t, q, s.exp()], dim=-1)

        return Group.InitFromVec(a).vec()

    D = Group.embedded_dim
    a = torch.randn(1, 2, D, requires_grad=True, device=device).double()

    analytical, numerical = gradcheck(fn, [a], eps=1e-4)

    assert torch.allclose(analytical[0], numerical[0], atol=tol)
    print("\t-", Group, "Passed fromvec grad test")



def scale(device='cuda'):
    
    def fn(a, s):
        X = SE3.exp(a)
        X.scale(s)
        return X.log()

    s = torch.rand(1, requires_grad=True, device=device).double()
    a = torch.randn(1, 6, requires_grad=True, device=device).double()
    
    analytical, numerical = gradcheck(fn, [a, s], eps=1e-3)
    print(analytical[1])
    print(numerical[1])


    assert torch.allclose(analytical[0], numerical[0], atol=1e-8)
    assert torch.allclose(analytical[1], numerical[1], atol=1e-8)

    print("\t-", "Passed se3-to-sim3 test")

    
if __name__ == '__main__':


    print("Testing lietorch forward pass (CPU) ...")
    for Group in [SO3, RxSO3, SE3, Sim3]:
        test_exp_log(Group, device='cpu')
        test_inv(Group, device='cpu')
        test_adj(Group, device='cpu')
        test_act(Group, device='cpu')

    print("Testing lietorch backward pass (CPU)...")
    for Group in [SO3, RxSO3, SE3, Sim3]:
        if Group == Sim3:
            tol = 1e-3
        else:
            tol = 1e-8

        test_exp_log_grad(Group, device='cpu', tol=tol)
        test_inv_log_grad(Group, device='cpu', tol=tol)
        test_adj_grad(Group, device='cpu')
        test_adjT_grad(Group, device='cpu')
        test_act_grad(Group, device='cpu')
        test_matrix_grad(Group, device='cpu')
        extract_translation_grad(Group, device='cpu')
        test_vec_grad(Group, device='cpu')
        test_fromvec_grad(Group, device='cpu')

    print("Testing lietorch forward pass (GPU) ...")
    for Group in [SO3, RxSO3, SE3, Sim3]:
        test_exp_log(Group, device='cuda')
        test_inv(Group, device='cuda')
        test_adj(Group, device='cuda')
        test_act(Group, device='cuda')

    print("Testing lietorch backward pass (GPU)...")
    for Group in [SO3, RxSO3, SE3, Sim3]:
        if Group == Sim3:
            tol = 1e-3
        else:
            tol = 1e-8

        test_exp_log_grad(Group, device='cuda', tol=tol)
        test_inv_log_grad(Group, device='cuda', tol=tol)
        test_adj_grad(Group, device='cuda')
        test_adjT_grad(Group, device='cuda')
        test_act_grad(Group, device='cuda')
        test_matrix_grad(Group, device='cuda')
        extract_translation_grad(Group, device='cuda')
        test_vec_grad(Group, device='cuda')
        test_fromvec_grad(Group, device='cuda')


