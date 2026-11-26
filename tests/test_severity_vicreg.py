import pytest
import torch
import torch.nn.functional as F
from bs.severity_vicreg import vicreg_loss


def test_vicreg_matches_author_formula_and_gradients():
    torch.manual_seed(7)
    x=torch.randn(17,12,dtype=torch.float64,requires_grad=True)
    y=torch.randn(17,12,dtype=torch.float64,requires_grad=True)
    def off_diagonal(a):
        n=a.shape[0]
        return a.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()
    xc=x-x.mean(0);yc=y-y.mean(0)
    std=(F.relu(1-torch.sqrt(xc.var(0)+.0001)).mean()+F.relu(1-torch.sqrt(yc.var(0)+.0001)).mean())/2
    cov=off_diagonal(xc.T@xc/16).square().sum()/12+off_diagonal(yc.T@yc/16).square().sum()/12
    expected=25*F.mse_loss(x,y)+25*std+cov
    actual,_=vicreg_loss(x,y)
    assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    expected_grad=torch.autograd.grad(expected,(x,y),retain_graph=True)
    actual_grad=torch.autograd.grad(actual,(x,y))
    for a,b in zip(actual_grad,expected_grad):assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)


def test_near_constant_representation_has_finite_variance_expanding_gradient():
    torch.manual_seed(8)
    x=(.01*torch.randn(16,8)).requires_grad_()
    loss,parts=vicreg_loss(x,x)
    gradient=torch.autograd.grad(loss,x)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert parts['variance']>.98
    updated=x-.01*gradient
    assert updated.std(0).mean()>x.std(0).mean()


def test_invalid_batch_rejected_and_half_statistics_promoted():
    with pytest.raises(ValueError):vicreg_loss(torch.ones(1,4),torch.ones(1,4))
    x=torch.randn(8,4,dtype=torch.float16)
    loss,_=vicreg_loss(x,x)
    assert loss.dtype==torch.float32 and torch.isfinite(loss)
