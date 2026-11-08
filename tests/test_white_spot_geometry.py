import torch
from bs.white_spot import gaussian_targets, decode_heatmap, point_loss

def test_subpixel_decode_recovers_interior_points():
    xy=torch.tensor([[.5,.5],[.103,.887],[.76,.24]])
    target=gaussian_targets(xy,(96,96))
    pred,_=decode_heatmap(target.clamp_min(1e-30).log())
    assert torch.max(torch.linalg.vector_norm(pred-xy,dim=1)*768)<.1

def test_flip_decoding_uses_pixel_center_coordinates():
    xy=torch.tensor([[.314,.687]])
    logits=gaussian_targets(xy,(96,96)).clamp_min(1e-30).log()
    pred,_=decode_heatmap(logits.flip((-1,-2)))
    torch.testing.assert_close(pred,1-xy,atol=1e-5,rtol=0)

def test_boundary_and_uniform_logits_remain_finite():
    for x in [torch.zeros(2,96,96),torch.randn(2,96,96)*100]:
        pred,mass=decode_heatmap(x)
        assert torch.isfinite(pred).all() and torch.isfinite(mass).all()
        assert ((pred>=0)&(pred<=1)).all()

def test_point_loss_has_finite_nonzero_gradients():
    logits=torch.zeros(2,96,96,requires_grad=True)
    loss=point_loss(logits,torch.tensor([[.2,.4],[.7,.6]]))
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
