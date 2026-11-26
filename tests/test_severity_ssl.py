import torch
from bs.severity_ssl import dec_target,student_assignments,sinkhorn_assignments,swapped_prediction_loss


def test_dec_target_frequency_correction_and_stop_gradient():
    q=torch.tensor([[.8,.2],[.6,.4],[.1,.9]],dtype=torch.float64,requires_grad=True)
    p=dec_target(q)
    expected=q.detach().square()/q.detach().sum(0)
    expected=expected/expected.sum(1,keepdim=True)
    assert torch.allclose(p,expected)
    assert not p.requires_grad


def test_student_assignment_matches_printed_alpha_one_and_has_centroid_gradients():
    z=torch.tensor([[0.,0.],[1.,0.]],requires_grad=True)
    c=torch.tensor([[0.,0.],[2.,0.]],requires_grad=True)
    q=student_assignments(z,c)
    assert torch.allclose(q[0],torch.tensor([5/6,1/6]))
    q[:,0].sum().backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(c.grad).all()
    assert c.grad.abs().sum()>0


def test_log_sinkhorn_matches_author_normal_domain_formula():
    torch.manual_seed(3)
    logits=torch.randn(23,4,dtype=torch.float64)*.2
    q=torch.exp(logits/.7).T;q/=q.sum()
    for _ in range(3):
        q/=q.sum(1,keepdim=True);q/=4
        q/=q.sum(0,keepdim=True);q/=23
    q=(q*23).T
    assert torch.allclose(sinkhorn_assignments(logits,.7,3),q,atol=1e-12)


def test_sinkhorn_extreme_scores_are_finite_and_sample_normalized():
    logits=torch.tensor([[1000.,-1000.,1.,2.],[-2000.,2000.,2.,1.],[2.,1.,3.,4.]])
    q=sinkhorn_assignments(logits)
    assert torch.isfinite(q).all()
    assert torch.allclose(q.sum(1),torch.ones(3),atol=1e-5)


def test_global_targets_predict_only_other_views():
    outputs=[torch.tensor([[2.,0.]],requires_grad=True) for _ in range(4)]
    targets=[torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]])]
    loss=swapped_prediction_loss(outputs,targets,temperature=1.)
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in outputs)
    # crop0 receives only target1, crop1 only target0, local crops both.
    assert outputs[0].grad[0,0]>0 and outputs[1].grad[0,0]<0
