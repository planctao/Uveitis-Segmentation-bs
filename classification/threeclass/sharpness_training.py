"""SAM gradients with exact parameter restoration and paired dropout randomness."""
import torch
import torch.nn.functional as F


def flat_parameters(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def flat_gradients(model):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad.detach()).flatten() for p in model.parameters()])


def flat_activity(model):
    return torch.cat([torch.full((p.numel(),),p.grad is not None,device=p.device,dtype=torch.bool) for p in model.parameters()])


def sharpness_backward(model, objective, rho, witness=False):
    """Leave second-pass gradients on model, at the exactly restored base parameters."""
    assert rho>=0
    params=list(model.parameters());device=params[0].device
    assert device.type=='cuda' and all(p.device==device for p in params)
    rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state(device)
    proof={};handle=None
    if witness:
        proof['before']=flat_parameters(model).cpu().numpy()
        def capture_mask(module,args):
            # Generate the exact upcoming mask without advancing any global stream.
            with torch.random.fork_rng(devices=[device.index]):
                mask=F.dropout(torch.ones_like(args[0]),p=module.p,training=True)
            key='dropout_mask_first' if 'dropout_mask_first' not in proof else 'dropout_mask_second'
            proof[key]=mask.detach().cpu().numpy()
        handle=model.encoder[2].register_forward_pre_hook(capture_mask)
    loss=objective();assert torch.isfinite(loss)
    loss.backward()
    first_cpu=torch.get_rng_state();first_cuda=torch.cuda.get_rng_state(device)
    norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().norm(2) for p in params if p.grad is not None]),2)
    assert torch.isfinite(norm)
    perturbations=[];backups=[]
    if witness:
        proof['gradient_first']=flat_gradients(model).cpu().numpy();proof['activity_first']=flat_activity(model).cpu().numpy()
    if rho:
        with torch.no_grad():
            for p in params:
                backups.append(p.detach().clone())
                delta=torch.zeros_like(p) if p.grad is None else p.grad*(rho/(norm+1e-12))
                p.add_(delta);perturbations.append(delta)
        if witness:
            proof['delta']=torch.cat([v.flatten() for v in perturbations]).cpu().numpy()
            proof['perturbed']=flat_parameters(model).cpu().numpy()
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda,device)
        try:
            sharp=objective();assert torch.isfinite(sharp);sharp.backward()
        finally:
            with torch.no_grad():
                for p,base in zip(params,backups):p.copy_(base)
        assert torch.equal(first_cpu,torch.get_rng_state()) and torch.equal(first_cuda,torch.cuda.get_rng_state(device))
    else:
        sharp=loss
        if witness:
            proof['delta']=torch.zeros_like(flat_parameters(model)).cpu().numpy();proof['perturbed']=proof['before'].copy()
            proof['dropout_mask_second']=proof['dropout_mask_first'].copy()
    if handle is not None:handle.remove()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
    if witness:
        proof['restored']=flat_parameters(model).cpu().numpy()
        proof['gradient_second']=flat_gradients(model).cpu().numpy();proof['activity_second']=flat_activity(model).cpu().numpy()
        proof['rng_before_cpu']=rng_cpu.numpy();proof['rng_before_cuda']=rng_cuda.cpu().numpy()
        proof['rng_after_first_cpu']=first_cpu.numpy();proof['rng_after_first_cuda']=first_cuda.cpu().numpy()
        proof['rng_after_second_cpu']=torch.get_rng_state().numpy();proof['rng_after_second_cuda']=torch.cuda.get_rng_state(device).cpu().numpy()
    return {'base_loss':float(loss.detach()),'sharp_loss':float(sharp.detach()),'first_gradient_norm':float(norm),
            'rho':rho,'paired_rng_equal':True},proof
