"""Exercise a real AMP overflow and recovery without moving the LR schedule."""
import torch
from experiment import ROOT,dump
from amp_optimizer import optimizer_update


def main():
    parameter=torch.nn.Parameter(torch.tensor(1.,device='cuda'))
    optimizer=torch.optim.SGD([parameter],lr=.1)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.,growth_interval=100)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=4)
    initial_epoch=scheduler.last_epoch
    scaler.scale(parameter.square()).backward()
    updated,_=optimizer_update(optimizer,scaler,scheduler,[parameter]);assert updated
    assert scheduler.last_epoch==initial_epoch+1
    prior_weight=parameter.detach().clone();prior_epoch=scheduler.last_epoch;prior_lr=optimizer.param_groups[0]['lr']
    scaler.scale(parameter.square()).backward();parameter.grad.fill_(float('inf'))
    updated,norm=optimizer_update(optimizer,scaler,scheduler,[parameter]);assert not updated
    assert torch.equal(parameter,prior_weight) and scheduler.last_epoch==prior_epoch and optimizer.param_groups[0]['lr']==prior_lr
    assert parameter.grad is None
    scaler.scale(parameter.square()).backward();updated,_=optimizer_update(optimizer,scaler,scheduler,[parameter])
    assert updated and scheduler.last_epoch==initial_epoch+2 and torch.isfinite(parameter)
    dump(ROOT/'verification_amp_optimizer.json',{'passed':True,'checks':['Successful AMP updates advance the schedule','Injected overflow leaves both parameters and LR schedule unchanged','Gradient buffers cleared after overflow','The next finite update resumes training and advances the schedule'],'successful_updates':2,'skipped_updates':1})
    print('Real AMP overflow and scheduler recovery checks passed',flush=True)


if __name__=='__main__':main()
