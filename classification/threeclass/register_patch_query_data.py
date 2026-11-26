"""Frozen development bank loader and exact saved-tail reconstruction."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from timm.layers import Mlp
from experiment import ROOT
from register_multiquery_data import load_all as load_original
from register_patch_query import FrozenDinoTail
from register_patch_query_features import BANK_KEYS
from kernel_residual_experiments import sha


def load_tail(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    tail=FrozenDinoTail.__new__(FrozenDinoTail);nn.Module.__init__(tail)
    tail.proj=nn.Linear(1024,1024);tail.norm2=nn.LayerNorm(1024,eps=1e-5);tail.mlp=Mlp(1024,4096,act_layer=nn.GELU);tail.norm=nn.LayerNorm(1024,eps=1e-5)
    tail.register_buffer('gamma1',torch.zeros(1024));tail.register_buffer('gamma2',torch.zeros(1024));tail.load_state_dict(state,strict=True)
    for parameter in tail.parameters():parameter.requires_grad_(False)
    return tail.eval()


def load_all():
    frame,features,digest=load_original();out=ROOT/'runs/register_patch_query_bank'
    proof=json.loads((ROOT/'verification_register_patch_query_bank.json').read_text())
    assert proof['passed'] and proof['manifest_sha256']==digest and proof['classification_fits']==proof['classification_scores']==proof['external_images_read']==0
    assert sha(out/'protocol.json')==proof['protocol_sha256']
    for name,expected in proof['frozen_artifacts'].items():assert sha(name)==expected,name
    bank={name:np.load(out/'arrays'/(name+'.npy'),mmap_mode='r') for name in BANK_KEYS}
    for name,value in bank.items():assert list(value.shape)==proof['shapes'][name] and value.dtype==np.dtype(proof['dtypes'][name])
    return frame,features,bank,load_tail(out/'frozen_tail.pt'),digest
