"""Check graph adjacency, mask isolation, neutral initialization and gradients."""
import io
import json
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, dump
from spatial_graph import SpatialGraph


def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    x = torch.arange(9.).reshape(1, 9, 1)
    expected = []
    for row in range(3):
        for col in range(3):
            neighbors = [x[0, r*3+c, 0] for r, c in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)] if 0 <= r < 3 and 0 <= c < 3]
            expected.append(torch.stack(neighbors).mean() - x[0, row*3+col, 0])
    assert torch.allclose(SpatialGraph.neighbor_difference(x).flatten(), torch.stack(expected))
    tokens = torch.randn(2, 9, 128)
    mask = torch.tensor([[1.,1.,0.,1.,1.,0.,1.,0.,0.]]).repeat(2,1)
    changed = tokens + (1-mask[:,:,None])*1000
    a, b = [SpatialGraph.neighbor_difference(value,mask) for value in [tokens,changed]]
    assert torch.allclose(a[mask.bool()], b[mask.bool()])
    assert not SpatialGraph.neighbor_difference(tokens, torch.zeros_like(mask)).count_nonzero()
    image = tokens.reshape(2,3,3,128)
    rotated = torch.rot90(image,1,[1,2]).reshape(2,9,128)
    difference = SpatialGraph.neighbor_difference(tokens).reshape(2,3,3,128)
    assert torch.allclose(SpatialGraph.neighbor_difference(rotated), torch.rot90(difference,1,[1,2]).reshape(2,9,128), atol=1e-6)
    for steps in [1,2]:
        graph = SpatialGraph(128,32,steps)
        assert torch.equal(graph(tokens),tokens)
        optimizer = torch.optim.SGD(graph.parameters(),lr=.1)
        target = torch.randn_like(tokens)
        for update in range(2):
            optimizer.zero_grad();loss=(graph(tokens)-target).square().mean();loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in graph.parameters())
            assert graph.outgoing.weight.grad.abs().sum() > 0
            if update == 1:assert graph.incoming.weight.grad.abs().sum() > 0
            optimizer.step()
        assert torch.equal(graph(tokens,torch.zeros_like(mask)),tokens)
        buffer=io.BytesIO();torch.save(graph.state_dict(),buffer);buffer.seek(0)
        restored=SpatialGraph(128,32,steps);restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
        assert torch.equal(graph(tokens),restored(tokens))
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(reference['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit');indices=np.flatnonzero(frame.split.eq('val'))[:7]
    values,_=normalize({key:value[indices] for key,value in features.items()},None,c['stats'])
    inputs={key:torch.from_numpy(value) for key,value in values.items()}
    for steps in [1,2]:
        torch.manual_seed(77);base=Model(c['cfg'],c['dims'],len(c['diseases'])).eval()
        torch.manual_seed(77);graph=Model({**c['cfg'],'region_graph_steps':steps},c['dims'],len(c['diseases'])).eval()
        for key,value in base.state_dict().items():assert torch.equal(graph.state_dict()[key],value)
        base_p=base(inputs)['p']; graph_p=graph(inputs)['p']
        assert torch.equal(base_p,graph_p), f'Initial prediction difference {float((base_p-graph_p).abs().max())}'
    dump(ROOT/'verification_spatial_graph.json',{'passed':True,'manifest_sha256':digest,
        'checks':['Exact four-connected neighbor averaging with correct image boundaries',
            'Masked neighbors cannot change valid messages and isolated regions get zero neighbor difference',
            'Neighbor differences equivariant to90degreegridrotations',
            'Every one/two-step graph initially exactly preserves tokens and common full-model predictions',
            'Outgoing layer gets first-step gradient and incoming layer gets second-step gradient',
            'Full invalid masks preserve original tokens without NaNs',
            'Strict saved/restored graph outputs identical']})
    print('Spatial graph adjacency/mask/gradient/neutrality checks passed',flush=True)


if __name__=='__main__':main()
