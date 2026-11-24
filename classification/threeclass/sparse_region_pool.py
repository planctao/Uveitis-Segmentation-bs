"""Parameter-matched softmax temperature and known sparse attention controls."""
import torch
import torch.nn.functional as F
from experiment import Model


def region_distribution(scores, kind='softmax', temperature=1., mask=None):
    assert kind in ['softmax', 'entmax15', 'sparsemax'] and temperature > 0
    valid = torch.ones(scores.shape[:2], dtype=torch.bool, device=scores.device) if mask is None else mask > .5
    valid = torch.where(valid.any(1, keepdim=True), valid, torch.ones_like(valid))
    x = (scores / temperature).masked_fill(~valid[:, :, None], -1e4)
    if kind == 'softmax': return x.softmax(1)
    x = x - x.max(1, keepdim=True).values
    if kind == 'entmax15': x = x / 2
    ordered = x.sort(1, descending=True).values
    rho = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)[None, :, None]
    mean = ordered.cumsum(1) / rho
    if kind == 'sparsemax':
        threshold = mean - 1 / rho
    else:
        variance_sum = (ordered.square().cumsum(1) - rho * mean.square()).clamp(min=0)
        discriminant = (1 - variance_sum) / rho
        # Unselected support candidates can have a zero discriminant. Their
        # infinite sqrt derivative must not produce 0*inf during gather backward.
        root = torch.where(discriminant > 0, discriminant.clamp(min=1e-12).sqrt(), torch.zeros_like(discriminant))
        threshold = mean - root
    support = (threshold <= ordered).sum(1, keepdim=True).clamp(min=1)
    tau = threshold.gather(1, support - 1)
    p = (x - tau).clamp(min=0)
    if kind == 'entmax15': p = p.square()
    return p / p.sum(1, keepdim=True).clamp(min=1e-12)


class SparseRegionModel(Model):
    def __init__(self, cfg, dims, n_diseases):
        super().__init__(cfg, dims, n_diseases)
        assert cfg['arch'] == 'attention' and cfg['mode'] == 'flat' and cfg['taxonomy_only']
        assert not cfg.get('context_query') and not cfg.get('region_graph_steps')
        self.region_gate.register_forward_pre_hook(self._tokens)
        self.region_gate.register_forward_hook(self._scores)
        self.encoder.register_forward_pre_hook(self._replace_pooled)

    def _tokens(self, module, args): self._pool_tokens = args[0]

    def _scores(self, module, args, output): self._pool_scores = output

    def _replace_pooled(self, module, args):
        self._sparse_attention = region_distribution(self._pool_scores, self.cfg['region_pool_kind'],
            self.cfg['region_pool_temperature'], self._pool_mask)
        pooled = torch.einsum('btk,btd->bkd', self._sparse_attention, self._pool_tokens).flatten(1)
        z = args[0]; count = pooled.shape[-1]
        return (torch.cat([z[:, :256], pooled, z[:, 256 + count:]], -1),)

    def forward(self, x):
        self._pool_mask = x.get('region_mask') if self.cfg.get('use_region_mask', True) else None
        out = super().forward(x)
        out.update(pool_scores=self._pool_scores, pool_tokens=self._pool_tokens, pool_attention=self._sparse_attention)
        return out
