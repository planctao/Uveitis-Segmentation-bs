"""Verify real adapter initialization survives unrelated work and resume skips."""
import gc
import json
import random
import numpy as np
import torch
from experiment import ROOT, dump
from lora_seed import construct_seeded_model, adapter_hash


def main():
    torch.set_num_threads(4)
    frozen = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    checkpoint = torch.load(frozen['selected']['checkpoints'][0], map_location='cpu', weights_only=False)
    configs = [{'rank': 4, 'blocks': 2, 'alpha': 8, 'adapter_lr': 1e-5},
               {'rank': 8, 'blocks': 4, 'alpha': 16, 'adapter_lr': 1e-5}]
    records = []
    for config in configs:
        hashes = []
        for seed in range(3):
            rng = 20261124 + seed
            streams = []
            digests = []
            for attempt in range(2):
                # Simulate arbitrary prior trials and skipped completed arms.
                torch.manual_seed(831 + attempt)
                _ = torch.rand(317 + 73 * attempt)
                model = construct_seeded_model(checkpoint, config, rng)
                state = model.delta()
                assert all(not value.count_nonzero() for name, value in state.items() if 'lora_b' in name)
                assert all(value.count_nonzero() for name, value in state.items() if 'lora_a' in name)
                digests.append(adapter_hash(state))
                streams.append((random.random(), float(np.random.random()), float(torch.rand(()))))
                del model, state
                gc.collect()
            assert digests[0] == digests[1]
            assert streams[0] == streams[1]
            hashes.append(digests[0])
            records.append({'rank': config['rank'], 'blocks': config['blocks'], 'seed': rng,
                            'initial_adapter_sha256': digests[0], 'repeat_matches': True})
            print(json.dumps(records[-1]), flush=True)
        assert len(set(hashes)) == 3
    dump(ROOT / 'verification_lora_seed.json', {'passed': True, 'real_models_constructed': 12,
        'checks': ['Actual adapter A and B parameters identical despite different preceding random work',
                   'Three formal seeds produce three distinct nonzero A initializations for each adapter shape',
                   'Every B matrix starts at zero',
                   'Training Python, NumPy and Torch RNG streams reset after construction'],
        'expected_initializations': records})


if __name__ == '__main__':
    main()
