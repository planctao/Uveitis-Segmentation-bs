"""Describe frozen Value fit behavior using saved development outputs only."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from integrity_reporting import load_integrity_snapshot

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/research_register_patch_value/descriptive_adapter_behavior.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'Preserve an existing diagnostic; version any revision.'
    folder = ROOT/'runs/register_patch_value'
    state = json.loads((folder/'state.json').read_text())
    verification = json.loads((ROOT/'results/research_register_patch_value/verification.json').read_text())
    assert state['status'] == 'completed' and verification['passed']
    products_path = folder/'training_products_freeze_before_audit.json'
    products = json.loads(products_path.read_text())['input_artifact_sha256']
    protocol_path = folder/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    inputs = {str(products_path): sha(products_path), str(protocol_path): sha(protocol_path)}
    base_record = load_integrity_snapshot()['base']
    base_path = ROOT/base_record['record']
    assert sha(base_path) == base_record['sha256']
    source_hashes = json.loads(base_path.read_text())['sha256']
    inputs[str(base_path)] = base_record['sha256']

    def frozen_read(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is None:
            expected = products.get(str(path), products.get(str(path.relative_to(ROOT))))
        assert expected is not None and actual == expected, str(path)
        inputs[str(path)] = actual

    records = []
    for row in verification['fits']:
        seed_folder = folder/row['arm']/row['phase']/f'seed_{row["seed"]}'
        for name in ['model.pt', 'val_predictions.npz', 'summary.json', 'history.json']:
            frozen_read(seed_folder/name)
        checkpoint = torch.load(seed_folder/'model.pt', map_location='cpu', weights_only=False)
        summary = json.loads((seed_folder/'summary.json').read_text())
        history = json.loads((seed_folder/'history.json').read_text())
        source_folder = Path(checkpoint['source_checkpoint']).parent
        source_probability = source_folder/'val_predictions.npz'
        expected = source_hashes[str(source_probability)]
        frozen_read(source_probability, expected)
        with np.load(seed_folder/'val_predictions.npz') as saved:
            p = saved['p'].copy()
        with np.load(source_probability) as saved:
            source_p = saved['p'].copy()
        assert p.shape == source_p.shape and p.shape[1] == 3
        assert np.isfinite(p).all() and np.isfinite(source_p).all()
        epoch = checkpoint['epoch']
        assert epoch == summary['epoch'] and epoch in [r['epoch'] for r in history]
        up = checkpoint['value_state_dict']['value_up']
        down = checkpoint['value_state_dict']['value_down']
        assert torch.isfinite(up).all() and torch.isfinite(down).all()
        if epoch == 0:
            assert torch.count_nonzero(up) == 0
        difference = np.abs(p-source_p)
        records.append({
            'arm': row['arm'], 'phase': row['phase'], 'seed': row['seed'],
            'selected_epoch': epoch, 'last_training_epoch': history[-1]['epoch'],
            'stop_images': len(p),
            'stop_decision_change_events': int((p.argmax(1) != source_p.argmax(1)).sum()),
            'mean_absolute_saved_stop_probability_difference': float(difference.mean()),
            'maximum_absolute_saved_stop_probability_difference': float(difference.max()),
            'nonzero_value_up_elements': int(torch.count_nonzero(up)),
            'value_up_frobenius_norm': float(torch.linalg.vector_norm(up)),
            'effective_value_matrix_frobenius_norm': float(torch.linalg.vector_norm(up@down)),
            'checkpoint_sha256': inputs[str(seed_folder/'model.pt')],
            'source_probability_sha256': inputs[str(source_probability)],
        })
    assert len(records) == 48
    summaries = {}
    for arm in sorted({row['arm'] for row in records}):
        rows = [row for row in records if row['arm'] == arm]
        assert len(rows) == 12
        summaries[arm] = {
            'fits': len(rows),
            'selected_epoch_histogram': dict(sorted(Counter(row['selected_epoch'] for row in rows).items())),
            'selected_after_epoch_zero': sum(row['selected_epoch'] > 0 for row in rows),
            'nonzero_selected_value_up_fits': sum(row['nonzero_value_up_elements'] > 0 for row in rows),
            'stop_decision_change_events_across_fits': sum(row['stop_decision_change_events'] for row in rows),
            'maximum_saved_stop_probability_change': max(row['maximum_absolute_saved_stop_probability_difference'] for row in rows),
        }
    for name, expected in inputs.items():
        assert sha(name) == expected, name
    document = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Descriptive inspection of selected frozen Value fits using saved development stop probabilities. These stop images selected checkpoints; changes are not independent generalization evidence. Change events sum repeated images across fits, not distinct patients. Saved source CPU/CUDA numerical differences can contribute to probability deltas.',
        'new_classification_fits': 0, 'new_classification_scores': 0,
        'new_model_predictions': 0, 'external_test_evaluated': False,
        'summary': summaries, 'records': records, 'input_sha256': inputs,
        'diagnostic_code_sha256': sha(__file__),
    }
    OUT.write_text(json.dumps(document, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
