"""Explicitly recover four mutable-reporting inventory entries, preserving all scientific bytes."""
import json
from pathlib import Path
from experiment import ROOT
from kernel_residual_experiments import sha
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
RECOVERY='results/dispersion_scientific_code_inventory_recovery_v2.json'
RECOVERY_SHA='84baa373b7ef8ad64faecfe2be63aab7e2b26104df2713d79dcdadfeec121a6a'

def verified_recovery():
    assert sha(ROOT/RECOVERY)==RECOVERY_SHA
    record=json.loads((ROOT/RECOVERY).read_text());assert record['passed']
    assert set(record['excluded_mutable_reporting'])==EXCLUDED
    for name,expected in record['scientific_code_sha256'].items():assert sha(ROOT/name)==expected,name
    for name,expected in record['input_artifact_sha256'].items():assert sha(name)==expected,name
    return record

def verified_proof_code(name,proof):
    recovery=verified_recovery();target=str(ROOT/name)
    assert target in recovery['input_artifact_sha256'] and sha(target)==recovery['input_artifact_sha256'][target]
    assert proof==json.loads(Path(target).read_text())
    result={}
    for path,expected in proof['code_sha256'].items():
        if path in EXCLUDED:
            assert expected==recovery['excluded_mutable_reporting'][path]['frozen_sha256'],path
        else:
            assert expected==recovery['scientific_code_sha256'][path] and sha(ROOT/path)==expected,path
            result[path]=expected
    return result
