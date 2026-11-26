"""Use the verified RAM-cache producer with the new study's explicit output scope."""
from experiment import ROOT
from large_lastblock_delta_lora_data import load_for_run


def load_original_all():
    return load_for_run('original', ROOT/'runs/large_lastblock_readout_coupling')


def load_fresh_all():
    return load_for_run('fresh', ROOT/'runs/large_lastblock_readout_coupling_confirmation')
