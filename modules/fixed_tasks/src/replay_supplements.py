"""Dispatch the completed supplements in the network-blocked replay process."""
import importlib.util
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(root: Path, output: Path) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError('Use a new output directory outside the repository')
    modules = root / 'supplements' / '20261008'
    d4 = load_module('d4_supplement_replay', modules / 'd4_trade' / 'replay.py')
    d4_result = d4.replay(root, output / 'd4_trade')
    assert d4_result['status'] == 'D4_SUPPLEMENT_REPLAY_PASS'
    print('D4_SUPPLEMENT_PASS', d4_result['query_scores_exact'], 'score values', flush=True)
    ntc = load_module('ntc_supplement_replay', modules / 'ntc' / 'replay.py')
    ntc_result = ntc.replay_to(output / 'ntc')
    assert ntc_result['status'] == 'PASS'
    print('NTC_SUPPLEMENT_PASS', ntc_result['repetitions'], 'resampling repetitions', flush=True)
    return {'status': 'SUPPLEMENTS_PASS', 'D4_TRADE': d4_result, 'D3_NTC': ntc_result,
            'new_data_acquisition': False, 'new_model_fitting': False}
