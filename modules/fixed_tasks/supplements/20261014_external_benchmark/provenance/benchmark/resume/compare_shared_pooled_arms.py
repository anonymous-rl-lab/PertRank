"""Check that expanded FR uses the same task and three internal arms as rank 100."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original', type=Path)
    parser.add_argument('expanded', type=Path)
    args = parser.parse_args()
    owner = json.loads((args.expanded / 'OWNER_COMPLETION.json').read_text())
    assert owner['owner_exit_code'] == 0
    assert owner['status'] == 'OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED'
    paths = [d / n for d in [args.original, args.expanded]
             for n in ['evaluation_report.json', 'evaluation_arrays.npz', 'compact.npz']]
    before = {str(p.resolve()): sha(p) for p in paths}
    checks = {}
    with np.load(args.original / 'compact.npz', allow_pickle=False) as old, np.load(args.expanded / 'compact.npz', allow_pickle=False) as new:
        for key in ['raw_scores', 'reference_scores', 'w0', 'wm', 'queries', 'candidates', 'blocks', 'input_reps', 'own_target']:
            checks['compact:' + key] = bool(np.array_equal(old[key], new[key]))
        for k in [25, 10]:
            key = f'six_arm_p{k}'
            checks['compact:' + key + ':common_three'] = bool(np.array_equal(old[key][:, :3], new[key][:, :3]))
    with np.load(args.original / 'evaluation_arrays.npz', allow_pickle=False) as old, np.load(args.expanded / 'evaluation_arrays.npz', allow_pickle=False) as new:
        for key in ['sampled_query_blocks', 'query_weights']:
            checks['evaluation:' + key] = bool(np.array_equal(old[key], new[key]))
        for k in [25, 10]:
            key = f'query_only_bootstrap_p{k}'
            checks['evaluation:' + key + ':common_three'] = bool(np.array_equal(old[key][:, :3], new[key][:, :3]))
    assert all(checks.values()), checks
    assert before == {str(p.resolve()): sha(p) for p in paths}
    output = {'status': 'PASS', 'scope': 'Same pooled task, raw/Euclidean/PertRank arms and paired query resampling; not an additional experiment.',
              'checks': checks, 'sources_sha256': before}
    destination = args.expanded / 'COMMON_THREE_ARMS_VS_ORIGINAL.json'
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'path': str(destination), 'checks_passed': len(checks)}))


if __name__ == '__main__':
    main()
