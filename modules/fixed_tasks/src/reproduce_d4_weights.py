"""Rebuild the four locked D4 weight vectors from frozen sufficient moments."""
from pathlib import Path
import ast
import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def reproduce(root):
    root = Path(root)
    stats = load(root / 'data/D4/weight_sufficient_stats.npz')
    model = load(root / 'data/D4/control_model.npz')
    cohort = load(root / 'data/D4/cohort_core.npz')
    path = root / 'provenance/code/d4_prediction_engine.py'
    tree = ast.parse(path.read_text())
    funcs = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'scalar']
    assert len(funcs) == 1
    ns = {'np': np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), ns)
    scalar = ns['scalar']
    g = len(model['gene_ids_keep']); nc = int(model['n_control']); beta = float(model['beta'])
    assert nc == 2000 and beta == .9 and int(model['k']) == 5
    nt = len(stats['target_ids'])
    checks = []
    for prefix, var_key, lam_key in [('log', 'cv', 'lam'), ('pr', 'pr_var', 'pr_lam')]:
        energy, noise = [stats[prefix + '_' + k] for k in ('aE', 'nuE')]
        pe, pv = [stats[prefix + '_projection_' + k] for k in ('mean', 'variance')]
        reconstructed = {
            'nuE': (stats[prefix + '_trace_target'] / 20 + model[var_key].sum() / nc) / g,
            'aM': energy - beta * np.sum(pe * pe, axis=-1) / g,
            'nuM': noise - beta * np.sum(pv / 20 + model[lam_key][None,None,:] / nc, axis=-1) / g,
        }
        for k, value in reconstructed.items():
            expected = stats[prefix + '_' + k]
            err = float(np.max(np.abs(value - expected)))
            assert np.allclose(value, expected, rtol=0, atol=1e-11), (prefix, k, err)
            checks.append({'quantity': prefix + '_' + k, 'max_abs_error': err})
    weights = np.stack([
        np.ones((5, nt)), scalar(stats['log_aE'], stats['log_nuE']).T,
        scalar(stats['log_aM'], stats['log_nuM']).T,
        scalar(stats['pr_aM'], stats['pr_nuM']).T,
    ])
    cells = stats['cell_ids']
    assert np.array_equal(cohort['input_cells'], cells.reshape(nt,5,20))
    assert np.array_equal(cohort['target_ids'], stats['target_ids'])
    parts = [cohort[k].ravel() for k in ('planning_cells','input_cells','truth_cells','cin','cev')]
    allrows = np.concatenate(parts)
    assert len(np.unique(allrows)) == len(allrows)
    assert np.array_equal(np.sort(allrows), np.sort(cohort['selected_source_rows']))
    assert np.array_equal(model['control_ids'],cohort['cin'])
    for group in range(5):
        added = stats['additions'][:,group]
        assert np.all((added >= 0) & (added < 100))
        assert not np.any((added >= group*20) & (added < (group+1)*20))
        assert np.all(np.diff(np.sort(added,axis=1),axis=1)>0)
    return weights, {'donor':'D4','arm_names':['raw','Euclidean','PertRank','PR_M'],
        'moment_checks':checks,'cell_partition_disjointness_verified':True,
        'control_directions_refitted':False,'raw_UMI_reloaded':False}
