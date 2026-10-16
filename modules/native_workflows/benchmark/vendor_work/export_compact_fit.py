"""Lossless frozen-query scoring export; keeps native factors for full reconstruction.

Dense original outputs remain untouched. This is explicitly not a copy of every
intermediate: B and U_tilde are omitted, U/W and native scaling are retained.
"""
from pathlib import Path
import argparse, json, shutil
import numpy as np
from threadpoolctl import threadpool_limits
from fr_native_runner import sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit-dir',required=True)
    p.add_argument('--axes',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args(); fit=Path(a.fit_dir); out=Path(a.out)
    report=json.loads((fit/'fit_report.json').read_text())
    if report['status']!='COMPLETE': raise ValueError('Only completed fits may be exported')
    if sha256(fit/'effects.npz')!=report['effects_sha256']: raise AssertionError('source effects hash mismatch')
    if sha256(fit/'native_unscaled.npz')!=report['native_unscaled_sha256']: raise AssertionError('source factors hash mismatch')
    z=np.load(fit/'effects.npz',allow_pickle=False)
    n=np.load(fit/'native_unscaled.npz',allow_pickle=False)
    ax=np.load(a.axes,allow_pickle=False)
    expected_axes=(report.get('input_provenance') or {}).get('axes_sha256')
    if expected_axes is not None and sha256(a.axes)!=expected_axes:
        raise AssertionError('input axes changed since native fitting')
    genes=z['genes']; targets=z['targets']; queries=ax['query_ids']
    if not np.array_equal(targets,ax['target_ids']): raise AssertionError('target axes mismatch')
    if not np.array_equal(genes,ax['gene_ids']): raise AssertionError('gene axes mismatch')
    if not np.array_equal(z['target_index'],ax['labels']): raise AssertionError('cell labels mismatch')
    if not np.array_equal(z['cell_ids'],ax['cell_ids'].astype(str)): raise AssertionError('cell IDs mismatch')
    idx={g:i for i,g in enumerate(genes)}
    qcols=np.array([idx[g] for g in queries],dtype=np.int64)
    U=n['U']; W=n['W']; original=n['B']
    with threadpool_limits(limits=2): reconstructed=U.dot(W)
    max_error=float(np.max(np.abs(original-reconstructed)))
    if not np.allclose(original,reconstructed,rtol=1e-12,atol=1e-12):
        raise AssertionError('stored native factors fail reconstruction')
    reconstruction_bitwise=bool(np.array_equal(original,reconstructed))
    del original,reconstructed
    keep=z['retained_gene_index']; sf=z['scale_factors']
    contents=dict(U=U,W=W,scale_factors=sf,genes=genes,targets=targets,
         retained_gene_index=keep,retained_genes=n['genes'],
         perturbation_names=n['perturbation_names'],cell_ids=z['cell_ids'],
         target_index=z['target_index'],query_ids=queries,query_columns=qcols,
         query_blocks=ax['query_blocks'],query_effect=z['effect'][:,qcols],
         query_effect_unscaled=z['effect_unscaled'][:,qcols],
         input_effects=ax['input_effects'],weights_euclidean=ax['weights_euclidean'],
         weights_pertrank=ax['weights_pertrank'],ntc_mean=n['ntc_mean'],
         logmeanexp=n['logmeanexp'])
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'compact.npz',**contents)
    c=np.load(out/'compact.npz',allow_pickle=False)
    for key,val in contents.items():
        if not np.array_equal(c[key],val): raise AssertionError('compact serialization changed '+key)
    for name in ['fit_report.json','fit.log']:
        shutil.copy2(fit/name,out/name)
    manifest=dict(format='lossless_frozen_query_scoring_plus_native_factor_reconstruction',
        exporter_sha256=sha256(__file__),
        original_fit_directory=str(fit.resolve()),source_axes=str(Path(a.axes).resolve()),
        source_axes_sha256=sha256(a.axes),source_effects_sha256=report['effects_sha256'],
        source_native_unscaled_sha256=report['native_unscaled_sha256'],
        compact_sha256=sha256(out/'compact.npz'),
        full_B_reconstruction_max_absolute_error=max_error,
        full_B_reconstruction_bitwise_equal=reconstruction_bitwise,
        all_compact_arrays_bitwise_verified=True,
        omitted=['dense full-gene scaled effects','dense full-gene unscaled B','cell latent U_tilde'],
        preserved=['native U and W','all native scale factors','full axes and cell IDs/labels',
                   'all frozen query effects at original precision','raw query effects and fixed weights',
                   'fit configuration, source hashes, warnings and runtime logs'],
        note='Original dense files remain untouched in the execution workspace; source SHA fields do not imply those files are included in this compact export.')
    (out/'COMPACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'README_COMPACT.txt').write_text('This is a compact scoring-reproduction export, not every dense intermediate.\n'
        'query_effect and query_effect_unscaled are exact original floating-point values, with no rounding.\n'
        'Six arms can be reconstructed using input_effects, weights_euclidean, weights_pertrank and query_effect.\n'
        'Full native B (including final NTC row) is U @ W on retained_genes. Divide columns by scale_factors,\n'
        'retain target rows, and embed by retained_gene_index into the full genes axis to reconstruct full effects.\n'
        'The recorded reconstruction check uses the same numerical environment and BLAS thread setting.\n'
        'Query scoring reproduction is independent of possible cross-platform BLAS rounding in full reconstruction.\n'
        'Original full effect archives and cell-latent U_tilde are omitted, explicitly recorded in COMPACT_MANIFEST.json.\n')
    print(json.dumps({'output':str(out),'size_bytes':(out/'compact.npz').stat().st_size,
                      'reconstruction_error':max_error,'reconstruction_bitwise':reconstruction_bitwise}))

if __name__=='__main__': main()
