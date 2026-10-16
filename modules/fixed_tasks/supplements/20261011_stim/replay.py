#!/usr/bin/env python3
"""Self-contained, offline, exact replay of the completed Stim fixed endpoints."""
from pathlib import Path
import argparse, hashlib, json, os, socket, sys, time

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent

def require(test, message):
    if not test:
        raise RuntimeError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def deny_network(*args, **kwargs):
    raise RuntimeError('Network is prohibited during the self-contained Stim replay')

def jsonable(value):
    import numpy as np
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

def replay(module_root, output_dir):
    start=time.monotonic()
    root, out=Path(module_root).resolve(),Path(output_dir).resolve()
    require(not out.exists(),'Output directory already exists; refusing overwrite')
    require(not out.is_relative_to(root),'Output must be outside the sealed module directory')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for rel, expected in manifest['file_sha256'].items():
        p=(root/rel).resolve()
        require(p.is_relative_to(root) and p.is_file(),'Missing or unsafe module member '+rel)
        require(sha(p)==expected,'Module hash mismatch '+rel)
    for obj,method in [(socket,'create_connection'),(socket,'getaddrinfo'),
                       (socket.socket,'connect'),(socket.socket,'connect_ex'),(socket.socket,'sendto')]:
        setattr(obj,method,deny_network)
    sys.path.insert(0,str(root/'frozen'));sys.path.insert(0,str(root))
    import numpy as np
    import scipy
    import compact_support as support
    from compact_analysis import analyze
    require(np.__version__=='2.3.5' and scipy.__version__=='1.17.0',
            'Exact archived replay requires numpy==2.3.5 and scipy==1.17.0; no tolerance is widened')
    data=support.load_npz(root/'data/main_support.npz')
    query=support.load_npz(root/'data/main_query_context.npz')
    main=support.reconstruct(data,responsive_positions=query['responsive_positions'])
    xdata=support.load_npz(root/'data/xguide_support.npz')
    xmain=support.reconstruct(xdata)
    for task,reconstructed in [('main',main),('xguide',xmain)]:
        audit=json.loads((root/f'audit/{task}_COMPACTION.json').read_text())
        for kind in ('prediction','truth'):
            key=kind+'_membership_sha256'
            require(reconstructed[key]==audit[key],task+' '+kind+' membership digest changed')
    cross=json.loads((root/'data/crossguide_pretruth_report.json').read_text())
    cross.update(predictions=xmain,query=support.load_npz(root/'data/xguide_query_context.npz'))
    alignment=json.loads((root/'data/alignment_pretruth_report.json').read_text())
    locked=support.load_npz(root/'data/alignment_pretruth.npz')
    alignment.update({key[len('design_'):]:value for key,value in locked.items() if key.startswith('design_')})
    declared=json.loads((root/'data/unavailable_arms.json').read_text())
    unavailable={arm:(v if isinstance(v,str) else v['reason']) for arm,v in declared.items()}
    report, arrays=analyze(main,query,unavailable,cross=cross,alignment=alignment)
    expected=support.load_npz(root/'expected/registered_scores.npz')
    require(set(arrays)==set(expected),'Output array keys differ from completed evaluation')
    counts={}
    for key,a in arrays.items():
        e=expected[key]
        require(a.dtype==e.dtype and a.shape==e.shape,'Array dtype/shape mismatch '+key)
        require(np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(e).tobytes(),
                'Bitwise output mismatch '+key)
        counts[key]=int(a.size)
    expected_report=json.loads((root/'expected/EVALUATION_REPORT.json').read_text())
    serial=json.loads(json.dumps(report,default=jsonable,allow_nan=False))
    for key,value in serial.items():require(value==expected_report[key],'Scientific report mismatch '+key)
    require(expected_report['protocol_status']=='investigator_directed_deviation' and
            expected_report['independent_e4_registration_verified'] is False,
            'Original E4 disclosure has been altered')
    require(serial['H1_H2']['decision']==expected_report['H1_H2']['decision'],'Decision mismatch')
    out.mkdir(parents=True)
    np.savez_compressed(out/'registered_scores.npz',**arrays)
    (out/'scientific_report.json').write_text(json.dumps(serial,indent=2,allow_nan=False)+'\n')
    verification={
        'status':'STIM_EXACT_REPRODUCTION_PASS',
        'scope':'Exact fixed-method weights, signed top-25 memberships, scores, H1/H2 and S1-S3 from sufficient original float64 inputs',
        'module_manifest_sha256':sha(root/'MANIFEST.json'),
        'numpy':np.__version__,'scipy':scipy.__version__,
        'matched_arrays':counts,'all_array_values_bitwise_exact':True,
        'matched_scientific_report_keys':sorted(serial),
        'all_report_fields_exact':True,
        'network_requests':0,'raw_cell_reads':0,'large_external_archive_required':False,
        'weights_reconstructed_from_scalar_moments':True,
        'original_effects_quantized':False,'arbitrary_new_method_support_claimed':False,
        'protocol_status':expected_report['protocol_status'],
        'independent_e4_registration_verified':False,
        'TRADE_scalar':'not_evaluable',
        'elapsed_seconds':time.monotonic()-start}
    (out/'VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2),flush=True)
    return verification

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();replay(ROOT,a.output_dir)

if __name__=='__main__':main()
