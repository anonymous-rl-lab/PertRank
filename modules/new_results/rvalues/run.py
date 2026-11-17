#!/usr/bin/env python3
"""Recompute fixed rvalues comparisons from losslessly retained native memberships.

This command recomputes nomination scores and paired uncertainty summaries. It
does not refit the native R model or the baseline effect estimators.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, os, shutil, tarfile
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def compare_numbers(a,b,path='root',errors=None):
    errors=[] if errors is None else errors
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),path
        for k in a:compare_numbers(a[k],b[k],path+'.'+k,errors)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare_numbers(x,y,path+f'[{i}]',errors)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert isinstance(b,(int,float)) and abs(float(a)-float(b))<=1e-12,(path,a,b)
        errors.append(abs(float(a)-float(b)))
    else:assert a==b,(path,a,b)
    return errors

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--keep-inputs',action='store_true',help='Retain the expanded, verified NPY inputs.')
    a=p.parse_args();out=a.out.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory.')
    manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text())
    assert sha(ROOT/manifest['archive'])==manifest['archive_sha256']
    assert sha(ROOT/'scorer.py')==manifest['source_scorer_sha256']
    out.mkdir(parents=True);runtime=out/'expanded_inputs';runtime.mkdir()
    with tarfile.open(ROOT/manifest['archive'],'r:xz') as t:
        for member in t.getmembers():
            q=Path(member.name)
            if q.is_absolute() or '..' in q.parts or not member.isfile():raise ValueError('Unexpected archive member')
            dest=runtime/q;dest.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(member) as f:dest.write_bytes(f.read())
    for n,h in manifest['sha256'].items():assert sha(runtime/n)==h,n
    for row in manifest['array_identity']:
        arr=np.load(runtime/row['file']/(row['key']+'.npy'),allow_pickle=False)
        assert list(arr.shape)==row['shape'] and str(arr.dtype)==row['dtype']
        assert hashlib.sha256(arr.tobytes()).hexdigest()==row['raw_sha256']
    spec=importlib.util.spec_from_file_location('original_fixed_scorer',ROOT/'scorer.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    scorer.ROOT=runtime
    scorer.load=lambda path:{p.stem:np.load(p,allow_pickle=False) for p in sorted(path.glob('*.npy'))}
    summary=scorer.score_all(runtime/'data/membership',out)
    expected=json.loads((ROOT/'expected_summary.json').read_text())
    errors=compare_numbers(expected,summary)
    expected_arrays=json.loads((ROOT/'EXPECTED_ARRAY_HASHES.json').read_text())
    array_checks=[]
    for name,values in expected_arrays.items():
        with np.load(out/name,allow_pickle=False) as z:
            assert set(z.files)==set(values)
            for key,row in values.items():
                arr=z[key]
                equal=(list(arr.shape)==row['shape'] and str(arr.dtype)==row['dtype'] and hashlib.sha256(arr.tobytes()).hexdigest()==row['raw_sha256'])
                array_checks.append({'file':name,'key':key,'bitwise_identical':equal})
                assert equal,(name,key)
    primary=0;maximum=0.0
    names={'D4 Rest':'D4_Rest','D4 Stim':'D4_Stim8hr','External RNA D3':'External_D3_Rest','External RNA D1':'External_D1_Rest','External RNA D2':'External_D2_Rest'}
    for row in csv.DictReader((ROOT/'expected_primary.csv').open()):
        r=summary['tasks'][names[row['task']]];c=r['contrasts']['PertRank_minus_rvalues']
        pairs=[(r['means']['PertRank'],row['PertRank']),(r['means']['rvalues'],row['rvalues']),(c['precision_difference'],row['difference']),(c['conditional_query_block_95_interval'][0],row['ci95_lower']),(c['conditional_query_block_95_interval'][1],row['ci95_upper']),(c['expected_reference_members_difference'],row['expected_member_difference'])]
        for x,y in pairs:maximum=max(maximum,abs(x-float(y)));primary+=1
    assert maximum<1e-12
    record={'status':'PASS','tasks':5,'aggregate_summaries':2,'native_fits_recorded':7742,
        'input_arrays_bitwise_preserved':len(manifest['array_identity']),
        'original_scorer_byte_identical':True,'summary_numeric_values_checked':len(errors),
        'summary_max_abs_error':max(errors),'output_arrays_checked':len(array_checks),
        'output_arrays_bitwise_identical':all(x['bitwise_identical'] for x in array_checks),
        'archived_primary_values_checked':primary,'archived_primary_max_abs_error':maximum,
        'native_model_refitted':False,'baseline_models_refitted':False,
        'scope':'Recomputed five-task membership-overlap scores, all retained baseline comparisons, 2000 paired query-block intervals, equal-state and equal-source aggregates from native memberships and retained baseline query scores.'}
    (out/'REPLAY_CHECK.json').write_text(json.dumps(record,indent=2)+'\n')
    if not a.keep_inputs:shutil.rmtree(runtime)
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
