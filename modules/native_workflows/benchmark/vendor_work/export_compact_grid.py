"""Sequential lossless compact export of every completed registered D3 fit."""
from pathlib import Path
import argparse, json, subprocess, sys, time
from fr_native_runner import sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--grid-dir',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args(); grid=Path(a.grid_dir); out=Path(a.out)
    full=json.loads((grid/'GRID_EXECUTION.json').read_text())
    if full['status']!='COMPLETE' or not all(x['status']=='COMPLETE' for x in full['runs']):
        raise ValueError('A complete grid is required before final compact export')
    axes=Path(full['input_dir'])/'axes.npz'
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','original_grid_sha256':sha256(grid/'GRID_EXECUTION.json'),
        'original_spec_sha256':full['spec_sha256'],'input_axes_sha256':sha256(axes),
        'driver_sha256':sha256(__file__),'runs':[]}
    started=time.time()
    for row in full['runs']:
        name=row['id']+'_rep0'; dest=out/name; tick=time.time()
        cmd=[sys.executable,str(Path(__file__).resolve().parent/'export_compact_fit.py'),
             '--fit-dir',str(grid/name),'--axes',str(axes),'--out',str(dest)]
        with open(out/(name+'_export.log'),'w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        rec={'id':row['id'],'status':'COMPLETE' if result.returncode==0 else 'FAILED',
             'returncode':result.returncode,'wall_seconds':time.time()-tick}
        if result.returncode==0:
            m=json.loads((dest/'COMPACT_MANIFEST.json').read_text())
            rec.update(compact_sha256=m['compact_sha256'],size_bytes=(dest/'compact.npz').stat().st_size,
                       full_B_reconstruction_bitwise_equal=m['full_B_reconstruction_bitwise_equal'],
                       all_compact_arrays_bitwise_verified=m['all_compact_arrays_bitwise_verified'])
        report['runs'].append(rec)
        (out/'COMPACT_GRID_EXECUTION.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(rec),flush=True)
    report.update(status='COMPLETE' if all(r['status']=='COMPLETE' for r in report['runs']) else 'INCOMPLETE',
                  total_wall_seconds=time.time()-started)
    (out/'COMPACT_GRID_EXECUTION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status']}),flush=True)

if __name__=='__main__': main()
