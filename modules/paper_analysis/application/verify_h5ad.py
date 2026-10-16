#!/usr/bin/env python3
"""Dense, CSR, backed H5AD and AnnData parity including full-axis totals."""
import json, resource, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from make_example import fixture
from run_counts import fit_counts
from run_h5ad import fit_anndata

def main():
    root=Path(__file__).resolve().parent;data=fixture()
    # A measured gene absent from controls but expressed in target cells must
    # remain in full-library denominators before weight-axis filtering.
    extra=np.where(data['is_control'],0,500).astype(data['X'].dtype)
    data['X']=np.column_stack([data['X'],extra]);data['gene_ids']=np.concatenate([data['gene_ids'],['CONTROL_ABSENT']])
    order=np.random.default_rng(12026).permutation(len(data['X']))
    for k in ('X','cell_ids','target_ids','target_gene_ids','batch_ids','is_control'):data[k]=data[k][order]
    dense,md=fit_counts(data)
    obs=pd.DataFrame({k:data[k] for k in ('target_ids','target_gene_ids','batch_ids','is_control')},index=data['cell_ids'])
    errors={};timings={}
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);qpath=tmp/'queries.txt';qpath.write_text('\n'.join(data['query_ids'])+'\n')
        for storage in ('dense','csr'):
            X=data['X'] if storage=='dense' else sparse.csr_matrix(data['X'])
            a=ad.AnnData(X=X,obs=obs,var=pd.DataFrame(index=data['gene_ids']))
            for mode in ('memory','backed'):
                path=tmp/f'{storage}.h5ad'
                if mode=='backed':a.write_h5ad(path);use=ad.read_h5ad(path,backed='r')
                else:use=a
                start=time.perf_counter();result,meta=fit_anndata(use,data['query_ids'],batch_rows=137)
                timings[storage+'_'+mode]=time.perf_counter()-start
                vals=[float(np.max(abs(result[k]-dense[k]))) for k in ('effects','energy','noise_trace','direction_energy','direction_noise')]
                vals.extend(float(np.max(abs(result['weights'][arm]-dense['weights'][arm]))) for arm in dense['weights'])
                errors[storage+'_'+mode]=max(vals)
                assert errors[storage+'_'+mode]<1e-10,errors
                assert meta['normalization_scales']==md['normalization_scales']
                assert len(result['gene_ids'])==320 and meta['full_gene_count']==321
                if mode=='backed':use.file.close()
            output=tmp/(storage+'_out')
            subprocess.run([sys.executable,str(root/'run_h5ad.py'),str(path),'--queries',str(qpath),'--out',str(output),'--top-k','3','--batch-rows','137'],check=True,capture_output=True,text=True)
            import csv
            rows=list(csv.DictReader((output/'all_candidate_scores.csv').open()))
            for arm in dense['weights']:
                for query in data['query_ids']:
                    for direction in ('up','down'):
                        part=[r for r in rows if r['method']==arm and r['query_id']==query and r['direction']==direction]
                        assert abs(sum(float(r['tie_inclusion_probability']) for r in part)-3)<1e-12
                        assert all(r['target_gene_id']!=query for r in part)
    report=dict(status='PASS',version='v20.0',fixture='Synthetic raw counts, 2000 controls, 8 targets x20 cells, 321 full genes including a target-only count gene',
        max_absolute_error=errors,elapsed_seconds=timings,peak_process_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        full_axis_normalization=True,shuffled_rows=True,dense_and_CSR=True,backed_and_in_memory=True,CLI_both_formats=True,
        scope='Software correctness and example cost; no claim of original raw cohort timing',anndata_version=ad.__version__)
    (root/'VALIDATION_H5AD_v20.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__': main()
