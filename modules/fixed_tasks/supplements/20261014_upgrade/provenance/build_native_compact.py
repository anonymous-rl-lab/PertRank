from pathlib import Path
import hashlib,json,shutil,ast,zipfile
import numpy as np

BASE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/scratch/9bb77e8faf2b/package_v130/extracted/PertRank_A19R_Native_Baselines_20261014')
OUT=BASE/'native_rna'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(p,**z):p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**z)
def cp(s,d):d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,d)
def ix(have,want):
    m={str(x):i for i,x in enumerate(have)};return np.array([m[str(x)] for x in want])
def membership(y,own):
    s=np.concatenate([y,-y]);mask=np.concatenate([own,own]);s[mask]=-np.inf
    threshold=np.partition(s,s.shape[1]-10,axis=1)[:,-10,None]
    above,tie=s>threshold,s==threshold
    return above.astype(float)+tie*((10-above.sum(1))/tie.sum(1))[:,None]
sources={};cert={};membership_entries=0
for donor in ['D3_Rest','D1_Rest','D2_Rest']:
    z=load(SOURCE/'legacy'/donor/'predictions_locked.npz');sources[donor]=z
    keep=[list(z['arm_names']).index(x) for x in ['raw','w0y','W1']]
    for key in ['effects','top10_probability','arm_names']:z[key]=z[key][keep]
    save(OUT/'legacy'/donor/'predictions_locked.npz',**z)
    for method,filename,field,axis in [('FR','effects.npz','effect','targets'),('ASH','posterior.npz','posterior','row_ids')]:
        src=SOURCE/'models'/donor/method/'native'/filename;m=load(src)
        qi=ix(m['genes'],z['query_ids']);ti=ix(m[axis],z['target_ids'])
        projection=m[field][np.ix_(ti,qi)]
        dest=OUT/'models'/donor/method/'native'/filename
        save(dest,**{field:projection,'genes':z['query_ids'],axis:z['target_ids']})
        cp(SOURCE/'models'/donor/method/'OWNER_COMPLETION.json',OUT/'provenance'/donor/(method+'_ORIGINAL_OWNER_COMPLETION.json'))
        record={'status':'LOSSLESS_QUERY_PROJECTION_OF_ARCHIVED_MODEL','donor':donor,'method':method,'original_full_model_sha256':sha(src),'projected_sha256':sha(dest),'candidates_retained':len(ti),'queries_retained':len(qi),'all_original_candidates_retained':True,'float_dtype':str(projection.dtype),'quantization':False}
        dest.parent.parent.joinpath('PROJECTION.json').write_text(json.dumps(record,indent=2)+'\n')
        cert[donor+'/'+method]=record
    old=load(SOURCE/'original_scores'/donor/'scores.npz')
    golden={k:v for k,v in old.items() if k not in ['predictions','reference_effect','own_mask','query_ids','target_ids','target_symbols','query_blocks']}
    save(OUT/'original_scores'/donor/'scores.npz',**golden)
    for fn in ['report.json','reference_propagation_shared_labels.npz','reference_propagation_batch_separate.npz']:
        cp(SOURCE/'original_scores'/donor/fn,OUT/'original_scores'/donor/fn)
for p in (SOURCE/'original_scores/equal_source').iterdir():cp(p,OUT/'original_scores/equal_source'/p.name)
cp(SOURCE/'legacy/point_results.json',OUT/'legacy/point_results.json')
ref=load(SOURCE/'reference/query_reference.npz')
save(OUT/'reference/query_reference.npz',**{k:ref[k] for k in ['query_ids','target_symbols','mean_effect']})
for scheme in ['shared_labels','batch_separate']:
    src=SOURCE/'reference'/('reference_draws_'+scheme+'_float64.npz');d=load(src)
    for donor,z in sources.items():
        qi=ix(ref['query_ids'],z['query_ids']);ti=ix(ref['target_symbols'],z['target_symbols'])
        powers=np.left_shift(np.uint64(1),np.arange(len(ti),dtype=np.uint64))
        full=np.zeros((500,2*len(qi)),dtype=np.uint64);partial=np.zeros_like(full)
        for i,e in enumerate(d['effect_draws']):
            p=membership(e[np.ix_(qi,ti)],z['own_mask'])
            full[i]=np.sum((p==1)*powers,axis=1,dtype=np.uint64)
            partial[i]=np.sum(((p>0)&(p<1))*powers,axis=1,dtype=np.uint64)
            a=(full[i,:,None]&powers)!=0;b=(partial[i,:,None]&powers)!=0
            count=b.sum(1);frac=np.divide(10-a.sum(1),count,out=np.zeros(len(a),dtype=float),where=count>0)
            rebuilt=a.astype(float)+b*frac[:,None]
            if not np.array_equal(rebuilt,p):raise AssertionError('Reference membership compaction is not exact')
            membership_entries+=p.size
        # Same-query adjacent draws; XOR is exactly reversible and saves repeated bits.
        full[1:]=np.bitwise_xor(full[1:],full[:-1]);partial[1:]=np.bitwise_xor(partial[1:],partial[:-1])
        save(OUT/'reference'/donor/('draws_'+scheme+'.npz'),definite_xor=full.T,fractional_xor=partial.T,
             query_ids=z['query_ids'],target_ids=z['target_ids'],source_sha256=np.array(sha(src)))
        print(json.dumps({'packed_reference':donor+'/'+scheme,'membership_exact':True}),flush=True)
    del d

code=OUT/'code';code.mkdir(exist_ok=True)
original=(SOURCE/'code/score_native.py').read_text()
cp(SOURCE/'code/score_native.py',OUT/'provenance/score_native_original.py')
old='''        root=Path(cfg[method]); owner=root/'OWNER_COMPLETION.json'
        ov=json.loads(owner.read_text())
        assert ov['status']=='VERIFIED_AFTER_OWNER_EXIT' and ov['method']==method and ov['donor']==cfg['donor']
        name='effects.npz' if method=='FR' else 'posterior.npz'
        model=root/'native'/name
        assert sha(model)==ov['archives'][name]['sha256']
        provenance[str(model)]=sha(model);provenance[str(owner)]=sha(owner)
'''
new='''        root=Path(cfg[method]); owner=root/'PROJECTION.json'
        ov=json.loads(owner.read_text())
        assert ov['status']=='LOSSLESS_QUERY_PROJECTION_OF_ARCHIVED_MODEL' and ov['method']==method and ov['donor']==cfg['donor']
        name='effects.npz' if method=='FR' else 'posterior.npz'
        model=root/'native'/name
        assert sha(model)==ov['projected_sha256']
        provenance[str(model)]=sha(model);provenance[str(owner)]=sha(owner)
'''
assert old in original;adapted=original.replace(old,new,1)
old='''        zz=arr(path);assert zz['effect_draws'].shape==(500,len(ref['query_ids']),len(ref['target_symbols']))
        assert zz['effect_draws'].dtype==np.float64
        assert np.array_equal(zz['query_ids'],ref['query_ids']) and np.array_equal(zz['target_symbols'],ref['target_symbols'])
        provenance[path]=sha(path)
        rv=np.empty((500,len(names)))
        for i,e in enumerate(zz['effect_draws']):
            tp=membership(e[np.ix_(qi,ti)],own)
'''
new='''        zz=arr(path)
        assert np.array_equal(zz['query_ids'],q) and np.array_equal(zz['target_ids'],t)
        full=np.bitwise_xor.accumulate(zz['definite_xor'].T,axis=0)
        partial=np.bitwise_xor.accumulate(zz['fractional_xor'].T,axis=0)
        assert full.shape==partial.shape==(500,2*len(q))
        powers=np.left_shift(np.uint64(1),np.arange(len(t),dtype=np.uint64))
        provenance[path]=sha(path)
        rv=np.empty((500,len(names)))
        for i in range(500):
            aa=(full[i,:,None]&powers)!=0;bb=(partial[i,:,None]&powers)!=0
            counts=bb.sum(1)
            fraction=np.divide(K-aa.sum(1),counts,out=np.zeros(len(aa),dtype=float),where=counts>0)
            tp=aa.astype(float)+bb*fraction[:,None]
            assert np.allclose(tp.sum(1),K,rtol=0,atol=1e-12)
'''
assert old in adapted;adapted=adapted.replace(old,new,1)
(code/'score_native.py').write_text(adapted)
cp(SOURCE/'code/combine_sources.py',code/'combine_sources.py')
replay=(SOURCE/'code/reproduce_scores.py').read_text()
old="'reference_draws':{s:str(root/'reference'/('reference_draws_'+s+'_float64.npz')) for s in ['shared_labels','batch_separate']}"
new="'reference_draws':{s:str(root/'reference'/d/('draws_'+s+'.npz')) for s in ['shared_labels','batch_separate']}"
assert old in replay;replay=replay.replace(old,new,1)
# Extend archived-output checks to both reference-draw propagation arrays.
needle="        checks[d]=errors;old.close();new.close()"
replacement="""        checks[d]=errors;old.close();new.close()
        for scheme in ['shared_labels','batch_separate']:
            name='reference_propagation_'+scheme+'.npz'
            with np.load(root/'original_scores'/d/name,allow_pickle=False) as before, np.load(out/d/name,allow_pickle=False) as after:
                assert np.array_equal(before['arm_names'],after['arm_names'])
                np.testing.assert_allclose(before['expected_members'],after['expected_members'],rtol=0,atol=1e-12)
"""
assert needle in replay;replay=replay.replace(needle,replacement,1)
(code/'reproduce_scores.py').write_text(replay)
for p in code.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
audit={'status':'EXACT_FIXED_TASK_DATA_COMPACTION','source_bundle':'PertRank_A19R_Native_Baselines_20261014.zip','model_projections':cert,
 'reference_membership_entries_compared':membership_entries,'reference_membership_mismatches':0,'full_candidate_pools_preserved':True,
 'float_prediction_quantization':False,'reference_encoding':'Lossless top-10 marginal membership with exact fractional ties, bit masks and reversible XOR; raw RNA magnitudes of each draw not reconstructed.',
 'scientific_scores_recomputed_during_packaging':False,'model_refits':0,'scope':'Fixed K10, fixed candidates/queries/methods. Preserves all reported scoring and reference sensitivity endpoints; not a general-purpose raw expression archive.'}
(OUT/'COMPACTION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
(OUT/'README.txt').write_text('Fixed-task, lossless numerical reproduction of the completed independent-RNA comparison.\nAll 52/59/63 candidates and 293/289/289 queries retained. Original float64 predictions and weights are unquantized.\nReference resampling preserves exact top-10 inclusion probabilities, including fractional ties, for all 500 draws in each of both schemes. This is sufficient for the archived fixed-ranking experiment, not for new methods/K/candidate sets.\nRun: python code/reproduce_scores.py --bundle . --out NEW_EXTERNAL_OUTPUT\nModel fitting and scientific endpoint recomputation were not performed during packaging; exact data representation checks were performed.\nFull raw counts and full-gene models remain in their separately archived original packages.\n')
with zipfile.ZipFile(BASE/'native_size_check.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
print(json.dumps({'native_compact_zip_bytes':(BASE/'native_size_check.zip').stat().st_size,'membership_entries_exact':membership_entries}),flush=True)
