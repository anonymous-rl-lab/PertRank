"""Preserve values while naming P@10 reference-resampling ranges accurately.
Adapted without changing numerical logic from the v27.4 export_results.py.
"""
def require(ok,path):
    if not ok:raise RuntimeError('Export changed source values at '+repr(path))

def normalize(obj,path=(),changes=None):
    changes=[] if changes is None else changes
    if isinstance(obj,list):return [normalize(x,path+(str(i),),changes) for i,x in enumerate(obj)]
    if not isinstance(obj,dict):return obj
    p10=any(k in ('P10','P@10') for k in path)
    mapping={'conditional_interval':'reference_resampling_range',
             'conditional_interval_level':'reference_resampling_range_level'} if p10 else {}
    out={}
    for k,v in obj.items():
        nk=mapping.get(k,k)
        if k!=nk:changes.append({'old':'/'.join(path+(k,)),'new':'/'.join(path+(nk,))})
        out[nk]=normalize(v,path+(k,),changes)
    if 'conditional_interval' in obj:
        out['uncertainty_metadata']={
            'kind':'reference_resampling_stability_range' if p10 else 'conditional_reference_interval',
            'resampling_unit':'reference target cells within culture wells and vehicle cells within wells; shared plate controls resampled jointly',
            'conditioning':'fixed predictions, shallow inputs, candidate pool, programs and observed culture conditions',
            'interpretation':'quantile range of re-ranked reference-list overlap; not a calibrated confidence interval' if p10 else 'reference measurement uncertainty of the fixed-list response effect'
        }
    return out

def assert_preserved(old,new,path=(),counts=None):
    counts={} if counts is None else counts
    p10=any(k in ('P10','P@10') for k in path)
    mapping={'conditional_interval':'reference_resampling_range','conditional_interval_level':'reference_resampling_range_level'} if p10 else {}
    if isinstance(old,dict):
        for k,v in old.items():assert_preserved(v,new[mapping.get(k,k)],path+(k,),counts)
        require(set(new)=={mapping.get(k,k) for k in old}|({'uncertainty_metadata'} if 'conditional_interval' in old else set()),path)
    elif isinstance(old,list):
        require(len(old)==len(new),path)
        for i,(a,b) in enumerate(zip(old,new)):assert_preserved(a,b,path+(str(i),),counts)
    else:
        require(old==new,path)
        if isinstance(old,(float,int)) and not isinstance(old,bool):counts['numeric_values_unchanged']=counts.get('numeric_values_unchanged',0)+1
    return counts

