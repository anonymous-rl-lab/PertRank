#!/usr/bin/env python3
"""Recompute the fixed sensitivity and independently check exact case masses."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json,math,subprocess,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(base,experiment,output,rebuild=False):
    reference=ROOT/'reference_sensitivity';expected=read(reference/'REFERENCE_SENSITIVITY_RESULTS.json')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise RuntimeError('Existing verification preserved')
    with tempfile.TemporaryDirectory(prefix='pertrank_reference_sensitivity_') as td:
        dest=Path(td)/'replayed'
        cmd=[sys.executable,str(ROOT/'reference_sensitivity.py'),'--base-root',str(base),'--experiment-root',str(experiment),'--output-root',str(dest)]
        if rebuild:cmd+=['--rebuild']
        subprocess.run(cmd,check=True,text=True,stdout=subprocess.DEVNULL)
        actual=read(dest/'REFERENCE_SENSITIVITY_RESULTS.json')
        assert {k:v for k,v in expected.items() if k!='protocol_sha256'}=={k:v for k,v in actual.items() if k!='protocol_sha256'}
        assert sha(dest/'REFERENCE_CASES.npz')==sha(reference/'REFERENCE_CASES.npz')
        with np.load(dest/'REFERENCE_CASES.npz',allow_pickle=False) as z:
            counts=z['case_counts'];weights=z['case_probability'];nominal=int(z['nominal_case_index'])
            rational=[]
            for pair in counts:
                p=Fraction(1)
                for a,b,c in pair:
                    p*=Fraction(math.factorial(3),math.factorial(int(a))*math.factorial(int(b))*math.factorial(int(c))*27)
                rational.append(p)
            assert sum(rational)==1 and len(set(tuple(c.flatten()) for c in counts))==100
            assert max(abs(float(p)-w) for p,w in zip(rational,weights))<1e-16
            checks=0;max_error=0.
            for source_index,scope in [(None,'aggregate')]+[(i,str(s)) for i,s in enumerate(z['source_names'])]:
                values=z['precision_by_case_method'] if source_index is None else z['precision_by_case_source_method'][:,source_index]
                for comp in ('original_PertRank','old_B'):
                    ci=z['method_names'].tolist().index(comp)
                    for mi,name in enumerate(z['method_names']):
                        delta=values[:,mi]-values[:,ci]
                        reported=actual['aggregate_comparisons'][comp][str(name)] if source_index is None else actual['source_comparisons'][comp][scope][str(name)]
                        positive=sum((p for p,d in zip(rational,delta) if d>1e-12),Fraction(0))
                        negative=sum((p for p,d in zip(rational,delta) if d < -1e-12),Fraction(0))
                        error=max(abs(float(positive)-reported['weighted_mass_positive']),abs(float(negative)-reported['weighted_mass_negative']))
                        max_error=max(max_error,error);assert error<1e-15;checks+=2
                        # Weighted sum computed independently through a Python scalar loop.
                        mean=sum(float(p)*float(d) for p,d in zip(rational,delta))
                        error=abs(mean-reported['weighted_mean_delta']);max_error=max(max_error,error);assert error<1e-15;checks+=1
                        order=sorted(range(100),key=lambda i:delta[i])
                        qs=[]
                        for threshold in (Fraction(1,40),Fraction(39,40)):
                            mass=Fraction(0)
                            for i in order:
                                mass+=rational[i]
                                if mass>=threshold:qs.append(float(delta[i]));break
                        assert qs==reported['weighted_empirical_2p5_97p5'];checks+=2
                        assert float(delta[nominal])==reported['nominal_delta'];checks+=1
    report=dict(passed=True,training_performed=False,model_selection_performed=False,
        all_sensitivity_results_reproduced=True,case_array_bytes_identical=True,
        exact_multinomial_mass_sum='1',exact_unique_cases=100,independent_statistic_checks=checks,
        independent_maximum_numeric_difference=max_error,
        original_results_sha256=sha(reference/'REFERENCE_SENSITIVITY_RESULTS.json'),
        case_array_sha256=sha(reference/'REFERENCE_CASES.npz'),temporary_replay_removed=True,
        note='Only fresh protocol timestamps and their hash differ; no change to original model scores, main results, or selection.')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-root',type=Path,required=True)
    p.add_argument('--experiment-root',type=Path,default=ROOT.parent)
    p.add_argument('--output',type=Path,default=ROOT/'REFERENCE_SENSITIVITY_CHECK.json')
    p.add_argument('--rebuild',action='store_true');a=p.parse_args();main(a.base_root,a.experiment_root,a.output,a.rebuild)
