"""Extract exact plotting values from the publication Source Data.

No fits, scores, resampling or new intervals are computed. Run before rendering:
    python figure_code/prepare_plot_inputs.py
"""
from pathlib import Path
import csv, hashlib, json, math

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FILES = {
    'rvalues': ROOT/'source_data/rvalues/primary_contrasts.csv',
    'program_external': ROOT/'source_data/programs/external_summary.json',
    'program_D3': ROOT/'source_data/programs/d3_mechanism_summary.json',
}
TASKS = {
    'D4_Rest': 'D4 Rest', 'D4_Stim8hr': 'D4 Stim',
    'External_D3_Rest': 'External RNA D3', 'External_D1_Rest': 'External RNA D1',
    'External_D2_Rest': 'External RNA D2',
}
rows=[]
for r in csv.DictReader(FILES['rvalues'].open()):
    rows.append({
        'task':TASKS[r['task']], 'K':r['K'], 'PertRank':r['PertRank'], 'rvalues':r['rvalues'],
        'difference':r['difference'], 'ci95_lower':r['conditional_95_low'],
        'ci95_upper':r['conditional_95_high'], 'expected_member_difference':r['reference_members_difference'],
    })
rows.sort(key=lambda r: list(TASKS.values()).index(r['task']))
e=json.loads(FILES['program_external'].read_text())
d=json.loads(FILES['program_D3'].read_text())
app='application_raw_PertRank_minus_native_ASH'
geom=['geometry_on_raw_readout','geometry_on_ASH_readout']
V={
    'provenance':{k:{'filename':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in FILES.items()},
    'rvalues':rows,
    'program_external':{
        'method_results':{k:v['utility']['overall'] for k,v in e['method_results'].items()},
        'application':e['contrasts'][app]['utility'],
        'geometry':{k:e['contrasts'][k]['utility']['overall'] for k in geom},
        'replacement':{k:e['replacement'][k] for k in [app,geom[1]]},
        'programs':e['programs'], 'cell_lines':e['cell_lines'], 'candidate_counts':e['candidate_counts'],
    },
    'program_D3':{k:{x:d['contrasts'][k][x] for x in ['delta_utility','interval_level','conditional_interval']} for k in ['original_geometry','ASH_geometry']},
}

def compare(old, new, path=''):
    """Check existing plotting payload against the normalized export, ignoring added metadata."""
    global checks,max_abs
    if path=='provenance' or path.endswith('uncertainty_metadata'): return
    if isinstance(old,dict):
        for k,v in old.items():
            if k not in new: raise AssertionError(f'Missing {path}/{k}')
            compare(v,new[k],f'{path}/{k}' if path else k)
    elif isinstance(old,list):
        assert len(old)==len(new),path
        for i,(a,b) in enumerate(zip(old,new)):compare(a,b,f'{path}/{i}')
    else:
        try:
            a,b=float(old),float(new)
        except (ValueError,TypeError):
            assert old==new,(path,old,new)
        else:
            assert math.isclose(a,b,abs_tol=1e-12,rel_tol=0),(path,a,b)
            checks+=1;max_abs=max(max_abs,abs(a-b))
checks=0;max_abs=0
out=HERE/'program_ranking_values.json'
if out.exists():compare(json.loads(out.read_text()),V)
out.write_text(json.dumps(V,indent=2)+'\n')
report={'scope':'Source Data extraction and previous plot-payload agreement; no reanalysis',
        'numeric_values_checked':checks,'maximum_absolute_difference':max_abs,'tolerance':1e-12,
        'source_files':V['provenance'], 'P10_ranges_used_in_main_figure_5':False}
(HERE/'PLOT_INPUT_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['numeric_values_checked','maximum_absolute_difference','P10_ranges_used_in_main_figure_5']}))
