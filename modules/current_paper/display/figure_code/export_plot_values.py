"""Write exact figure values with explicit uncertainty semantics."""
from pathlib import Path
import csv

def export_plot_values(V, outdir):
    rows=[]
    def add(panel, item, metric, value, interval=None, level=None, kind='point_estimate', source=''):
        rows.append({'panel':panel,'item':item,'metric':metric,'estimate':value,
                     'lower':interval[0] if interval else '', 'upper':interval[1] if interval else '',
                     'interval_level':level if level else '', 'uncertainty_kind':kind,'source':source})
    rv={r['task']:r for r in V['rvalues']}
    rs='source_data/rvalues/primary_contrasts.csv'
    add('2b','rvalues / equal-state D4','P25',sum(float(rv[t]['rvalues']) for t in ['D4 Rest','D4 Stim'])/2,source=rs)
    for source in ['D3','D1','D2']:
        r=rv['External RNA '+source]
        add('3a',source+' / rvalues','P10',float(r['rvalues']),source=rs)
        add('3b',source+' / PertRank minus rvalues','delta_P10',float(r['difference']),
            [float(r['ci95_lower']),float(r['ci95_upper'])],.95,'conditional_query_block_interval',rs)
    P=V['program_external'];es='source_data/programs/external_summary.json'
    for arm,q in P['method_results'].items():
        add('5b',arm,'utility',q['estimate'],source=es)
    for group in ['overall','by_program','by_cell_line']:
        obj={'overall':P['application'][group]} if group=='overall' else P['application'][group]
        for k,q in obj.items():
            add('5c','application / '+k,'delta_utility',q['estimate'],q['conditional_interval'],
                q['conditional_interval_level'],'conditional_reference_interval',es)
    ds='source_data/programs/d3_mechanism_summary.json'
    for k,q in V['program_D3'].items():
        add('5d','D3 / '+k,'delta_utility',q['delta_utility'],q['conditional_interval'],
            q['interval_level'],'conditional_reference_interval',ds)
    for k,q in P['geometry'].items():
        add('5d','sci-Plex3 / '+k,'delta_utility',q['estimate'],q['conditional_interval'],
            q['conditional_interval_level'],'conditional_reference_interval',es)
    for k,q in P['replacement'].items():
        for field,metric in [('mean_replacement_mass','replaced_positions'),
                             ('mass_weighted_entering_reference_mean','entering_reference_response'),
                             ('mass_weighted_leaving_reference_mean','leaving_reference_response'),
                             ('mean_utility_difference','delta_utility')]:
            add('5e',k,metric,q[field],source=es)
    with (Path(outdir)/'plotted_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return len(rows)
