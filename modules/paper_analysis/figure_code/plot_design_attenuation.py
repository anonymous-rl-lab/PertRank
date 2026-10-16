"""Reproduce Supplementary Fig. 5 and its descriptive table derivation.

Run: python figure_code/plot_design_attenuation.py
Inputs are the 14 displayed SN3.1 table rows, not raw expression or new model
outputs. No resampling, regression, hypothesis test or model fitting is run.
Requires Python, numpy and matplotlib. Outputs: CSV-derived JSON plus PNG/PDF/SVG.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source_data' / 'si' / 'v24_design_attenuation.csv'
OUT = ROOT / 'manuscript' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def average_ranks(values):
    """1-based average ranks, including exact ties, without a scipy dependency."""
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def spearman(x, y):
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


rows = list(csv.DictReader(DATA.open()))
assert len(rows) == 14
energy = np.array([float(r['query_row_energy_attenuation_percent']) for r in rows])
pr = np.array([float(r['pertrank_score']) for r in rows])
loss = np.array([float(r['fixed_weight_coordinate_cost']) for r in rows])
relative = loss / pr
assert np.allclose(relative, [float(r['relative_cost_fraction']) for r in rows], atol=1e-15)
rhos = (spearman(energy, loss), spearman(energy, relative))
assert np.allclose(rhos, [0.9472527472527472, 0.621978021978022], atol=1e-12)
summary = {
    'source': 'Supplementary Information v24, SN3.1; displayed table precision',
    'identity': 'Post hoc descriptive derivation from existing results; no new experiment',
    'csv_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
    'settings': len(rows),
    'experimental_sources': ['T cells (D3 and D1)', 'KOLF iPSCs', 'Replogle K562', 'Orion HCT116'],
    'absolute_cost_definition': 'P@K(w_M y) minus P@K(w_M A y)',
    'relative_cost_definition': 'absolute cost divided by P@K(w_M y)',
    'spearman_absolute_cost': rhos[0],
    'spearman_relative_cost': rhos[1],
    'uncertainty': 'No p value or interval. Settings are correlated across sources, panels and depths.',
    'd3_responsive_20_example': {
        'precision': 'derived from displayed scores and contrasts in SN3.1 and SN3.2',
        'w0_y': 0.2758, 'wM_y': 0.3114, 'w0_Ay': 0.2553, 'wM_Ay': 0.2853,
        'geometry_at_y': 0.0356, 'geometry_at_Ay': 0.0300,
        'readout_at_w0': -0.0205, 'readout_at_wM': -0.0261,
        'difference_in_differences': -0.0056,
        'uncertainty': 'Point contrast only; no interaction interval or test.'
    },
}
(DATA.with_suffix('.json')).write_text(json.dumps(summary, indent=2) + '\n')

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
    'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none',
})
colours = {'T cells': '#D97918', 'KOLF': '#8464A3', 'Replogle': '#2477AD', 'Orion': '#248A78'}
labels = {'T cells': 'T cells (D3, D1)', 'KOLF': 'KOLF iPSCs', 'Replogle': 'Replogle K562', 'Orion': 'Orion HCT116'}
fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.6))
for ax, y, label, letter, title, rho in zip(
        axes, (loss, 100 * relative),
        ('Fixed-weight readout cost (P@K)', 'Relative readout cost (%)'),
        ('a', 'b'), ('Absolute cost', 'Relative cost'), rhos):
    for source, colour in colours.items():
        mask = np.array([r['source'] == source for r in rows])
        ax.scatter(energy[mask], y[mask], c=colour, s=33, edgecolors='white', linewidths=.6, zorder=3)
    ax.set_xlabel('Query-row energy attenuation (%)')
    ax.set_ylabel(label)
    ax.set_title(f'{letter}  {title}', loc='left', fontweight='bold', pad=11)
    ax.text(.04, .94, rf'Spearman $\rho$ = {rho:.3f}', transform=ax.transAxes, va='top', fontsize=9)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(bottom=-.001 if letter == 'a' else -.7, top=.045 if letter == 'a' else 18)
    ax.grid(color='#E8E8E8', linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
handles = [Line2D([0], [0], marker='o', lw=0, color=c, markersize=5, label=labels[s]) for s,c in colours.items()]
fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.52, .005))
fig.subplots_adjust(left=.095, right=.985, bottom=.30, top=.86, wspace=.40)
for suffix in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'figS5_design_attenuation_v25.{suffix}', dpi=300, facecolor='white')
plt.close(fig)
print(json.dumps({'settings': len(rows), 'spearman_absolute': rhos[0], 'spearman_relative': rhos[1]}))
