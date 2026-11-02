"""Fig. 4: synthesis-level dialect control. (a) distribution of the instrument's decisions; (b) metrics with anchors."""
from config import ROOT
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from figs_common import style, save, panel, GREYS, HATCH
from fig_synth import boot_ci
style()
R = ROOT + '/work/res'; REG = ['Northern', 'Southern', 'Western', 'Central']
S = json.load(open(f'{R}/synth_stats.json'))
A = pd.read_csv(f'{R}/eval_anchors_anchors.csv'); F = pd.read_csv(f'{R}/eval_seen_full_s0.csv'); C = pd.read_csv(f'{R}/eval_seen_clone_s0.csv')
rows = [('Natural test\nutterances', S['pred_dist']['natural_test']),
        ('Clone', S['pred_dist']['clone']), ('VAZHAKKU', S['pred_dist']['full'])]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.5), gridspec_kw=dict(width_ratios=[1.0, 1.25], wspace=0.38))
ax = axs[0]; y = np.arange(len(rows)); left = np.zeros(len(rows))
fills = ['black', GREYS[2], 'white', GREYS[4]]; hatches = ['', '', '///', '...']
for k, r in enumerate(REG):
    v = np.array([d.get(r, 0.0) for _, d in rows]) * 100
    ax.barh(y, v, 0.55, left=left, color=fills[k], hatch=hatches[k], edgecolor='black', lw=0.5, label=r)
    for yy, l, vv in zip(y, left, v):
        if vv >= 7: ax.text(l + vv / 2, yy, f'{vv:.0f}', ha='center', va='center', fontsize=6.0, color='black', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
    left += v
ax.set_yticks(y); ax.set_yticklabels([n for n, _ in rows]); ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel("Instrument's decision (% of clips)"); ax.grid(axis='y', visible=False)
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.30), ncol=4, borderaxespad=0.0, handlelength=1.2, columnspacing=0.8)
panel(ax, 'a')
ax = axs[1]
systems = [('Natural', A[A.system == 'natural']), ('Vocoded', A[A.system == 'resynth']),
           ('Clone', C), ('VAZHAKKU', F)]
metrics = [('DCA', 'correct'), ('SECS', 'secs'), ('1 − CER', 'cer')]
w = 0.19; x = np.arange(len(metrics))
for k, (name, d) in enumerate(systems):
    vals, lo, hi = [], [], []
    for _, col in metrics:
        dd = d.copy()
        if col == 'cer': dd['v'] = 1 - dd.cer.clip(upper=1.0)
        else: dd['v'] = dd[col]
        m = dd.v.mean(); l, h = boot_ci(dd, 'v'); vals.append(m); lo.append(m - l); hi.append(h - m)
    ax.bar(x + (k - 1.5) * w, vals, w, yerr=[lo, hi], color=['black', GREYS[2], 'white', GREYS[4]][k], hatch=['', '', '///', '...'][k],
           edgecolor='black', lw=0.5, error_kw=dict(lw=0.6, capsize=1.5), label=name)
ax.axhline(0.25, color='black', lw=0.6, ls=':')
import matplotlib.transforms as mt
ax.text(1.01, 0.25, 'chance\nDCA', fontsize=5.6, va='center', ha='left', linespacing=0.9, transform=mt.blended_transform_factory(ax.transAxes, ax.transData))
ax.set_xticks(x); ax.set_xticklabels([m for m, _ in metrics]); ax.set_ylim(0, 1.05); ax.set_ylabel('Score (higher is better)')
ax.grid(axis='x', visible=False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=4, borderaxespad=0.0, handlelength=1.2, columnspacing=0.8)
panel(ax, 'b')
save(fig, 'fig4_synthesis')
print('ok')
