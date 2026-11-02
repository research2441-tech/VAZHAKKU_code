"""Fig. 2: (a) intelligibility of the encoder prior and of flow sampling across backbone updates (pretrained front-end)
with the from-scratch encoder at matched budget (markers offset by 120 updates for legibility); (b) decoding configurations at the final backbone.
Values are the diagnostics logged in res/backbone_diagnostics.json (8 held-out SLR65 sentences)."""
from config import ROOT
import json, numpy as np, matplotlib.pyplot as plt
from figs_common import style, save, panel, GREYS
style()
D = json.load(open(ROOT + '/work/res/backbone_diagnostics.json'))
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.35), gridspec_kw=dict(wspace=0.42, width_ratios=[1.05, 1]))
ax = axs[0]; P = D['pretrained']; steps = [p['step'] for p in P]
ax.plot(steps, [p['prior'] * 100 for p in P], '-o', color='black', ms=3.5, label='prior, pretrained front-end')
ax.plot(steps, [p['flow'] * 100 for p in P], '--s', color='black', mfc='white', ms=3.5, label='flow ($\\tau$=0.667, $N$=10), pretrained')
S = D['scratch']
ax.plot([S['step'] - 120], [S['prior'] * 100], '^', color=GREYS[2], mec='black', ms=5, label='prior, from-scratch encoder')
ax.plot([S['step'] + 120], [S['flow'] * 100], 'v', color='white', mec='black', ms=5, label='flow, from-scratch encoder')
ax.axhline(D['natural_cer'] * 100, color='black', lw=0.6, ls=':')
ax.text(250, D['natural_cer'] * 100 + 2.5, 'vocoded natural speech', fontsize=6.0)
ax.axvline(3500, color=GREYS[3], lw=0.6, ls='-.'); ax.text(3580, 14, 'source centred\non prior from here', fontsize=5.8, va='center', bbox=dict(fc='white', ec='none', pad=0.2))
ax.set_xlabel('backbone updates'); ax.set_ylabel('CER (%)'); ax.set_ylim(0, 92); ax.set_xlim(0, 7400)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=5.9, columnspacing=0.8, handletextpad=0.4)
panel(ax, 'a')
ax = axs[1]; C = D['decoding']
lab = [c['label'] for c in C]; y = np.arange(len(C))
ax.barh(y, [c['cer'] * 100 for c in C], 0.62, color=[GREYS[1] if c['label'].startswith('prior') else GREYS[4] for c in C], edgecolor='black', lw=0.5)
for yy, c in zip(y, C):
    ax.text(c['cer'] * 100 + 1.2, yy, f"{c['cer'] * 100:.1f} | {c['mcd']:.1f}", va='center', fontsize=5.9)
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.0); ax.tick_params(axis='y', pad=7); ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel('CER (%)  [label: CER | MCD dB]'); ax.grid(axis='y', visible=False)
panel(ax, 'b')
save(fig, 'fig2_backbone'); print('ok')
