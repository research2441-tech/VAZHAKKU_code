"""Fig. 5: few-shot dialect identification of unseen speakers from K support clips (prototype and query sets from
disjoint speakers, mean over regions) for (a) generic representations and (b) dialect-code variants; (c) the speaker
code: dialect decodability against speaker EER."""
from config import ROOT
import json, sys, numpy as np, matplotlib.pyplot as plt
from figs_common import style, save, panel, GREYS, MARK, LS
style()
R = ROOT + '/work/res'
D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f'{R}/repr_final.json'))
REG = ['Northern', 'Southern', 'Western', 'Central']
def curve(fs): 
    Ks = sorted(int(k) for k in fs); return Ks, [np.mean([fs[str(k)][r] for r in REG]) * 100 for k in Ks]
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.55), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.42))
def draw(ax, series, jitter=0.0):
    for i, (lab, fs, c, l, m, f) in enumerate(series):
        Ks, v = curve(fs); off = (i - (len(series) - 1) / 2) * jitter
        ax.plot([k * (1 + off) for k in Ks], v, color=c, ls=l, marker=m, mfc=f, mec=c, ms=3.6, lw=0.9, label=lab)
    ax.axhline(25, color='black', lw=0.5, ls=':'); ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(['1', '2', '5', '10']); ax.set_xlabel('Support clips per region $K$')
    ax.set_ylim(15, 62); ax.minorticks_off()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1, borderaxespad=0.0, fontsize=6.0, handlelength=2.4)
names = {'full': 'code $\\mathbf{z}$: VAZHAKKU', 'nomi': 'code $\\mathbf{z}$: Hyper', 'grl': 'code $\\mathbf{z}$: Hyper+GRL', 'collapsed': 'code $\\mathbf{z}$: collapsed'}
sty = {'full': ('black', '-.', 'D', GREYS[3]), 'nomi': (GREYS[1], (0, (5, 1)), '^', 'white'), 'grl': (GREYS[1], (0, (1, 1)), 'P', GREYS[4]), 'collapsed': (GREYS[2], '--', 'X', GREYS[2])}
base = [('wav2vec 2.0 (layers 6–8)', D['baselines']['wav2vec2'], 'black', '-', 'o', 'black'),
        ('mel statistics', D['baselines']['mel statistics'], 'black', '--', 's', 'white'),
        ('ECAPA', D['baselines']['ECAPA'], GREYS[2], ':', 'v', 'white'),
        (names['full'], D['models']['full']['fewshot']) + sty['full']]
draw(axs[0], base); axs[0].set_ylabel('Region accuracy, unseen speakers (%)'); panel(axs[0], 'a')
codes = [(names[k], D['models'][k]['fewshot']) + sty[k] for k in ['full', 'nomi', 'grl', 'collapsed'] if k in D['models'] and 'fewshot' in D['models'][k]]
draw(axs[1], codes, jitter=0.04); panel(axs[1], 'b')
ax = axs[2]

pts = [('ECAPA input', D['ecapa_input'])] + [(lab, D['models'][k]) for k, lab in
       [('clone', 'Clone'), ('collapsed', 'Collapsed'), ('nomi', 'Hyper'), ('grl', 'Hyper+GRL'), ('full', 'VAZHAKKU')] if k in D['models']]
for i, (lab, d) in enumerate(pts):
    ax.plot(d['dialect_decodability'] * 100, d['speaker_eer'] * 100, marker=MARK[i], color='black', mfc=['black', 'white', GREYS[2], 'white', GREYS[4], GREYS[1]][i], ms=5, ls='none', label=lab)
ax.set_xlabel('Dialect decodability (bal. acc., %)'); ax.set_ylabel('Speaker EER of speaker code (%)')
ax.set_xlim(44, 60)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, borderaxespad=0.0, fontsize=6.0, handlelength=1.0, columnspacing=0.8)
panel(ax, 'c')
save(fig, 'fig5_representation'); print('ok', [p[0] for p in pts])
