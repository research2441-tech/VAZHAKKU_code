"""Fig. 2: register bridge. (a) WER by system (mean, s.d. over seeds); (b) identity-bias sweep; (c) learning curve."""
from config import ROOT
import json, numpy as np, matplotlib.pyplot as plt
from figs_common import style, save, panel, GREYS
style()
R = ROOT + '/work/res'
B = json.load(open(f'{R}/bridge_final.json'))
order = [('Copy', 'Copy (standard text)'), ('Seq2seq', 'Char. transformer'), ('Lattice-noctx', 'Lattice, no context'),
         ('Lattice-none', 'Lattice, no region'), ('Lattice-post', 'Lattice, posterior'), ('Lattice-hard', 'Lattice, hard district'),
         ('Lattice-hard+d', 'Lattice, hard district, $\\delta$'), ('Lattice-post+d', 'Lattice, posterior, $\\delta$')]
order = [(k, l) for k, l in order if k in B]
fig = plt.figure(figsize=(7.1, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 0.85], wspace=0.55)
ax = fig.add_subplot(gs[0])
y = np.arange(len(order)); m = [B[k]['WER_mean'] * 100 for k, _ in order]; s = [B[k]['WER_sd'] * 100 for k, _ in order]
cols = ['white' if k == 'Copy' else (GREYS[1] if k == 'Seq2seq' else (GREYS[3] if '+d' not in k else 'black')) for k, _ in order]
hat = ['///' if k == 'Copy' else '' for k, _ in order]
ax.barh(y, m, 0.62, xerr=s, color=cols, hatch=hat, edgecolor='black', lw=0.5, error_kw=dict(lw=0.6, capsize=1.5))
for yy, mm, ss in zip(y, m, s):
    ax.text(mm + ss + 1.5, yy, f'{mm:.1f}', va='center', fontsize=6.0, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'), zorder=5)
ax.axvline(B['Copy']['WER_mean'] * 100, color='black', lw=0.5, ls=':')
ax.set_yticks(y); ax.set_yticklabels([l for _, l in order], fontsize=6.3); ax.invert_yaxis()
ax.set_xlabel('WER against spoken reference (%)'); ax.set_xlim(0, 92); ax.grid(axis='y', visible=False); panel(ax, 'a')
ax = fig.add_subplot(gs[1])
src = B.get('Lattice-post+d') or B.get('Lattice-hard+d')
if src and src.get('margin_curve'):
    mc = src['margin_curve']; dl = [r['delta'] for r in mc]
    ax.plot(dl, [r['val_WER'] * 100 for r in mc], marker='o', color='black', mfc='white', ms=3.5, label='validation WER')
    ax.plot(dl, [r['test_WER'] * 100 for r in mc], marker='s', color='black', ls='--', mfc='black', ms=3.2, label='test WER')
    ax.plot(dl, [r['test_changed'] * 100 for r in mc], marker='^', color=GREYS[2], ls=':', mfc=GREYS[3], ms=3.5, label='changed words (test)')
    ax.axhline(B['reference_change_rate'] * 100, color=GREYS[2], lw=0.5, ls='-.')
    ax.text(dl[-1], B['reference_change_rate'] * 100 + 0.8, 'reference\nchange rate', ha='right', va='bottom', fontsize=5.6, linespacing=0.9)
    best = min(mc, key=lambda r: r['val_WER'])
    ax.axvline(best['delta'], color='black', lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.text(best['delta'] + 0.06, 30.0, 'selected\non validation', fontsize=5.6, ha='left', va='bottom', linespacing=0.9, bbox=dict(fc='white', ec='none', pad=0.3))
    ax.set_xlabel('Identity bias $\\delta$'); ax.set_ylabel('Percent of words')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1, borderaxespad=0.0, fontsize=6.0)
panel(ax, 'b')
ax = fig.add_subplot(gs[2])
S = B.get('sensitivity', {}); pts = [(f, S[k]['WER'] * 100) for f, k in [(25, 'frac25'), (50, 'frac50')] if k in S]
pts.append((100, B['Lattice-post']['WER'][0] * 100))
ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='D', color='black', mfc=GREYS[3], ms=3.8)
ax.axhline(B['Copy']['WER_mean'] * 100, color='black', lw=0.5, ls=':'); ax.text(100, B['Copy']['WER_mean'] * 100 + 0.3, 'copy', ha='right', va='bottom', fontsize=5.8, bbox=dict(fc='white', ec='none', pad=0.2))
ax.set_xticks([25, 50, 100]); ax.set_xlabel('Training speakers (%)'); ax.set_ylabel('Test WER (%)')
lo = min(p[1] for p in pts); hi = max([p[1] for p in pts] + [B['Copy']['WER_mean'] * 100]); ax.set_ylim(lo - 2.5, hi + 2.0)
panel(ax, 'c')
save(fig, 'fig3_bridge'); print('ok')
