"""Synthesis figures from evaluation CSVs (grayscale, legends outside data)."""
from config import ROOT
import glob, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from figs_common import style, save, panel, GREYS, HATCH, MARK, LS
style()
R = ROOT + '/work/res'
REG = ['Northern', 'Southern', 'Western', 'Central']
ABBR = {'Northern': 'N', 'Southern': 'S', 'Western': 'W', 'Central': 'C'}

def boot_ci(df, col, n=2000, seed=0):
    """95% percentile CI by resampling voices (speakers), not items"""
    rng = np.random.RandomState(seed); spk = df.spk.unique(); g = df.groupby('spk')[col].agg(['sum', 'count'])
    vals = []
    for _ in range(n):
        s = rng.choice(spk, len(spk), replace=True); sub = g.loc[s]
        vals.append(sub['sum'].sum() / sub['count'].sum())
    return np.percentile(vals, [2.5, 97.5])

def heatmaps(systems, labels, name='fig4_control'):
    fig, axs = plt.subplots(1, len(systems), figsize=(7.1, 2.05), gridspec_kw=dict(wspace=0.12))
    for ax, s, lab in zip(axs, systems, labels):
        df = pd.read_csv(f'{R}/eval_seen_{s}.csv')
        M = np.zeros((4, 4))
        for i, a in enumerate(REG):
            for j, b in enumerate(REG):
                sub = df[(df.src == a) & (df.tgt == b)]
                M[i, j] = sub.p_tgt.mean() if len(sub) else np.nan
        ax.imshow(M, cmap='Greys', vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.2, color='white' if M[i, j] > 0.55 else 'black')
        ax.set_xticks(range(4)); ax.set_xticklabels([ABBR[r] for r in REG]); ax.set_yticks(range(4))
        ax.set_yticklabels([ABBR[r] for r in REG] if ax is axs[0] else [])
        ax.set_xlabel('target region'); ax.grid(False)
        if ax is axs[0]: ax.set_ylabel('voice region')
        ax.set_title(lab, fontsize=7.2, pad=3)
    save(fig, name)

def tradeoff(systems, labels, name='fig5_tradeoff'):
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.55), gridspec_kw=dict(wspace=0.32))
    pts = []
    for k, (s, lab) in enumerate(zip(systems, labels)):
        df = pd.read_csv(f'{R}/eval_seen_{s}.csv'); x = df[df.cross == 1]
        ali = x.p_src.mean(); lo, hi = boot_ci(x, 'p_src'); sc = df.secs.mean(); slo, shi = boot_ci(df, 'secs')
        dca = df.correct.mean(); dlo, dhi = boot_ci(df, 'correct')
        pts.append((lab, ali, lo, hi, sc, slo, shi, dca, dlo, dhi))
        axs[0].errorbar(ali, sc, xerr=[[ali - lo], [hi - ali]], yerr=[[sc - slo], [shi - sc]], fmt=MARK[k % 8], color='black',
                        mfc=['black', 'white', GREYS[2], 'white', GREYS[3], 'black', 'white', GREYS[1]][k % 8], ms=4.5, lw=0.6, capsize=1.5, label=lab)
    axs[0].set_xlabel('Accent leakage index ALI (lower is better)'); axs[0].set_ylabel('Speaker similarity SECS')
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=6.0)
    panel(axs[0], 'a')
    ax = axs[1]; ax.set_position([0.70, ax.get_position().y0, 0.27, ax.get_position().height])
    y = np.arange(len(pts))
    ax.barh(y, [p[7] * 100 for p in pts], 0.6, color=GREYS[3], edgecolor='black', lw=0.5,
            xerr=[[ (p[7] - p[8]) * 100 for p in pts], [(p[9] - p[7]) * 100 for p in pts]], error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_yticks(y); ax.set_yticklabels([p[0] for p in pts], fontsize=6.0); ax.invert_yaxis()
    ax.set_xlabel('DCA (%)'); ax.axvline(25, color='black', lw=0.6, ls=':'); ax.grid(axis='y', visible=False)
    panel(ax, 'b')
    save(fig, name); return pts

def kshot(systems, labels, name='fig6_kshot'):
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw=dict(wspace=0.3))
    for k, (s, lab) in enumerate(zip(systems, labels)):
        f = f'{R}/eval_kshot_{s}.csv'
        try: df = pd.read_csv(f)
        except FileNotFoundError: continue
        g = df.groupby('K'); x = sorted(df.K.unique())
        axs[0].plot(x, [g.get_group(v).correct.mean() * 100 for v in x], marker=MARK[k], ls=LS[k], color='black', mfc='white' if k % 2 else 'black', ms=3.5, label=lab)
        xc = df[df.cross == 1].groupby('K')
        axs[1].plot(x, [xc.get_group(v).p_src.mean() for v in x], marker=MARK[k], ls=LS[k], color='black', mfc='white' if k % 2 else 'black', ms=3.5, label=lab)
    for ax, yl, l in [(axs[0], 'DCA (%)', 'a'), (axs[1], 'ALI', 'b')]:
        ax.set_xscale('log', base=2); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(['1', '2', '5', '10', '20'])
        ax.set_xlabel('support clips $K$'); ax.set_ylabel(yl); panel(ax, l)
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save(fig, name)

def lodo(folds, systems, labels, name='fig7_lodo'):
    """DCA on the held-out region (target = held-out) for each fold; bars per system."""
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw=dict(wspace=0.32))
    w = 0.8 / len(systems)
    for k, (s, lab) in enumerate(zip(systems, labels)):
        acc, ali = [], []
        for f in folds:
            p = f'{R}/eval_lodo_lodo_{s}_{f}.csv'
            try: df = pd.read_csv(p)
            except FileNotFoundError: acc.append(np.nan); ali.append(np.nan); continue
            acc.append(df.correct.mean() * 100); ali.append(df[df.cross == 1].p_src.mean())
        x = np.arange(len(folds)) + (k - (len(systems) - 1) / 2) * w
        axs[0].bar(x, acc, w, color=[GREYS[1], GREYS[3], 'white', GREYS[4]][k % 4], hatch=HATCH[k % 7], edgecolor='black', lw=0.5, label=lab)
        axs[1].bar(x, ali, w, color=[GREYS[1], GREYS[3], 'white', GREYS[4]][k % 4], hatch=HATCH[k % 7], edgecolor='black', lw=0.5, label=lab)
    for ax, yl, l in [(axs[0], 'DCA on held-out region (%)', 'a'), (axs[1], 'ALI', 'b')]:
        ax.set_xticks(range(len(folds))); ax.set_xticklabels([f'held out: {f}' for f in folds]); ax.set_ylabel(yl)
        ax.grid(axis='x', visible=False); panel(ax, l)
    axs[0].axhline(25, color='black', lw=0.6, ls=':')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save(fig, name)
