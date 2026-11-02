"""Aggregate evaluation CSVs into the numbers reported in the manuscript (writes res/summary.json)."""
from config import ROOT
import glob, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
R = ROOT + '/work/res'

def boot(df, col, n=2000, seed=0):
    rng = np.random.RandomState(seed); g = df.groupby('spk')[col].agg(['sum', 'count']); spk = g.index.values
    v = []
    for _ in range(n):
        s = rng.choice(spk, len(spk)); sub = g.loc[s]; v.append(sub['sum'].sum() / sub['count'].sum())
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj

def system_summary(name, protocol='seen'):
    df = pd.read_csv(f'{R}/eval_{protocol}_{name}.csv'); x = df[df.cross == 1]; m = df[df.cross == 0]
    out = dict(n=len(df), DCA=df.correct.mean(), DCA_ci=boot(df, 'correct'),
               DCA_cross=x.correct.mean(), DCA_matched=m.correct.mean(),
               ALI=x.p_src.mean(), ALI_ci=boot(x, 'p_src'), Ptgt=df.p_tgt.mean())
    if 'secs' in df:
        out.update(SECS=df.secs.mean(), SECS_ci=boot(df, 'secs'), CER=df.cer.mean(), CER_ci=boot(df, 'cer'))
    if 'mcd' in df and df.mcd.notna().any():
        out.update(MCD=df.mcd.mean(), MCD_ci=boot(df[df.mcd.notna()], 'mcd'))
    if 'synth_s' in df:
        out.update(RTF=(df.synth_s.sum() / (df.frames.sum() * 256 / 16000)))
    return {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in out.items()}

def paired(a, b, col, protocol='seen', subset=None):
    A = pd.read_csv(f'{R}/eval_{protocol}_{a}.csv'); B = pd.read_csv(f'{R}/eval_{protocol}_{b}.csv')
    key = ['spk', 'tgt', 'text']
    M = A.merge(B, on=key, suffixes=('_a', '_b'))
    if subset == 'cross': M = M[M.cross_a == 1]
    da, db = M[f'{col}_a'].values, M[f'{col}_b'].values
    diff = da - db
    p = float(wilcoxon(da, db, zero_method='zsplit').pvalue) if np.any(diff != 0) else 1.0
    # matched-pairs rank-biserial effect size
    nz = diff[diff != 0]; r = np.argsort(np.argsort(np.abs(nz))) + 1
    rb = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()) if len(nz) else 0.0
    return dict(n=int(len(M)), mean_a=float(da.mean()), mean_b=float(db.mean()), p=p, rank_biserial=rb)

if __name__ == '__main__':
    import sys
    systems = [s for s in sys.argv[1].split(',')]
    S = {s: system_summary(s) for s in systems}
    json.dump(S, open(f'{R}/summary_seen.json', 'w'), indent=1)
    for s, v in S.items():
        print(s, {k: (round(v[k], 4) if isinstance(v[k], float) else v[k]) for k in v})
