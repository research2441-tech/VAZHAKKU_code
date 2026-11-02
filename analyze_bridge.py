"""Register-bridge statistics: mean ± s.d. over training seeds, 95% bootstrap CIs over test speakers (seed 0),
paired Wilcoxon tests on sentence WER with Holm correction, Friedman test across speakers, edit precision/recall."""
from config import ROOT
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
R = ROOT + '/work/res'
SYS = [('Copy', None), ('Seq2seq', 'bridge_seq2seq_posterior_s{}'), ('Lattice-noctx', 'bridge_lattice_noctx_s{}'),
       ('Lattice-none', 'bridge_lattice_none_s{}'), ('Lattice-post', 'bridge_lattice_posterior_s{}'),
       ('Lattice-hard', 'bridge_lattice_hard_s{}'), ('Lattice-none+d', 'bmargin_none_s{}'),
       ('Lattice-post+d', 'bmargin_posterior_s{}'), ('Lattice-hard+d', 'bmargin_hard_s{}')]
def boot(df, e, n, B=2000, seed=0):
    rng = np.random.RandomState(seed); g = df.groupby('spk')[[e, n]].sum(); spk = g.index.values; v = []
    for _ in range(B):
        s = rng.choice(spk, len(spk)); sub = g.loc[s]; v.append(sub[e].sum() / sub[n].sum())
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]
def edits(d):
    tp = fp = fn = 0
    for s, h, v in zip(d.src.fillna(''), d.hyp.fillna(''), d.ref.fillna('')):
        s, h, v = str(s).split(), str(h).split(), str(v).split()
        if not (len(s) == len(h) == len(v)): continue
        for a, b, c in zip(s, h, v):
            if b != a and b == c: tp += 1
            elif b != a: fp += 1
            if c != a and b != c: fn += 1
    return tp / max(1, tp + fp), tp / max(1, tp + fn)
out = {}; sent = {}; spkw = {}
base = pd.read_csv(f'{R}/bridge_lattice_posterior_s0_utts.csv')
for name, pat in SYS:
    if pat is None:
        d = base; rec = dict(seeds=[0], WER=[d.we_copy.sum() / d.wn.sum()], CER=[d.ce_copy.sum() / d.cn.sum()],
                             WER_ci=boot(d, 'we_copy', 'wn'), CER_ci=boot(d, 'ce_copy', 'cn'), changed=[0.0], novel=[0.0])
        sent[name] = (d.we_copy / d.wn).values; spkw[name] = d.groupby('spk').apply(lambda g: g.we_copy.sum() / g.wn.sum())
        out[name] = rec; continue
    runs = [(k, pat.format(k)) for k in range(5) if os.path.exists(f'{R}/{pat.format(k)}_utts.csv')]
    if not runs: continue
    rec = dict(seeds=[], WER=[], CER=[], changed=[], novel=[], prec=[], rec=[], delta=[]); sl, sk, dfs = [], [], []
    for k, f in runs:
        d = pd.read_csv(f'{R}/{f}_utts.csv'); J = json.load(open(f'{R}/{f}.json'))
        rec['seeds'].append(k); rec['WER'].append(d.we.sum() / d.wn.sum()); rec['CER'].append(d.ce.sum() / d.cn.sum())
        rec['changed'].append(d.changed.sum() / d.nh.sum()); rec['novel'].append(d.novel.sum() / d.nh.sum())
        p, r = edits(d); rec['prec'].append(p); rec['rec'].append(r); rec['delta'].append(J.get('delta', 0.0))
        dfs.append(d); sl.append((d.we / d.wn).values); sk.append(d.groupby('spk').apply(lambda g: g.we.sum() / g.wn.sum()))
        if k == runs[0][0]:
            for key in ['test_oracle_coverage', 'WER_swapped_region', 'per_region', 'margin_curve']:
                if key in J: rec[key] = J[key]
    P = pd.concat(dfs)            # pooled over seeds: speakers resampled, errors summed over all seeds' outputs
    rec['WER_ci'] = boot(P, 'we', 'wn'); rec['CER_ci'] = boot(P, 'ce', 'cn')
    a = np.mean(sl, 0); sent[name] = a; spkw[name] = pd.concat(sk, axis=1).mean(1)   # seed-averaged scores
    rec['p_vs_copy'] = float(wilcoxon(a, sent['Copy'], zero_method='zsplit').pvalue)
    rec['sent_better'] = float((a < sent['Copy']).mean()); rec['sent_worse'] = float((a > sent['Copy']).mean())
    out[name] = rec
for name, rec in out.items():
    for k in ['WER', 'CER', 'changed', 'novel', 'prec', 'rec']:
        if k in rec and rec[k]: rec[k + '_mean'] = float(np.mean(rec[k])); rec[k + '_sd'] = float(np.std(rec[k], ddof=1)) if len(rec[k]) > 1 else 0.0
# paired comparisons among lattice variants on seed-averaged sentence WER, Holm-corrected
pairs = [('Lattice-hard+d', 'Lattice-hard'), ('Lattice-post+d', 'Lattice-post'), ('Lattice-hard', 'Lattice-post'),
         ('Lattice-post', 'Lattice-none'), ('Lattice-post', 'Lattice-noctx'), ('Lattice-hard+d', 'Lattice-post+d')]
tests = []
for a, b in pairs:
    if a in sent and b in sent:
        tests.append([a, b, float(np.mean(sent[a]) - np.mean(sent[b])), float(wilcoxon(sent[a], sent[b], zero_method='zsplit').pvalue)])
order = np.argsort([t[3] for t in tests]); m = len(tests); adj = [0.0] * m; run = 0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * tests[i][3])); adj[i] = run
for t, p in zip(tests, adj): t.append(p)
out['paired'] = [dict(a=t[0], b=t[1], mean_diff=t[2], p=t[3], p_holm=t[4]) for t in tests]
fr = [n for n in ['Copy', 'Lattice-none', 'Lattice-post', 'Lattice-hard', 'Lattice-hard+d', 'Lattice-post+d'] if n in spkw]
M = pd.concat([spkw[n] for n in fr], axis=1).dropna()
st = friedmanchisquare(*[M.iloc[:, j].values for j in range(M.shape[1])])
out['friedman'] = dict(systems=fr, n_speakers=int(M.shape[0]), chi2=float(st.statistic), p=float(st.pvalue),
                       mean_rank={n: float(r) for n, r in zip(fr, M.rank(axis=1).mean().values)})
ref = []
for s, v in zip(base.src, base.ref):
    a, b = str(s).split(), str(v).split()
    if len(a) == len(b): ref += [x != y for x, y in zip(a, b)]
out['reference_change_rate'] = float(np.mean(ref))
sens = {}
for f in sorted(glob.glob(f'{R}/bsens_*.json')):
    J = json.load(open(f)); sens[os.path.basename(f)[6:-5]] = dict(WER=J['WER'], CER=J['CER'], coverage=J.get('test_oracle_coverage'),
                                                                 changed=J['changed_rate'], novel=J['novel_rate'])
out['sensitivity'] = sens
json.dump(out, open(f'{R}/bridge_final.json', 'w'), indent=1)
for k, v in out.items():
    if isinstance(v, dict) and 'WER_mean' in v:
        print(f"{k:16s} WER {v['WER_mean']*100:.2f}±{v['WER_sd']*100:.2f} n={len(v['seeds'])} CER {v['CER_mean']*100:.2f} ch {v['changed_mean']*100:.1f} nov {v['novel_mean']*100:.1f} P {v.get('prec_mean',0):.3f} R {v.get('rec_mean',0):.3f} d {v.get('delta')}")
print(out['paired']); print(out['friedman']); print(sens)
