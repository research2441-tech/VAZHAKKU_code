"""Dialect identification instrument: multinomial logistic regression on pooled hidden states of a frozen Tamil
wav2vec2 recogniser. Trained on dialect-corpus TRAIN speakers, model selection on VAL speakers, reported on TEST
speakers. Also writes the noisy-prior posterior q(r|u) for IndicVoices speakers.
python dialect_probe.py
"""
from config import ROOT
import json, pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data import Corpus, make_splits, REGIONS, R_ID, district_prior, FEAT

D = 1536                                  # per-layer block (mean + std of 768)
LAYER_IDX = {5: 0, 6: 1, 7: 2, 8: 3}
PKL = f'{FEAT}/dialect_probe.pkl'

def cols(layers):
    return np.concatenate([np.arange(LAYER_IDX[l] * D, (LAYER_IDX[l] + 1) * D) for l in layers])

def load():
    return pickle.load(open(PKL, 'rb'))

def predict_wavs(probe, wavs):
    from w2v_feats import embed
    X = np.stack([embed(w) for w in wavs])
    return predict_feats(probe, X)

def predict_feats(probe, X):
    sc, clf, layers = probe
    return clf.predict_log_proba(sc.transform(X[:, cols(layers)].astype(np.float32)))

def bal_acc(y, p):
    return float(np.mean([(p[y == k] == k).mean() for k in range(4) if (y == k).any()]))

if __name__ == '__main__':
    import sys
    IV_ONLY = '--iv_only' in sys.argv; NO_IV = '--no_iv' in sys.argv
    c = Corpus('dialect'); sp = np.array(make_splits()['dialect']); y = c.meta.region.map(R_ID).values
    X = np.load(f'{FEAT}/dialect_w2v.npy').astype(np.float32)
    tr, va, te = [np.where(sp == s)[0] for s in ('train', 'val', 'test')]
    best = None; grid = []
    if IV_ONLY:
        probe = load(); res = json.load(open(ROOT + '/work/res/dialect_probe.json'))
    for layers in ([] if IV_ONLY else [(5,), (6,), (7,), (8,), (6, 7), (5, 6, 7), (6, 7, 8), (5, 6, 7, 8)]):
        for C in [0.003, 0.01, 0.03, 0.1]:
            cc = cols(layers); sc = StandardScaler().fit(X[tr][:, cc])
            clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(sc.transform(X[tr][:, cc]), y[tr])
            b = bal_acc(y[va], clf.predict(sc.transform(X[va][:, cc])))
            grid.append(dict(layers=list(layers), C=C, val_bal=b)); print(layers, C, round(b, 4), flush=True)
            if best is None or b > best[0]:
                best = (b, (sc, clf, layers))
    if not IV_ONLY:
        probe = best[1]; pickle.dump(probe, open(PKL, 'wb'))
    lp = predict_feats(probe, X[te]); pred = lp.argmax(1)
    cm = np.zeros((4, 4), int)
    for t_, p_ in zip(y[te], pred): cm[t_, p_] += 1
    df = pd.DataFrame(lp); df['spk'] = c.meta.spk.values[te]; df['y'] = y[te]
    g = df.groupby('spk').mean()
    if not IV_ONLY: res = dict(layers=list(probe[2]), C=probe[1].C, val_bal=best[0], test_acc=float((pred == y[te]).mean()), test_bal_acc=bal_acc(y[te], pred),
               test_spk_acc=float((g[[0, 1, 2, 3]].values.argmax(1) == g.y.values).mean()), confusion=cm.tolist(),
               n_test=int(len(te)), n_test_spk=int(c.meta.spk.iloc[te].nunique()), grid=grid)
    if NO_IV:
        json.dump(res, open(ROOT + '/work/res/dialect_probe.json', 'w'), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != 'grid'}, indent=1)); sys.exit(0)
    # ---- noisy prior x acoustic likelihood -> q(r|u) for IndicVoices speakers
    iv = Corpus('iv'); Xi = np.load(f'{FEAT}/iv_w2v.npy').astype(np.float32); idx = np.load(f'{FEAT}/iv_w2v_idx.npy')
    lpi = predict_feats(probe, Xi)
    dfi = pd.DataFrame(lpi, columns=REGIONS); dfi['spk'] = iv.meta.spk.values[idx]
    rows = []
    for s, g in dfi.groupby('spk'):
        d = iv.meta.district[iv.meta.spk == s].iloc[0]; prior = district_prior(d)
        ll = g[REGIONS].values.mean(0)
        post = np.log(prior) + ll; post = np.exp(post - post.max()); post /= post.sum()
        rows.append(dict(spk=s, district=d, **{r: post[k] for k, r in enumerate(REGIONS)}, prior_arg=REGIONS[int(np.argmax(prior))],
                         post_arg=REGIONS[int(np.argmax(post))], cls_arg=REGIONS[int(np.argmax(ll))],
                         prior_max=float(prior.max()), post_max=float(post.max())))
    P = pd.DataFrame(rows); P.to_csv(f'{FEAT}/iv_region_posterior.csv', index=False)
    res.update(iv_speakers=len(P), iv_post_agrees_prior=float((P.prior_arg == P.post_arg).mean()),
               iv_cls_agrees_prior=float((P.prior_arg == P.cls_arg).mean()),
               iv_prior_entropy=float(np.mean([-(district_prior(d) * np.log(district_prior(d))).sum() for d in P.district])),
               iv_post_entropy=float(-(P[REGIONS].values * np.log(P[REGIONS].values + 1e-12)).sum(1).mean()),
               iv_cls_agree_by_prior={r: float((g.cls_arg == r).mean()) for r, g in P.groupby('prior_arg')})
    json.dump(res, open(ROOT + '/work/res/dialect_probe.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'grid'}, indent=1))
