"""Representation-level evaluation on unseen TEST speakers of the dialect corpus.

(A) Few-shot dialect identification from support sets: prototype sets and query sets are drawn from DISJOINT
    speakers of each region, so identity cannot be matched; nearest-prototype (cosine) accuracy for K clips.
    Representations: support-set code z (each trained model), and baselines with the same protocol:
    mean wav2vec2 states, mean ECAPA embedding, mean CMN mel statistics.
(B) Speaker-code disentanglement: dialect decodability (balanced accuracy of a logistic regression trained on
    TRAIN speakers, tested on TEST speakers) from the projected speaker code, and speaker-verification EER of the
    same code on test utterances.
python repr_analysis.py out.json model1:ckpt1 model2:ckpt2 ...
"""
import sys, json, random, numpy as np, torch
torch.set_num_threads(1)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from data import Corpus, make_splits, REGIONS, R_ID, FEAT
from model import Acoustic, SupportEncoder

D = Corpus('dialect'); sp = np.array(make_splits()['dialect']); y = D.meta.region.map(R_ID).values
te = np.where(sp == 'test')[0]; tr = np.where(sp == 'train')[0]
spk = np.asarray(D.meta.spk.astype(str).tolist(), dtype=object)
te_spk = {r: sorted(set(spk[te][y[te] == R_ID[r]])) for r in REGIONS}
by_spk = {s: [i for i in te if spk[i] == s] for s in set(spk[te])}
KS = [1, 2, 5, 10]; EPIS = 40

# ---- baseline feature banks (per utterance)
W2V = np.load(f'{FEAT}/dialect_w2v.npy').astype(np.float32)[:, 1536:6144]       # layers 6-8
W2V = StandardScaler().fit(W2V[tr]).transform(W2V)
ECA = D.ecapa / np.linalg.norm(D.ecapa, axis=1, keepdims=True)
def melstat(i):
    m = D.get_raw_mel(i); m = m - m.mean(1, keepdims=True); return np.concatenate([m.std(1), np.abs(np.diff(m, axis=1)).mean(1)])
MEL = np.stack([melstat(i) for i in range(len(D.meta))]); MEL = StandardScaler().fit(MEL[tr]).transform(MEL)

def episodes(K, seed):
    rng = random.Random(seed); eps = []
    for e in range(EPIS):
        ep = {}
        for r in REGIONS:
            ss = te_spk[r][:]; rng.shuffle(ss); h = max(1, len(ss) // 2)
            P = [i for s in ss[:h] for i in by_spk[s]]; Q = [i for s in ss[h:] for i in by_spk[s]]
            ep[r] = (rng.sample(P, min(K, len(P))), rng.sample(Q, min(K, len(Q))))
        eps.append(ep)
    return eps

def proto_acc(embed_set, K, seed=0, regions=REGIONS):
    """embed_set: list of utterance indices -> vector. returns per-region accuracy dict"""
    hits = {r: [] for r in regions}
    for ep in episodes(K, seed):
        protos = {r: embed_set(ep[r][0]) for r in REGIONS}
        for r in regions:
            q = embed_set(ep[r][1])
            sims = {c: float(np.dot(q, protos[c]) / (np.linalg.norm(q) * np.linalg.norm(protos[c]) + 1e-9)) for c in REGIONS}
            hits[r].append(max(sims, key=sims.get) == r)
    return {r: float(np.mean(v)) for r, v in hits.items()}

def bank(B):
    return lambda idx: B[idx].mean(0)

def eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores); fnr = 1 - tpr; k = np.nanargmin(np.abs(fnr - fpr)); return float((fpr[k] + fnr[k]) / 2)

def bal(yt, yp):
    return float(np.mean([(yp[yt == k] == k).mean() for k in range(4) if (yt == k).any()]))

def spk_code_metrics(S):
    """S: (N,d) speaker codes for all dialect-corpus utterances"""
    sc = StandardScaler().fit(S[tr]); clf = LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced').fit(sc.transform(S[tr]), y[tr])
    dec = bal(y[te], clf.predict(sc.transform(S[te])))
    Z = S[te] / np.linalg.norm(S[te], axis=1, keepdims=True); sim = Z @ Z.T
    iu = np.triu_indices(len(te), 1); same = (spk[te][:, None] == spk[te][None, :])[iu]
    return dict(dialect_decodability=dec, speaker_eer=eer(sim[iu], same))

out = {'baselines': {}, 'models': {}}
for name, B in [('wav2vec2', W2V), ('ECAPA', ECA), ('mel statistics', MEL)]:
    out['baselines'][name] = {K: proto_acc(bank(B), K) for K in KS}
out['ecapa_input'] = spk_code_metrics(D.ecapa)
for spec in sys.argv[2:]:
    name, path = spec.split(':', 1)
    ck = torch.load(path, weights_only=False); rec = {}
    m = Acoustic(59); m.load_state_dict(ck['model']); m.eval()
    with torch.no_grad():
        S = m.speaker(torch.from_numpy(D.ecapa).float()).numpy()
    rec.update(spk_code_metrics(S))
    if ck.get('senc') is not None and ck['args'].get('method') == 'hyper':
        se = SupportEncoder(); se.load_state_dict(ck['senc'], strict=False); se.eval()
        mels = {}
        def zset(idx):
            with torch.no_grad():
                return se([[torch.from_numpy(D.get_mel(i)) for i in idx]])[0].numpy()
        rec['fewshot'] = {K: proto_acc(zset, K) for K in KS}
        # code statistics: between-region vs within-region spread of 5-clip codes
        Zs, lab = [], []
        for ep in episodes(5, 99)[:15]:
            for r in REGIONS:
                Zs.append(zset(ep[r][0])); lab.append(r)
        Zs = np.array(Zs); lab = np.array(lab)
        within = np.mean([Zs[lab == r].std(0).mean() for r in REGIONS]); between = np.array([Zs[lab == r].mean(0) for r in REGIONS]).std(0).mean()
        rec['code_within_std'] = float(within); rec['code_between_std'] = float(between); rec['code_ratio'] = float(between / (within + 1e-12))
        if ck.get('zcls') is not None:
            W = torch.nn.Linear(128, 4); W.load_state_dict(ck['zcls'])
            hold = ck['args'].get('holdout', '')
            acc = []
            for ep in episodes(5, 7):
                for r in REGIONS:
                    if r == hold: continue
                    with torch.no_grad(): acc.append(int(W(torch.from_numpy(zset(ep[r][1]))).argmax()) == R_ID[r])
            rec['readout_acc_K5'] = float(np.mean(acc))
    rec['holdout'] = ck['args'].get('holdout', '')
    out['models'][name] = rec
    print(name, {k: v for k, v in rec.items() if k != 'fewshot'}, flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(out['baselines'], indent=0)[:1500])
