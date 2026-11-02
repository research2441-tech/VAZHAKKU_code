"""Register bridge: standard (normalised) Tamil -> region-conditioned spoken form.

Candidates for every input word come from a lattice of alternations attested in the training pairs
(whole-word mappings, suffix rewrites, single-character internal substitutions); a small context model
scores the candidates. An unconstrained character-level seq2seq model is trained as the comparison.

python bridge.py --mode {lattice,seq2seq} --region {posterior,hard,none} --seed 0 --out res/bridge_xxx.json
"""
import argparse, json, math, random, time, collections, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from data import FEAT, REGIONS, district_prior, make_splits
from tamil_g2p import clean

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='lattice'); ap.add_argument('--region', default='posterior')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--no_context', action='store_true')
ap.add_argument('--threads', type=int, default=1); ap.add_argument('--cmin', type=int, default=3)
ap.add_argument('--max_cand', type=int, default=12); ap.add_argument('--frac', type=float, default=1.0)
ap.add_argument('--margins', default='', help='comma list of identity biases; the best on validation speakers is used for test')
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

meta = pd.read_csv(f'{FEAT}/iv_meta.csv', keep_default_na=False)
split = np.array(make_splits()['iv'])
post_path = f'{FEAT}/iv_region_posterior.csv'
if a.region == 'posterior':
    P = pd.read_csv(post_path).set_index('spk')
    meta['rp'] = [P.loc[s, REGIONS].values.astype(np.float32) for s in meta.spk]
elif a.region == 'hard':
    meta['rp'] = [np.eye(4, dtype=np.float32)[int(np.argmax(district_prior(d)))] for d in meta.district]
else:
    meta['rp'] = [np.full(4, 0.25, dtype=np.float32) for _ in meta.district]
# evaluation always reports the region of record (hard district map) for per-region breakdowns
meta['hreg'] = [int(np.argmax(district_prior(d))) for d in meta.district]

def pairs(df):
    out = []
    for n, v, rp in zip(df.normalized, df.text, df.rp):
        nw, vw = clean(n).split(), clean(v).split()
        if nw and len(nw) == len(vw):
            out.append((nw, vw, rp))
    return out

tr = meta[split == 'train']; va = meta[split == 'val']; te = meta[split == 'test']
P_tr = pairs(tr)
if a.frac < 1.0:   # learning curve: keep a random subset of training SPEAKERS' sentences
    rs = np.random.RandomState(a.seed); spk_tr = sorted(tr.spk.unique()); keep = set(rs.choice(spk_tr, int(round(a.frac * len(spk_tr))), replace=False))
    P_tr = pairs(tr[tr.spk.isin(keep)])
print('train sentence pairs', len(P_tr), 'test sentences', len(te), flush=True)

# ------------------------------------------------------------------ lattice of attested alternations
word_tab = collections.defaultdict(lambda: np.zeros(4))      # (w,v) -> region-weighted counts
suf_tab = collections.defaultdict(lambda: np.zeros(4))       # (sigma, sigma') -> counts
sub_tab = collections.defaultdict(lambda: np.zeros(4))       # (c, c') internal substitution
spoken_vocab = set()
def lcp(x, y):
    k = 0
    while k < min(len(x), len(y)) and x[k] == y[k]: k += 1
    return k
for nw, vw, rp in P_tr:
    for w, v in zip(nw, vw):
        word_tab[(w, v)] += rp; spoken_vocab.add(v)
        if w == v: continue
        k = lcp(w, v)
        if k >= 2 and len(w) - k <= 6 and len(v) - k <= 6:
            suf_tab[(w[k:], v[k:])] += rp
        if len(w) == len(v):
            d = [i for i in range(len(w)) if w[i] != v[i]]
            if len(d) == 1:
                sub_tab[(w[d[0]], v[d[0]])] += rp
by_w = collections.defaultdict(list)
for (w, v), c in word_tab.items(): by_w[w].append((v, c))
sufs = [(s, t, c) for (s, t), c in suf_tab.items() if c.sum() >= a.cmin - 1e-6]
sufs_by_end = collections.defaultdict(list)
for s, t, c in sufs: sufs_by_end[s[-1:] if s else ''].append((s, t, c))
subs = [(s, t, c) for (s, t), c in sub_tab.items() if c.sum() >= a.cmin - 1e-6]
print('lattice: words', len(by_w), 'suffix rules', len(sufs), 'subs', len(subs), flush=True)

def candidates(w):
    """returns list of (cand, feature vector[8])"""
    C = {}
    def add(v, kind, c):
        f = C.setdefault(v, np.zeros(4 * 3 + 2, dtype=np.float32))
        f[kind * 4:(kind + 1) * 4] += c
    add(w, 0, np.zeros(4))
    for v, c in by_w.get(w, []): add(v, 0, c)
    for s, t, c in sufs_by_end.get(w[-1:], []) + sufs_by_end.get('', []):
        if w.endswith(s) and len(w) - len(s) >= 2:
            add(w[:len(w) - len(s)] + t, 1, c)
    for i, ch in enumerate(w[:3]):          # internal alternations are mostly in the first syllables
        for s, t, c in subs:
            if ch == s: add(w[:i] + t + w[i + 1:], 2, c)
    items = list(C.items())
    items.sort(key=lambda kv: -(kv[1][:4].sum() * 10 + kv[1][4:8].sum() + kv[1][8:12].sum() * 0.5 + (kv[0] == w) * 1e-3))
    items = items[:a.max_cand]
    if w not in [k for k, _ in items]: items[-1] = (w, C[w])
    out = []
    for v, f in items:
        g = np.concatenate([np.log1p(f[:12]), [float(v == w), (len(v) - len(w)) / 4.0, float(v in spoken_vocab)]]).astype(np.float32)
        out.append((v, g))
    return out

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev = d.copy(); d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)

def cer(ref, hyp):
    e, n = wer(' '.join(list(ref.replace(' ', '_'))), ' '.join(list(hyp.replace(' ', '_'))))
    return e, n

# ------------------------------------------------------------------ char vocabulary
chars = sorted({ch for nw, vw, _ in P_tr for w in nw + vw for ch in w})
CH = {c: i + 3 for i, c in enumerate(chars)}   # 0 pad 1 bos 2 eos
def enc_word(w): return [CH.get(c, 0) for c in w][:24]

class WordEnc(nn.Module):
    def __init__(self, d=96):
        super().__init__(); self.emb = nn.Embedding(len(CH) + 3, d, padding_idx=0); self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
    def forward(self, words):
        L = max(1, max(len(w) for w in words)); X = torch.zeros(len(words), L, dtype=torch.long)
        for i, w in enumerate(words): X[i, :len(w)] = torch.tensor(w) if w else torch.zeros(0)
        h, _ = self.gru(self.emb(X)); m = (X > 0).float()[:, :, None]
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

class LatticeScorer(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        self.we = WordEnc(d); self.ctx = nn.GRU(2 * d, d, batch_first=True, bidirectional=True)
        self.reg = nn.Linear(4, 2 * d); self.bil = nn.Bilinear(2 * d, 2 * d, 1)
        self.feat = nn.Sequential(nn.Linear(15 + 4 * 3, 32), nn.SiLU(), nn.Linear(32, 1))
    def forward(self, nw, rp, cands):
        r = torch.from_numpy(rp)[None]
        if a.no_context:
            ctx = self.reg(r).repeat(len(nw), 1)
        else:
            h, _ = self.ctx(self.we([enc_word(w) for w in nw])[None]); ctx = h[0] + self.reg(r)
        scores = []
        for i, cs in enumerate(cands):
            ce = self.we([enc_word(v) for v, _ in cs])
            f = torch.from_numpy(np.stack([g for _, g in cs]))
            rr = r.repeat(len(cs), 1)
            # region-specific count features weighted by the region code
            rf = torch.cat([f[:, :12] * rr.repeat(1, 3)], 1)
            s = self.bil(ctx[i:i + 1].repeat(len(cs), 1), ce)[:, 0] + self.feat(torch.cat([f, rf], 1))[:, 0]
            scores.append(s)
        return scores

def run_lattice():
    model = LatticeScorer(); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    cache = {}
    def cands(w):
        if w not in cache: cache[w] = candidates(w)
        return cache[w]
    cover = tot = 0
    for nw, vw, _ in P_tr:
        for w, v in zip(nw, vw):
            tot += 1; cover += v in [c for c, _ in cands(w)]
    print('train oracle coverage', cover / tot, flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        random.shuffle(P_tr); tl = 0; n = 0
        for nw, vw, rp in P_tr:
            cs = [cands(w) for w in nw]
            tgt = [[c for c, _ in c_].index(v) if v in [c for c, _ in c_] else -1 for c_, v in zip(cs, vw)]
            if all(t < 0 for t in tgt): continue
            sc = model(nw, rp, cs)
            loss = sum(F.cross_entropy(s[None], torch.tensor([t])) for s, t in zip(sc, tgt) if t >= 0) / max(1, sum(t >= 0 for t in tgt))
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); n += 1
        print('epoch', ep, tl / n, round(time.time() - t0), flush=True)
    model.eval()
    def predict(n_text, rp, delta=0.0):
        nw = clean(n_text).split()
        if not nw: return ''
        cs = [cands(w) for w in nw]
        with torch.no_grad(): sc = model(nw, rp, cs)
        out = []
        for w, c, s in zip(nw, cs, sc):
            s = s.clone()
            if delta:   # identity bias: keep the standard word unless an alternative wins by more than delta
                s[[k for k, (v, _) in enumerate(c) if v == w][0]] += delta
            out.append(c[int(s.argmax())][0])
        return ' '.join(out)
    return predict, cands

class Seq2Seq(nn.Module):
    def __init__(self, d=192):
        super().__init__()
        V = len(CH) + 4; self.V = V
        self.emb = nn.Embedding(V, d, padding_idx=0); self.pos = nn.Embedding(512, d); self.reg = nn.Linear(4, d)
        self.tf = nn.Transformer(d, 4, 3, 3, 512, 0.1, batch_first=True); self.out = nn.Linear(d, V)
    def encode_txt(self, s): return [CH.get(c, len(CH) + 3) if c != ' ' else len(CH) + 3 for c in s][:300]
    def forward(self, src, tgt_in, rp):
        ps = torch.arange(src.shape[1])[None]; pt = torch.arange(tgt_in.shape[1])[None]
        S = self.emb(src) + self.pos(ps) + self.reg(rp)[:, None]; T = self.emb(tgt_in) + self.pos(pt)
        mask = self.tf.generate_square_subsequent_mask(tgt_in.shape[1])
        return self.out(self.tf(S, T, tgt_mask=mask, src_key_padding_mask=src == 0, tgt_key_padding_mask=tgt_in == 0))

def run_seq2seq():
    m = Seq2Seq(); opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    data = [(m.encode_txt(' '.join(nw)), m.encode_txt(' '.join(vw)), rp) for nw, vw, rp in P_tr]
    data = [d for d in data if len(d[0]) <= 160 and len(d[1]) <= 160]
    sp = len(CH) + 3; inv = {v: k for k, v in CH.items()}; inv[sp] = ' '
    t0 = time.time(); bs = 32
    for ep in range(a.epochs * 3):
        random.shuffle(data); tl = n = 0
        for k in range(0, len(data), bs):
            B = data[k:k + bs]; Ls = max(len(s) for s, _, _ in B); Lt = max(len(t) for _, t, _ in B) + 1
            src = torch.zeros(len(B), Ls, dtype=torch.long); tin = torch.zeros(len(B), Lt, dtype=torch.long); tout = torch.zeros(len(B), Lt, dtype=torch.long)
            for i, (s, t, _) in enumerate(B):
                src[i, :len(s)] = torch.tensor(s); tin[i, :len(t) + 1] = torch.tensor([1] + t); tout[i, :len(t) + 1] = torch.tensor(t + [2])
            rp = torch.from_numpy(np.stack([r for _, _, r in B]))
            lg = m(src, tin, rp); loss = F.cross_entropy(lg.reshape(-1, m.V), tout.reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); tl += loss.item(); n += 1
        print('epoch', ep, tl / n, round(time.time() - t0), flush=True)
    m.eval()
    def predict(n_text, rp):
        s = m.encode_txt(clean(n_text))
        if not s: return ''
        src = torch.tensor([s]); out = [1]
        with torch.no_grad():
            S = m.emb(src) + m.pos(torch.arange(src.shape[1])[None]) + m.reg(torch.from_numpy(rp)[None])[:, None]
            mem = m.tf.encoder(S)                                       # encoder run once per sentence
            for _ in range(int(len(s) * 1.3) + 5):
                tin = torch.tensor([out]); T = m.emb(tin) + m.pos(torch.arange(tin.shape[1])[None])
                h = m.tf.decoder(T, mem, tgt_mask=m.tf.generate_square_subsequent_mask(tin.shape[1]))
                nx = int(m.out(h)[0, -1].argmax())
                if nx == 2: break
                out.append(nx)
        return ''.join(inv.get(c, '') for c in out[1:]).strip()
    return predict, None

predict, cands = run_lattice() if a.mode == 'lattice' else run_seq2seq()

DELTA = 0.0
def evaluate(df, swap=False, delta=None):
    delta = DELTA if delta is None else delta
    rows = []
    for n, v, rp, hr, spk in zip(df.normalized, df.text, df.rp, df.hreg, df.spk):
        if not clean(n) or not clean(v): continue
        r_in = rp
        if swap:   # counterfactual: condition on a different region than the speaker's
            r_in = np.roll(rp, 2)
        h = predict(n, r_in, delta) if a.mode == 'lattice' else predict(n, r_in)
        we, wn = wer(clean(v), h); ce, cn = cer(clean(v), h)
        we0, _ = wer(clean(v), clean(n)); ce0, _ = cer(clean(v), clean(n))
        hw = h.split(); inw = set(clean(n).split())
        unatt = sum(w not in spoken_vocab for w in hw)
        novel = sum((w not in spoken_vocab) and (w not in inw) for w in hw)          # invented forms
        changed = sum(w not in inw for w in hw)
        rows.append(dict(spk=spk, hreg=hr, we=we, wn=wn, ce=ce, cn=cn, we_copy=we0, ce_copy=ce0, unatt=unatt, novel=novel,
                         changed=changed, nh=len(hw), src=clean(n), hyp=h, ref=clean(v)))
    return pd.DataFrame(rows)

margin_curve = []
if a.margins and a.mode == 'lattice':
    for dl in [float(x) for x in a.margins.split(',')]:
        Rv = evaluate(va, delta=dl); Rt = evaluate(te, delta=dl)
        margin_curve.append(dict(delta=dl, val_WER=Rv.we.sum() / Rv.wn.sum(), test_WER=Rt.we.sum() / Rt.wn.sum(),
                                 val_changed=Rv.changed.sum() / Rv.nh.sum(), test_changed=Rt.changed.sum() / Rt.nh.sum()))
        print('margin', margin_curve[-1], flush=True)
    DELTA = min(margin_curve, key=lambda r: r['val_WER'])['delta']
    print('selected delta on validation', DELTA, flush=True)
t0 = time.time()
R = evaluate(te)
res = dict(mode=a.mode, region=a.region, seed=a.seed, no_context=a.no_context,
           WER=R.we.sum() / R.wn.sum(), CER=R.ce.sum() / R.cn.sum(),
           WER_copy=R.we_copy.sum() / R.wn.sum(), CER_copy=R.ce_copy.sum() / R.cn.sum(),
           unattested=R.unatt.sum() / max(1, R.nh.sum()), novel_rate=R.novel.sum() / max(1, R.nh.sum()),
           changed_rate=R.changed.sum() / max(1, R.nh.sum()),
           per_region={REGIONS[k]: dict(WER=g.we.sum() / g.wn.sum(), CER=g.ce.sum() / g.cn.sum(), n=len(g)) for k, g in R.groupby('hreg')},
           eval_seconds=time.time() - t0, delta=DELTA, margin_curve=margin_curve)
if a.mode == 'lattice':
    cov = tot = 0
    for n, v in zip(te.normalized, te.text):
        nw, vw = clean(n).split(), clean(v).split()
        if len(nw) == len(vw):
            for w, x in zip(nw, vw):
                tot += 1; cov += x in [c for c, _ in cands(w)]
    res['test_oracle_coverage'] = cov / max(1, tot)
if a.region != 'none' and a.mode == 'lattice':
    Rs = evaluate(te, swap=True)
    res['WER_swapped_region'] = Rs.we.sum() / Rs.wn.sum(); res['CER_swapped_region'] = Rs.ce.sum() / Rs.cn.sum()
R.to_csv(a.out.replace('.json', '_utts.csv'), index=False)
json.dump(res, open(a.out, 'w'), indent=1)
print(json.dumps(res, indent=1))
