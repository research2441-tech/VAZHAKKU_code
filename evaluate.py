"""Synthesis-side evaluation.

python evaluate.py --ckpt ckpt/x.pt --name X [--protocol seen|lodo|kshot|nfe|pipeline] [--ft] [--audio]
All test voices are dialect-corpus TEST speakers (never seen in any training stage).
"""
from config import ROOT
import argparse, json, time, random, numpy as np, pandas as pd, torch, torch.nn.functional as F
from data import Corpus, make_splits, collate, SYM, SYM_ID, REGIONS, R_ID, FEAT, denorm
from model import Acoustic, SupportEncoder, HyperLoRA, lora_init_vector
from tamil_g2p import g2p
from tokens import text_to_ids
import dialect_probe, audio_utils, asr_utils

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--name', required=True)
ap.add_argument('--protocol', default='seen'); ap.add_argument('--ft', action='store_true', help='test-time LoRA fine-tuning on support')
ap.add_argument('--audio', action='store_true', help='vocode and run SECS / CER'); ap.add_argument('--K', type=int, default=5)
ap.add_argument('--n_text', type=int, default=2); ap.add_argument('--seed', type=int, default=123)
ap.add_argument('--nfe', type=int, default=10); ap.add_argument('--threads', type=int, default=1)
ap.add_argument('--inner_steps', type=int, default=-1); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--lodo_region', default=''); ap.add_argument('--bridge_file', default='')
a = ap.parse_args()
torch.set_num_threads(a.threads)
import os
SEL = json.load(open(ROOT + '/work/res/decoding_selection.json')) if os.path.exists(ROOT + '/work/res/decoding_selection.json') else dict(temp=0.667, nfe=10)
TEMP = SEL['temp']; T0 = SEL.get('t0', 0.0)
if a.protocol != 'nfe': a.nfe = SEL['nfe']
ck = torch.load(a.ckpt, weights_only=False)
args = ck['args']; method = args.get('method', 'clone'); regression = ck.get('regression', args.get('regression', False))
model = Acoustic(len(SYM), regression=regression); model.load_state_dict(ck['model']); model.eval()
senc = SupportEncoder(); hyper = None
if method == 'hyper':
    senc.load_state_dict(ck['senc']); hyper = HyperLoRA(model); hyper.load_state_dict(ck['hyper']); senc.eval(); hyper.eval()
D = Corpus('dialect'); sp = np.array(make_splits()['dialect'])
test_idx = np.where(sp == 'test')[0]
test_spk = sorted(D.meta.spk.iloc[test_idx].unique())
by_spk = {s: [i for i in test_idx if D.meta.spk.iloc[i] == s] for s in test_spk}
spk_reg = {s: D.meta.region.iloc[by_spk[s][0]] for s in test_spk}
probe = dialect_probe.load()
def dialect_lp(wav):
    return dialect_probe.predict_wavs(probe, [wav])[0]
rng = random.Random(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

def adapter(r, K, exclude_spk, inner_steps=None):
    """LoRA vector for target dialect r using K support utterances of TEST speakers of r (not the voice)."""
    cand = [i for s in test_spk if spk_reg[s] == r and s != exclude_spk for i in by_spk[s]]
    sup = rng.sample(cand, min(K, len(cand)))
    if method == 'hyper':
        with torch.no_grad():
            return hyper(senc([[torch.from_numpy(D.get_mel(i)) for i in sup]]))
    if method == 'table':
        if ck['table'] is None or r == args.get('holdout'): return None
        return ck['table'][R_ID[r]][None]
    if method in ('maml', 'clone') and (method == 'maml' or a.ft):
        phi = (ck['phi0'] if method == 'maml' else lora_init_vector(model)).clone().requires_grad_(True)
        steps = inner_steps if inner_steps is not None else (args.get('inner_steps', 3) if method == 'maml' else 20)
        items = [(D.ph[i], D.get_mel(i), D.ecapa[i]) for i in sup]
        x, xl, y, yl, e, H = collate(items, [D.get_mms(i) for i in sup])
        opt = torch.optim.Adam([phi], lr=1e-2) if method == 'clone' else None
        for _ in range(steps):
            Ls = model.losses(x, xl, y, yl, e, lora=phi[None].expand(len(items), -1), h0=H)
            g, = torch.autograd.grad(Ls['dur'] + Ls['prior'] + Ls['dec'], phi)
            if opt is None:
                phi = (phi - args.get('inner_lr', 0.05) * g).detach().requires_grad_(True)
            else:
                phi.grad = g; opt.step(); opt.zero_grad()
        return phi.detach()[None]
    return None

def synth(text, spk_ecapa, lora):
    ph = text_to_ids(text)
    x = torch.from_numpy(ph)[None]; xl = torch.tensor([len(ph)])
    t0 = time.time()
    mel, yl = model.synthesize(x, xl, torch.from_numpy(spk_ecapa)[None].float(), lora=lora, n_steps=a.nfe, temp=TEMP, t_start=T0 if a.nfe > 0 else 0.0)
    return denorm(mel[0, :, :int(yl[0])].numpy()), time.time() - t0

def voice(s, n_ref=3):
    refs = by_spk[s][:n_ref]
    return D.ecapa[refs].mean(0), refs

def build_items():
    """(voice speaker, target region, text, ground-truth index or -1)"""
    items = []
    for s in test_spk:
        _, refs = voice(s)
        own = [i for i in by_spk[s] if i not in refs]
        for r in REGIONS:
            if a.protocol == 'lodo' and r != (a.lodo_region or args.get('holdout')):
                continue
            if r == spk_reg[s]:
                pool = own
            else:
                pool = [i for t in test_spk if spk_reg[t] == r for i in by_spk[t]]
            for i in rng.sample(pool, min(a.n_text, len(pool))):
                items.append((s, r, D.meta.text.iloc[i], i if r == spk_reg[s] else -1))
    if a.limit: items = items[:a.limit]
    return items

if True:
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe = AutoFeatureExtractor.from_pretrained(ROOT + '/models/microsoft__wavlm-base-plus-sv')
    sv = WavLMForXVector.from_pretrained(ROOT + '/models/microsoft__wavlm-base-plus-sv').eval()
    def sv_emb(w):
        with torch.no_grad():
            return F.normalize(sv(**fe(w, sampling_rate=16000, return_tensors='pt')).embeddings, dim=-1)[0]
    ref_emb = {}
    def spk_ref_emb(s):
        if s not in ref_emb:
            _, refs = voice(s); ref_emb[s] = F.normalize(torch.stack([sv_emb(D.get_wav(i)) for i in refs]).mean(0), dim=0)
        return ref_emb[s]
    def clean_asr(t):
        return ' '.join(t.replace('<s>', '').split())

def cer(ref, hyp):
    r, h = list(ref.replace(' ', '')), list(hyp.replace(' ', ''))
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev = d.copy(); d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))

def mcd_dtw(m1, m2):
    """mel-cepstral distortion (dB) between two log-mel spectrograms, DTW aligned, c0 excluded"""
    from scipy.fft import dct
    c1 = dct(m1.T, type=2, norm='ortho', axis=1)[:, 1:25]; c2 = dct(m2.T, type=2, norm='ortho', axis=1)[:, 1:25]
    import librosa
    Dm = ((c1[:, None, :] - c2[None, :, :]) ** 2).sum(-1) ** 0.5
    _, wp = librosa.sequence.dtw(C=Dm)
    return float(10 / np.log(10) * np.sqrt(2) * np.mean([Dm[i, j] for i, j in wp]))

rows = []
t_start = time.time()
if a.protocol in ('seen', 'lodo'):
    items = build_items()
    cache = {}
    for s, r, text, gt in items:
        key = (r, s)
        if key not in cache:
            cache[key] = adapter(r, a.K, s, None if a.inner_steps < 0 else a.inner_steps)
        e, _ = voice(s)
        mel, dt = synth(text, e, cache[key])
        t1 = time.time(); w = audio_utils.vocode(mel); voc_s = time.time() - t1
        lp = dialect_lp(w)
        row = dict(system=a.name, spk=s, src=spk_reg[s], tgt=r, cross=int(r != spk_reg[s]), text=text,
                   p_tgt=float(np.exp(lp[R_ID[r]])), p_src=float(np.exp(lp[R_ID[spk_reg[s]]])), pred=REGIONS[int(lp.argmax())],
                   frames=mel.shape[1], synth_s=dt)
        row['correct'] = int(row['pred'] == r)
        if gt >= 0:
            row['mcd'] = mcd_dtw(mel, D.get_raw_mel(gt))
        row['voc_s'] = voc_s; row['dur_s'] = len(w) / 16000
        if a.audio:
            row['secs'] = float(sv_emb(w) @ spk_ref_emb(s)); hyp = clean_asr(asr_utils.transcribe(w))
            row['asr'] = hyp; row['cer'] = cer(text, hyp)
        rows.append(row)
elif a.protocol == 'kshot':
    # dialect controllability as a function of the support size K (no vocoder needed)
    items = build_items()
    for K in [1, 2, 5, 10, 20]:
        cache = {}
        for s, r, text, gt in items:
            if (r, s) not in cache: cache[(r, s)] = adapter(r, K, s)
            e, _ = voice(s); mel, dt = synth(text, e, cache[(r, s)])
            lp = dialect_lp(audio_utils.vocode(mel))
            rows.append(dict(system=a.name, K=K, spk=s, src=spk_reg[s], tgt=r, cross=int(r != spk_reg[s]),
                             p_tgt=float(np.exp(lp[R_ID[r]])), p_src=float(np.exp(lp[R_ID[spk_reg[s]]])), correct=int(lp.argmax() == R_ID[r])))
elif a.protocol == 'nfe':
    # decoding configurations: (temperature, Euler steps, shallow start); N=0 decodes the encoder prior
    items = build_items()[:48]
    for temp, nfe, t0 in [(0.0, 0, 0.0), (1.0, 3, 0.85), (1.0, 5, 0.5), (0.667, 10, 0.0), (1.0, 10, 0.0), (1.0, 20, 0.0)]:
        for s, r, text, gt in items:
            lv = adapter(r, a.K, s); e, _ = voice(s)
            ph = text_to_ids(text); x = torch.from_numpy(ph)[None]; t1 = time.time()
            mel, yl = model.synthesize(x, torch.tensor([len(ph)]), torch.from_numpy(e)[None].float(), lora=lv, n_steps=nfe, temp=temp, t_start=t0)
            dt = time.time() - t1; mel = denorm(mel[0, :, :int(yl[0])].numpy()); w = audio_utils.vocode(mel)
            lp = dialect_lp(w); hyp = clean_asr(asr_utils.transcribe(w))
            row = dict(system=a.name, temp=temp, nfe=nfe, t0=t0, p_tgt=float(np.exp(lp[R_ID[r]])), correct=int(lp.argmax() == R_ID[r]),
                       synth_s=dt, frames=mel.shape[1], rtf=dt / (mel.shape[1] * 256 / 16000), cer=cer(text, hyp))
            if gt >= 0: row['mcd'] = mcd_dtw(mel, D.get_raw_mel(gt))
            rows.append(row)
elif a.protocol == 'pipeline':
    # literary input -> {copy | bridge | gold spoken} -> TTS in the speaker's region of record
    B = pd.read_csv(a.bridge_file, keep_default_na=False)
    voices = test_spk
    for k, row in B.iterrows():
        s = voices[k % len(voices)]; r = row.region
        lv = adapter(r, a.K, s); e, _ = voice(s)
        for cond in ['copy', 'bridge', 'gold']:
            text = row[cond]
            if not text.strip(): continue
            mel, dt = synth(text, e, lv); w = audio_utils.vocode(mel); lp = dialect_lp(w)
            rec = dict(system=a.name, item=k, cond=cond, region=r, p_tgt=float(np.exp(lp[R_ID[r]])), correct=int(lp.argmax() == R_ID[r]))
            if a.audio:
                hyp = clean_asr(asr_utils.transcribe(w))
                rec['cer_vs_gold'] = cer(row['gold'], hyp); rec['cer_vs_input'] = cer(text, hyp)
            rows.append(rec)
if a.protocol == 'anchors':
    # natural recordings and vocoder resynthesis of the same test utterances (upper references for every metric)
    items = build_items()
    for s, r, text, gt in items:
        if gt < 0: continue
        for kind in ['natural', 'resynth']:
            w = D.get_wav(gt) if kind == 'natural' else audio_utils.vocode(D.get_raw_mel(gt))
            lp = dialect_lp(w); hyp = clean_asr(asr_utils.transcribe(w))
            row = dict(system=kind, spk=s, src=spk_reg[s], tgt=r, cross=0, p_tgt=float(np.exp(lp[R_ID[r]])),
                       correct=int(lp.argmax() == R_ID[r]), secs=float(sv_emb(w) @ spk_ref_emb(s)), cer=cer(text, hyp), asr=hyp)
            if kind == 'resynth': row['mcd'] = mcd_dtw(audio_utils.mel(w), D.get_raw_mel(gt))
            rows.append(row)
df = pd.DataFrame(rows)
out = fROOT + '/work/res/eval_{a.protocol}_{a.name}.csv'
df.to_csv(out, index=False)
print(a.name, a.protocol, len(df), 'rows', round(time.time() - t_start), 's')
if a.protocol in ('seen', 'lodo'):
    print(df.groupby('cross')[['correct', 'p_tgt', 'p_src'] + (['secs', 'cer'] if a.audio else [])].mean())
