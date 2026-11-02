"""Decode, trim, featurise (80-band log-mel), phonemise and embed (ECAPA) every utterance.
Usage: python prep.py <corpus>   corpus in {dialect, iv, slr}"""
from config import ROOT
import sys, glob, json, numpy as np, pandas as pd, pyarrow.parquet as pq, torch
torch.set_num_threads(1)
from audio_utils import decode, trim, mel, SR
from tamil_g2p import g2p, clean
from speechbrain.inference.speaker import EncoderClassifier
corpus = sys.argv[1]; D = ROOT + '/data'; OUT = ROOT + '/work/feat'
import os; os.makedirs(OUT, exist_ok=True)
ecapa = EncoderClassifier.from_hparams(source=ROOT + '/models/speechbrain__spkrec-ecapa-voxceleb',
                                       savedir=ROOT + '/models/ecapa_sb', run_opts={'device': 'cpu'})
REG = {'THA': 'Central', 'CH': 'Northern', 'S': 'Southern', 'KG': 'Western'}
meta, embs, lens, wlens = [], [], [], []
fm = open(f'{OUT}/{corpus}_mel.raw', 'wb'); fw = open(f'{OUT}/{corpus}_wav.raw', 'wb')
def handle(wav, rec, keep_wav):
    w = trim(wav)
    dur = len(w) / SR
    if dur < 1.0 or dur > 10.0: return
    ph = g2p(rec['text'])
    if len(ph) < 3: return
    m = mel(w)
    with torch.no_grad():
        e = ecapa.encode_batch(torch.from_numpy(w)[None]).squeeze().numpy()
    rec.update(dur=dur, n_frames=m.shape[1], phones=' '.join(ph))
    meta.append(rec); embs.append(e.astype(np.float32)); lens.append(m.shape[1])
    fm.write(np.ascontiguousarray(m.T.astype(np.float16)).tobytes())          # stored time-major (T, 80)
    if keep_wav:
        wi = (np.clip(w, -1, 1) * 32767).astype(np.int16); fw.write(wi.tobytes()); wlens.append(len(wi))
def batches(pattern):
    for f in sorted(glob.glob(pattern)):
        for b in pq.ParquetFile(f).iter_batches(batch_size=32):
            yield b.to_pandas()
if corpus == 'dialect':
    for t in batches(f'{D}/dialect/train-*.parquet'):
        for a, txt, d in zip(t.audio, t.text, t.dialect):
            p = a['path']; parts = p.split('_')
            handle(decode(a), dict(uid='DLC_' + p[:-4], corpus='dialect', spk='DLC_' + '_'.join(parts[:3]),
                   region=REG[parts[1]], gender=parts[2], district='', text=clean(txt), normalized=''), True)
elif corpus == 'iv':
    for t in batches(f'{D}/iv10k/*.parquet'):
        for i, r in t.iterrows():
            handle(decode(r.audio_filepath), dict(uid=f'IV_{len(meta)}', corpus='iv', spk='IV_' + r.speaker_id,
                   region='', gender=r.gender[:1].upper(), district=r.district, scenario=r.scenario,
                   text=clean(r.verbatim), normalized=clean(r.normalized)), True)
elif corpus == 'slr':
    for t in batches(f'{D}/slr65/*.parquet'):
        for i, r in t.iterrows():
            handle(decode(r.audio), dict(uid='SLR_' + r.file_id, corpus='slr', spk='SLR_' + r.file_id.split('_')[1],
                   region='Literary', gender=r.gender[:1].upper(), district='', text=clean(r.transcription), normalized=''), False)
    print('done', len(meta))
fm.close(); fw.close()
def raw_to_npy(raw, dtype, shape, out):
    src = np.memmap(raw, dtype=dtype, mode='r', shape=shape)
    dst = np.lib.format.open_memmap(out, mode='w+', dtype=dtype, shape=shape)
    for k in range(0, shape[0], 200000): dst[k:k + 200000] = src[k:k + 200000]
    dst.flush(); del dst, src; os.remove(raw)
lens = np.array(lens); off = np.concatenate([[0], np.cumsum(lens)])
raw_to_npy(f'{OUT}/{corpus}_mel.raw', np.float16, (int(off[-1]), 80), f'{OUT}/{corpus}_mel.npy')
np.save(f'{OUT}/{corpus}_off.npy', off); np.save(f'{OUT}/{corpus}_ecapa.npy', np.stack(embs))
if wlens:
    woff = np.concatenate([[0], np.cumsum(wlens)])
    raw_to_npy(f'{OUT}/{corpus}_wav.raw', np.int16, (int(woff[-1]),), f'{OUT}/{corpus}_wav.npy'); np.save(f'{OUT}/{corpus}_woff.npy', woff)
else:
    os.remove(f'{OUT}/{corpus}_wav.raw')
pd.DataFrame(meta).to_csv(f'{OUT}/{corpus}_meta.csv', index=False)
print(corpus, len(meta), lens.sum() / 62.5 / 3600, 'h')
