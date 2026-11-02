"""Pre-compute wav2vec2 dialect features: every dialect-corpus utterance; up to two per IndicVoices speaker."""
import numpy as np, torch, pandas as pd
torch.set_num_threads(1)
from data import Corpus, FEAT
from w2v_feats import embed
c = Corpus('dialect')
X = np.stack([embed(c.get_wav(i)) for i in range(len(c.meta))]).astype(np.float16)
np.save(f'{FEAT}/dialect_w2v.npy', X); print('dialect', X.shape, flush=True)
iv = Corpus('iv'); idx = []
for s, g in iv.meta.groupby('spk'):
    idx += list(g.index[:2])
idx = np.array(sorted(idx))
X = np.stack([embed(iv.get_wav(i)) for i in idx]).astype(np.float16)
np.save(f'{FEAT}/iv_w2v.npy', X); np.save(f'{FEAT}/iv_w2v_idx.npy', idx); print('iv', X.shape, flush=True)
