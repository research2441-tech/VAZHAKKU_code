"""Per-channel mel mean / std over TRAIN-split utterances of all corpora (written once)."""
import numpy as np
from data import Corpus, make_splits, FEAT
sp = make_splits(); S = np.zeros(80); SS = np.zeros(80); N = 0
import sys
for name in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['iv', 'slr']):
    c = Corpus(name, normalize=False); ok = np.array(sp[name]) == 'train'
    for i in np.where(ok)[0][::3]:
        m = c.get_raw_mel(i).astype(np.float64); S += m.sum(1); SS += (m ** 2).sum(1); N += m.shape[1]
mu = S / N; sd = np.sqrt(SS / N - mu ** 2)
np.savez(f'{FEAT}/mel_stats.npz', mean=mu, std=sd); print(mu.mean(), sd.mean(), N)
