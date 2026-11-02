"""Quick quality check of a backbone: synthesize held-out sentences in their own speakers' voices."""
from config import ROOT
import sys, numpy as np, torch, json
torch.set_num_threads(1)
from data import Corpus, make_splits, SYM, SYM_ID, denorm
from model import Acoustic
from tamil_g2p import g2p
from tokens import text_to_ids
import audio_utils, asr_utils
ck = torch.load(sys.argv[1], weights_only=False); reg = ck['args'].get('regression', False)
m = Acoustic(len(SYM), regression=reg); m.load_state_dict(ck['model']); m.eval()
corpus = sys.argv[2] if len(sys.argv) > 2 else 'iv'
c = Corpus(corpus); sp = np.array(make_splits()[corpus]); te = np.where(sp == 'test')[0]
rng = np.random.RandomState(0); idx = rng.choice(te, 12, replace=False)
def cer(r, h):
    r, h = list(r.replace(' ', '')), list(h.replace(' ', '')); d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p = d.copy(); d[0] = i
        for j in range(1, len(h) + 1): d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
rows = []
for i in idx:
    ph = text_to_ids(c.meta.text.iloc[i]); e = c.ecapa[i]
    mel, yl = m.synthesize(torch.from_numpy(ph)[None], torch.tensor([len(ph)]), torch.from_numpy(e)[None].float(), n_steps=10)
    mel = denorm(mel[0, :, :int(yl[0])].numpy()); w = audio_utils.vocode(mel)
    hs = ' '.join(asr_utils.transcribe(w).replace('<s>', '').split()); hn = ' '.join(asr_utils.transcribe(c.get_wav(i) if c.wav is not None else audio_utils.vocode(c.get_raw_mel(i))).replace('<s>', '').split())
    rows.append(dict(len_ratio=mel.shape[1] / c.meta.n_frames.iloc[i], cer_syn=cer(c.meta.text.iloc[i], hs), cer_nat=cer(c.meta.text.iloc[i], hn)))
import pandas as pd
df = pd.DataFrame(rows); print(df.round(3).to_string()); print(df.mean().round(3).to_dict())
import soundfile as sf; sf.write(ROOT + '/work/sanity_last.wav', w, 16000)
