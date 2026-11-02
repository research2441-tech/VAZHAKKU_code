"""Choose sampling temperature and Euler steps on VALIDATION speakers only (dialect corpus + SLR65),
using the backbone. Writes res/decoding_selection.json."""
from config import ROOT
import sys, json, numpy as np, torch, itertools
torch.set_num_threads(1)
from data import Corpus, make_splits, denorm
from model import Acoustic
from tokens import text_to_ids
import audio_utils, asr_utils
from evaluate_utils import cer, mcd_dtw
ck = torch.load(sys.argv[1], weights_only=False); m = Acoustic(59); m.load_state_dict(ck['model']); m.eval()
rows = []
for name, n in [('dialect', 16), ('slr', 8)]:
    c = Corpus(name); va = np.where(np.array(make_splits()[name]) == 'val')[0]
    idx = np.random.RandomState(7).choice(va, n, replace=False)
    for i in idx:
        x = torch.from_numpy(text_to_ids(c.meta.text.iloc[i]))[None]; e = torch.from_numpy(c.ecapa[i])[None].float()
        for temp, nfe, t0 in [(0.0, 0, 0.0), (1.0, 3, 0.85), (1.0, 4, 0.7), (1.0, 5, 0.5), (0.667, 10, 0.0), (1.0, 10, 0.0)]:
            torch.manual_seed(0)
            mel, yl = m.synthesize(x, torch.tensor([x.shape[1]]), e, n_steps=nfe, temp=temp, t_start=t0)
            mel = denorm(mel[0, :, :int(yl[0])].numpy()); w = audio_utils.vocode(mel)
            hyp = ' '.join(asr_utils.transcribe(w).replace('<s>', '').split())
            rows.append(dict(corpus=name, i=int(i), temp=temp, nfe=nfe, t0=t0, cer=cer(c.meta.text.iloc[i], hyp), mcd=mcd_dtw(mel, c.get_raw_mel(i))))
import pandas as pd
df = pd.DataFrame(rows); g = df.groupby(['temp', 'nfe', 't0'])[['cer', 'mcd']].mean().reset_index()
print(g.round(3).to_string())
best = g.sort_values(['cer', 'mcd']).iloc[0]
json.dump(dict(table=g.to_dict('records'), temp=float(best.temp), nfe=int(best.nfe), t0=float(best.t0)), open(ROOT + '/work/res/decoding_selection.json', 'w'), indent=1)
print('selected', best.to_dict())
