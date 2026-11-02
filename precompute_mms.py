"""Cache frozen MMS-TTS Tamil text-encoder states (fp16) for every utterance: they never change during training."""
import sys, numpy as np, torch
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
from data import Corpus, FEAT
from model import mms_text_encoder
name = sys.argv[1]; c = Corpus(name); enc = mms_text_encoder()
lens = np.array([len(p) for p in c.ph]); off = np.concatenate([[0], np.cumsum(lens)])
out = np.lib.format.open_memmap(f'{FEAT}/{name}_mms.npy', mode='w+', dtype=np.float16, shape=(int(off[-1]), 192))
order = np.argsort(lens); B = 16
with torch.no_grad():
    for k in range(0, len(order), B):
        ix = order[k:k + B]; T = int(lens[ix].max())
        x = torch.zeros(len(ix), T, dtype=torch.long); m = torch.zeros(len(ix), T)
        for b, i in enumerate(ix): x[b, :lens[i]] = torch.from_numpy(c.ph[i]); m[b, :lens[i]] = 1
        h = enc(input_ids=x, padding_mask=m[:, :, None], attention_mask=m.long()).last_hidden_state
        for b, i in enumerate(ix): out[off[i]:off[i + 1]] = h[b, :lens[i]].numpy().astype(np.float16)
out.flush(); np.save(f'{FEAT}/{name}_mms_off.npy', off); print(name, out.shape)
