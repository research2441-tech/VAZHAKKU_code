import sys, numpy as np, torch
torch.set_num_threads(1)
from data import Corpus, make_splits, SYM, SYM_ID, denorm
from model import Acoustic, seq_mask
from tamil_g2p import g2p
from tokens import text_to_ids
import audio_utils, asr_utils
ck = torch.load(sys.argv[1], weights_only=False); m = Acoustic(len(SYM)); m.load_state_dict(ck['model']); m.eval()
corpus = sys.argv[2]; c = Corpus(corpus); te = np.where(np.array(make_splits()[corpus]) == 'test')[0]
idx = np.random.RandomState(1).choice(te, 8, replace=False)
def cer(r, h):
    r, h = list(r.replace(' ', '')), list(h.replace(' ', '')); d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p = d.copy(); d[0] = i
        for j in range(1, len(h) + 1): d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
asr = lambda w: ' '.join(asr_utils.transcribe(w).replace('<s>', '').split())
res = {k: [] for k in ['mu_pred', 'dec_t0.667', 'dec_t0.3', 'mu_gtdur', 'dec_gtdur']}
lr = []
for i in idx:
    text = c.meta.text.iloc[i]; ph = torch.from_numpy(text_to_ids(text))[None]; xl = torch.tensor([ph.shape[1]])
    e = torch.from_numpy(c.ecapa[i])[None].float(); s = m.speaker(e)
    with torch.no_grad():
        mu_x, logw, xm = m.enc(ph, xl, s, {})
        # predicted durations
        w = torch.ceil(torch.exp(logw) * xm)[:, 0]
        # ground-truth durations via MAS
        y = torch.from_numpy(c.get_mel(i))[None]; yl = torch.tensor([y.shape[2]])
        from model import mono_align
        A = mono_align(mu_x, y, xl, yl); w_gt = A.sum(2)
    lr.append(float(w.sum() / w_gt.sum()))
    for name, dur, mode, temp in [('mu_pred', w, 'mu', 0), ('dec_t0.667', w, 'dec', 0.667), ('dec_t0.3', w, 'dec', 0.3), ('mu_gtdur', w_gt, 'mu', 0), ('dec_gtdur', w_gt, 'dec', 0.667)]:
        Ty = int(dur.sum()); cum = torch.cumsum(dur, 1); fr = torch.arange(Ty)[None, None]
        attn = ((fr < cum[:, :, None]) & (fr >= (cum - dur)[:, :, None])).float()
        mu_y = torch.einsum('bct,bts->bcs', mu_x, attn); ym = torch.ones(1, 1, Ty)
        if mode == 'mu': out = mu_y
        else:
            z = mu_y + torch.randn_like(mu_y) * temp
            with torch.no_grad():
                for k in range(10):
                    z = z + 0.1 * m.dec(z, mu_y, ym, torch.full((1,), k * 0.1), s, {})
            out = z
        wav = audio_utils.vocode(denorm(out[0].numpy())); res[name].append(cer(text, asr(wav)))
print('pred/gt duration ratio', np.mean(lr))
for k, v in res.items(): print(k, round(np.mean(v), 3))
