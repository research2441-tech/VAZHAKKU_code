"""Stage S0: speaker-conditioned backbone on general Tamil speech (no dialect conditioning).
python train_backbone.py --out ckpt/backbone_cfm.pt [--regression] --steps 12000"""
import argparse, time, json, numpy as np, torch, random
from data import Corpus, make_splits, collate, SYM
from model import Acoustic

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--steps', type=int, default=12000)
ap.add_argument('--bs', type=int, default=16); ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--regression', action='store_true'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--threads', type=int, default=1); ap.add_argument('--corpora', default='slr,iv')
ap.add_argument('--max_frames', type=int, default=500); ap.add_argument('--resume', default='')
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
splits = make_splits()
pool = []  # (corpus, idx, speaker-utterance list)
corp = {}
for name in a.corpora.split(','):
    c = Corpus(name); corp[name] = c
    sp = np.array(splits[name]); ok = (sp == 'train') & (c.meta.n_frames.values <= a.max_frames)
    by = {}
    for i in np.where(ok)[0]:
        by.setdefault(c.meta.spk.iloc[i], []).append(i)
    for s, ix in by.items():
        for i in ix:
            pool.append((name, i, ix))
print('training utterances', len(pool), flush=True)
model = Acoustic(len(SYM), regression=a.regression)
print('params', sum(p.numel() for p in model.parameters()) / 1e6, 'M', flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
start = 0; log = []
if a.resume:
    rk = torch.load(a.resume, weights_only=False); model.load_state_dict(rk['model']); log = rk['log']
    start = rk.get('step', log[-1]['step'] // 2000 * 2000 + 2000 if log else 0)
    if 'opt' in rk: opt.load_state_dict(rk['opt'])
    print('resumed at step', start, flush=True)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + start + 1) / 500) * max(0.05, 1 - (k + start) / a.steps))
t0 = time.time()
for step in range(start, a.steps):
    items, hs = [], []
    for name, i, ix in random.sample(pool, a.bs):
        c = corp[name]; j = random.choice(ix) if len(ix) > 1 else i
        items.append((c.ph[i], c.get_mel(i), c.ecapa[j]))   # reference embedding from another utterance
        hs.append(c.get_mms(i))
    x, xl, y, yl, e, H = collate(items, hs)
    L = model.losses(x, xl, y, yl, e, h0=H)
    loss = L['dur'] + L['prior'] + L['dec']
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 50 == 0:
        rec = dict(step=step, dur=L['dur'].item(), prior=L['prior'].item(), dec=L['dec'].item(), t=time.time() - t0)
        log.append(rec); print(json.dumps(rec), flush=True)
    if (step + 1) % 500 == 0 or step == a.steps - 1:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1, args=vars(a), log=log), a.out)
