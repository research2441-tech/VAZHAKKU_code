"""Wall-clock cost of producing region adapters from K=5 support clips (one CPU thread), 10 repetitions."""
from config import ROOT
import time, json, numpy as np, torch
torch.set_num_threads(1)
from data import Corpus, make_splits, collate
from model import Acoustic, SupportEncoder, HyperLoRA, lora_init_vector
D = Corpus('dialect'); te = np.where(np.array(make_splits()['dialect']) == 'test')[0]
rng = np.random.RandomState(0)
ck = torch.load(ROOT + '/work/ckpt/full_s0.pt', weights_only=False)
m = Acoustic(59); m.load_state_dict(ck['model']); m.eval()
se = SupportEncoder(); se.load_state_dict(ck['senc']); se.eval(); h = HyperLoRA(m); h.load_state_dict(ck['hyper']); h.eval()
res = {'hyper_forward': [], 'finetune_20_adam': [], 'maml_3_sgd': []}
for rep in range(10):
    sup = rng.choice(te, 5, replace=False)
    t0 = time.time()
    with torch.no_grad(): lv = h(se([[torch.from_numpy(D.get_mel(i)) for i in sup]]))
    res['hyper_forward'].append(time.time() - t0)
    items = [(D.ph[i], D.get_mel(i), D.ecapa[i]) for i in sup]
    x, xl, y, yl, e, H = collate(items, [D.get_mms(i) for i in sup])
    for name, steps, use_adam in [('finetune_20_adam', 20, True), ('maml_3_sgd', 3, False)]:
        phi = lora_init_vector(m).clone().requires_grad_(True); opt = torch.optim.Adam([phi], lr=1e-2)
        t0 = time.time()
        for _ in range(steps):
            L = m.losses(x, xl, y, yl, e, lora=phi[None].expand(5, -1), h0=H)
            g, = torch.autograd.grad(L['dur'] + L['prior'] + L['dec'], phi)
            if use_adam: phi.grad = g; opt.step(); opt.zero_grad()
            else: phi = (phi - 0.05 * g).detach().requires_grad_(True)
        res[name].append(time.time() - t0)
out = {k: dict(mean_s=float(np.mean(v)), sd_s=float(np.std(v))) for k, v in res.items()}
json.dump(out, open(ROOT + '/work/res/adapt_cost.json', 'w'), indent=1); print(out)
