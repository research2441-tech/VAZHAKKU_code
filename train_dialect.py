"""Stage S1: dialect stage, episodic training from the S0 backbone.

--method  clone  : no dialect conditioning (zero-shot speaker cloning baseline)
          table  : one learned LoRA vector per dialect (multi-dialect TTS with dialect identity)
          maml   : first-order MAML over a LoRA initialisation (inner loop on the support set)
          hyper  : hypernetwork-generated LoRA from the support set (VAZHAKKU)
--disent  none | club | grl   (speaker-dialect disentanglement on the speaker code)
--iv      none | hard | posterior  (IndicVoices speakers as additional, noisily-labelled tasks)
--holdout <region>  leave this dialect out entirely (few-shot generalisation protocol)
"""
import argparse, json, time, random, numpy as np, torch, torch.nn.functional as F
from data import Corpus, make_splits, collate, SYM, REGIONS, R_ID, district_prior, FEAT
from model import Acoustic, SupportEncoder, HyperLoRA, CLUB, GradReverse, lora_init_vector
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--backbone', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--method', default='hyper'); ap.add_argument('--disent', default='club'); ap.add_argument('--iv', default='posterior')
ap.add_argument('--holdout', default=''); ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--E', type=int, default=4); ap.add_argument('--Q', type=int, default=4); ap.add_argument('--K', type=int, default=5)
ap.add_argument('--lam_mi', type=float, default=0.1); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--lr_bb', type=float, default=1e-4)
ap.add_argument('--inner_steps', type=int, default=3); ap.add_argument('--inner_lr', type=float, default=0.05)
ap.add_argument('--threads', type=int, default=1); ap.add_argument('--lam_z', type=float, default=0.5); ap.add_argument('--max_frames', type=int, default=500)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

ck = torch.load(a.backbone)
regression = ck['args'].get('regression', False)
model = Acoustic(len(SYM), regression=regression); model.load_state_dict(ck['model'])
splits = make_splits()
D = Corpus('dialect'); spD = np.array(splits['dialect'])
regions = [r for r in REGIONS if r != a.holdout]

# ---- task pools: region -> speaker -> utterance indices   (corpus tag, idx)
pool = {r: {} for r in regions}; wts = {r: {} for r in regions}
for i in np.where((spD == 'train') & (D.meta.n_frames.values <= a.max_frames))[0]:
    r = D.meta.region.iloc[i]
    if r in pool:
        pool[r].setdefault(D.meta.spk.iloc[i], []).append(('dialect', i)); wts[r][D.meta.spk.iloc[i]] = 1.0
corp = {'dialect': D}
if a.iv != 'none':
    IV = Corpus('iv'); corp['iv'] = IV; spI = np.array(splits['iv'])
    if a.iv == 'posterior':
        P = pd.read_csv(f'{FEAT}/iv_region_posterior.csv').set_index('spk')
    for i in np.where((spI == 'train') & (IV.meta.n_frames.values <= a.max_frames))[0]:
        s = IV.meta.spk.iloc[i]
        if a.iv == 'hard':
            r = REGIONS[int(np.argmax(district_prior(IV.meta.district.iloc[i])))]; w = 1.0
        else:
            p = P.loc[s, REGIONS].values.astype(float); r = REGIONS[int(np.argmax(p))]; w = float(p.max())
        if r in pool:
            pool[r].setdefault(s, []).append(('iv', i)); wts[r][s] = w
for r in regions:
    print(r, 'speakers', len(pool[r]), 'utts', sum(len(v) for v in pool[r].values()), flush=True)

def other_ecapa(tag, i, spk_utts):
    j = random.choice(spk_utts)
    return corp[j[0]].ecapa[j[1]]

def sample_episode(r):
    spks = list(pool[r].keys()); w = np.array([wts[r][s] for s in spks]); w = w / w.sum()
    # support speakers drawn from the labelled dialect corpus when possible (clean labels)
    d_spk = [s for s in spks if s.startswith('DLC_')]
    sup_spk = random.sample(d_spk, min(3, len(d_spk)))
    support = []
    for k in range(a.K):
        s = sup_spk[k % len(sup_spk)]; tag, i = random.choice(pool[r][s]); support.append(corp[tag].get_mel(i))
    rest = [s for s in spks if s not in sup_spk]; wr = np.array([wts[r][s] for s in rest]); wr = wr / wr.sum()
    q_spk = np.random.choice(rest, a.Q, p=wr, replace=len(rest) < a.Q)
    query, sup_items = [], []
    for s in q_spk:
        tag, i = random.choice(pool[r][s]); c = corp[tag]
        query.append((c.ph[i], c.get_mel(i), other_ecapa(tag, i, pool[r][s]), c.get_mms(i)))
    # support items with phonemes (needed by the MAML inner loop)
    for s in sup_spk:
        tag, i = random.choice(pool[r][s]); c = corp[tag]
        sup_items.append((c.ph[i], c.get_mel(i), other_ecapa(tag, i, pool[r][s]), c.get_mms(i)))
    return support, query, sup_items

ldim = model.lora_dim(); print('lora dim', ldim, flush=True)
extra = []
senc = SupportEncoder(); hyper = None; table = None; phi0 = None
zcls = None
if a.method == 'hyper':
    hyper = HyperLoRA(model); zcls = torch.nn.Linear(128, len(REGIONS))
    extra += list(senc.parameters()) + list(hyper.parameters()) + list(zcls.parameters())
elif a.method == 'table':
    table = torch.nn.Parameter(torch.stack([lora_init_vector(model) for _ in REGIONS])); extra += [table]
elif a.method == 'maml':
    phi0 = torch.nn.Parameter(lora_init_vector(model)); extra += [phi0]
critic = None; cls_head = None
if a.disent == 'club':
    critic = CLUB(192, 128 if a.method == 'hyper' else 4); copt = torch.optim.Adam(critic.parameters(), lr=1e-3)
elif a.disent == 'grl':
    cls_head = torch.nn.Sequential(torch.nn.Linear(192, 128), torch.nn.SiLU(), torch.nn.Linear(128, len(REGIONS))); extra += list(cls_head.parameters())
opt = torch.optim.AdamW([{'params': model.parameters(), 'lr': a.lr_bb}, {'params': extra, 'lr': a.lr}], weight_decay=1e-2)
log = []; t0 = time.time()

def episode_lora(r, support, sup_items):
    """returns flat LoRA vector (1, ldim) for one episode plus the dialect code (for MI)"""
    if a.method == 'hyper':
        z = senc([[torch.from_numpy(m) for m in support]]); return hyper(z), z
    if a.method == 'table':
        return table[R_ID[r]][None], F.one_hot(torch.tensor([R_ID[r]]), 4).float()
    if a.method == 'maml':
        phi = phi0.detach().clone().requires_grad_(True)
        x, xl, y, yl, e, H = collate([it[:3] for it in sup_items], [it[3] for it in sup_items])
        for _ in range(a.inner_steps):
            Ls = model.losses(x, xl, y, yl, e, lora=phi[None].expand(len(sup_items), -1), h0=H)
            g, = torch.autograd.grad(Ls['dur'] + Ls['prior'] + Ls['dec'], phi)
            phi = (phi - a.inner_lr * g).detach().requires_grad_(True)
        return phi[None], F.one_hot(torch.tensor([R_ID[r]]), 4).float()
    return None, F.one_hot(torch.tensor([R_ID[r]]), 4).float()

for step in range(a.steps):
    ep_regions = [random.choice(regions) for _ in range(a.E)]
    items, loras, zs, labels, phis = [], [], [], [], []
    for r in ep_regions:
        support, query, sup_items = sample_episode(r)
        lv, z = episode_lora(r, support, sup_items)
        items += query; zs.append(z.expand(a.Q, -1)); labels += [R_ID[r]] * a.Q
        if lv is not None:
            loras.append(lv.expand(a.Q, -1)); phis.append(lv)
    x, xl, y, yl, e, H = collate([it[:3] for it in items], [it[3] for it in items])
    lora = torch.cat(loras) if loras else None
    L = model.losses(x, xl, y, yl, e, lora=lora, h0=H)
    loss = L['dur'] + L['prior'] + L['dec']
    zcat = torch.cat(zs); s = L['s']; mi = torch.tensor(0.); lz = torch.tensor(0.)
    if zcls is not None:
        # region-discriminative dialect code: one cross-entropy term per episode over training regions
        ze = zcat[::a.Q]; lz = F.cross_entropy(zcls(ze), torch.tensor([R_ID[r] for r in ep_regions])); loss = loss + a.lam_z * lz
    if a.disent == 'club':
        # 1) fit the variational critic on detached pairs  2) penalise the CLUB bound w.r.t. the speaker code only
        for _ in range(2):
            cl = -critic.loglik(s.detach(), zcat.detach()); copt.zero_grad(); cl.backward(); copt.step()
        mi = critic.bound(s, zcat.detach()); loss = loss + a.lam_mi * mi
    elif a.disent == 'grl':
        lg = cls_head(GradReverse.apply(s, 1.0)); mi = F.cross_entropy(lg, torch.tensor(labels)); loss = loss + a.lam_mi * mi
    opt.zero_grad()
    if a.method == 'maml':
        # first-order MAML: the query gradient w.r.t. the adapted vector is applied to the initialisation
        for p in phis: p.retain_grad()
    loss.backward()
    if a.method == 'maml':
        phi0.grad = torch.stack([p.grad.reshape(-1) for p in phis]).mean(0) if phis[0].grad is not None else None
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + extra, 1.0); opt.step()
    if step % 50 == 0:
        rec = dict(step=step, dur=L['dur'].item(), prior=L['prior'].item(), dec=L['dec'].item(), mi=float(mi), lz=float(lz), zstd=float(zcat[::a.Q].std(0).mean()), t=time.time() - t0)
        log.append(rec); print(json.dumps(rec), flush=True)
state = dict(model=model.state_dict(), senc=senc.state_dict(), args=vars(a), log=log, regression=regression,
             hyper=hyper.state_dict() if hyper else None, zcls=zcls.state_dict() if zcls is not None else None, table=table.detach() if table is not None else None,
             phi0=phi0.detach() if phi0 is not None else None)
torch.save(state, a.out)
print('saved', a.out, time.time() - t0)
