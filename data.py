"""Feature store, speaker-disjoint splits, batching and dialect episodes."""
from config import ROOT
import numpy as np, pandas as pd, torch, json, os
from tamil_g2p import PHONES

FEAT = ROOT + '/work/feat'
SYM = ['<pad>'] + PHONES
SYM_ID = {p: i for i, p in enumerate(SYM)}
REGIONS = ['Northern', 'Southern', 'Western', 'Central']
R_ID = {r: i for i, r in enumerate(REGIONS)}

# District -> macro-dialect prior. Administrative districts do not coincide with dialect boundaries,
# so districts lying on a boundary split their mass between the two neighbouring regions.
CORE = {
    'Northern': ['Chennai', 'Tiruvallur', 'Kanchipuram', 'Chengalpattu', 'Vellore', 'Ranipet', 'Tirupattur', 'Tiruvannamalai'],
    'Western': ['Coimbatore', 'Tiruppur', 'Erode', 'Salem', 'Namakkal', 'Nilgiris'],
    'Central': ['Thanjavur', 'Tiruvarur', 'Nagapattinam', 'Mayiladuthurai', 'Perambalur', 'Ariyalur'],
    'Southern': ['Madurai', 'Theni', 'Virudhunagar', 'Tirunelveli', 'Thoothukudi', 'Kanyakumari', 'Tenkasi', 'Ramanathapuram'],
}
BOUNDARY = {  # district: (primary, secondary)
    'Viluppuram': ('Northern', 'Central'), 'Kallakurichi': ('Northern', 'Central'), 'Cuddalore': ('Central', 'Northern'),
    'Dharmapuri': ('Western', 'Northern'), 'Krishnagiri': ('Western', 'Northern'), 'Karur': ('Western', 'Central'),
    'Tiruchirappalli': ('Central', 'Western'), 'Pudukkottai': ('Central', 'Southern'), 'Sivaganga': ('Southern', 'Central'),
    'Dindigul': ('Southern', 'Western'),
}

def district_prior(d, eps=0.04, w_primary=0.6):
    p = np.full(4, eps)
    for r, ds in CORE.items():
        if d in ds:
            p[R_ID[r]] = 1.0
    if d in BOUNDARY:
        a, b = BOUNDARY[d]; p[R_ID[a]] = w_primary; p[R_ID[b]] = 1 - w_primary
    return p / p.sum()

def hard_district(d):
    return int(np.argmax(district_prior(d)))


MEL_STATS = f'{FEAT}/mel_stats.npz'

def mel_stats():
    st = np.load(MEL_STATS)
    return st['mean'].astype(np.float32), st['std'].astype(np.float32)

def denorm(m):
    mu, sd = mel_stats(); return m * sd[:, None] + mu[:, None]


class Corpus:
    def __init__(self, name, normalize=True):
        self.name = name
        self.normalize = normalize and os.path.exists(MEL_STATS)
        if self.normalize:
            self.mu, self.sd = mel_stats()
        self.meta = pd.read_csv(f'{FEAT}/{name}_meta.csv', keep_default_na=False)
        self.mel = np.load(f'{FEAT}/{name}_mel.npy', mmap_mode='r')
        self.off = np.load(f'{FEAT}/{name}_off.npy')
        self.ecapa = np.load(f'{FEAT}/{name}_ecapa.npy')
        wp = f'{FEAT}/{name}_wav.npy'
        self.wav = np.load(wp, mmap_mode='r') if os.path.exists(wp) else None
        self.woff = np.load(f'{FEAT}/{name}_woff.npy') if self.wav is not None else None
        cache = f'{FEAT}/{name}_tok.npy'
        if os.path.exists(cache):
            self.ph = list(np.load(cache, allow_pickle=True))
        else:
            from tokens import text_to_ids
            self.ph = [text_to_ids(t) for t in self.meta.text]
            np.save(cache, np.array(self.ph, dtype=object), allow_pickle=True)

    def get_mms(self, i):
        if not hasattr(self, '_mms'):
            p = f'{FEAT}/{self.name}_mms.npy'
            self._mms = np.load(p, mmap_mode='r') if os.path.exists(p) else None
            self._mms_off = np.load(f'{FEAT}/{self.name}_mms_off.npy') if self._mms is not None else None
        if self._mms is None:
            return None
        return np.asarray(self._mms[self._mms_off[i]:self._mms_off[i + 1]], dtype=np.float32)

    def get_mel(self, i):
        m = self.get_raw_mel(i)
        return (m - self.mu[:, None]) / self.sd[:, None] if self.normalize else m

    def get_raw_mel(self, i):
        if self.mel.shape[0] == 80 and self.mel.shape[1] != 80:      # (80, T) layout
            return np.asarray(self.mel[:, self.off[i]:self.off[i + 1]], dtype=np.float32)
        return np.asarray(self.mel[self.off[i]:self.off[i + 1]], dtype=np.float32).T  # (T, 80) layout

    def get_wav(self, i):
        return np.asarray(self.wav[self.woff[i]:self.woff[i + 1]], dtype=np.float32) / 32767.0


def speaker_split(meta, group_col, seed=0, frac_test=0.2, frac_val=0.1):
    """Speaker-disjoint split stratified by group (dialect region)."""
    rng = np.random.RandomState(seed); split = {}
    for g, sub in meta.groupby(group_col):
        spk = sorted(sub.spk.unique()); rng.shuffle(spk)
        n = len(spk); nt = max(1, int(round(frac_test * n))); nv = max(1, int(round(frac_val * n)))
        for s in spk[:nt]: split[s] = 'test'
        for s in spk[nt:nt + nv]: split[s] = 'val'
        for s in spk[nt + nv:]: split[s] = 'train'
    return meta.spk.map(split).values


def make_splits():
    """Fixed speaker-disjoint splits written once per corpus and reused by every experiment."""
    out = {}
    for name in ['dialect', 'iv', 'slr']:
        path = f'{FEAT}/split_{name}.json'
        if os.path.exists(path):
            out[name] = json.load(open(path)); continue
        mp = f'{FEAT}/{name}_meta.csv'
        if not os.path.exists(mp):
            continue
        m = pd.read_csv(mp, keep_default_na=False)
        if name == 'dialect':
            sp = speaker_split(m, 'region', seed=0)
        elif name == 'iv':
            m['hreg'] = [REGIONS[hard_district(x)] for x in m.district]
            sp = speaker_split(m, 'hreg', seed=0)
        else:
            sp = speaker_split(m, 'gender', seed=0, frac_test=0.1, frac_val=0.05)
        out[name] = list(sp); json.dump(out[name], open(path, 'w'))
    return out


def collate(items, h0=None):
    """items: list of (token_ids, mel(80,T), ecapa) -> padded tensors; optional cached text-encoder states"""
    B = len(items)
    xl = torch.tensor([len(p) for p, _, _ in items]); yl = torch.tensor([m.shape[1] for _, m, _ in items])
    x = torch.zeros(B, int(xl.max()), dtype=torch.long); y = torch.zeros(B, 80, int(yl.max()))
    for b, (p, m, _) in enumerate(items):
        x[b, :len(p)] = torch.from_numpy(p); y[b, :, :m.shape[1]] = torch.from_numpy(m)
    e = torch.from_numpy(np.stack([it[2] for it in items])).float()
    if h0 is not None:
        H = torch.zeros(B, int(xl.max()), 192)
        for b, h in enumerate(h0): H[b, :h.shape[0]] = torch.from_numpy(h)
        return x, xl, y, yl, e, H
    return x, xl, y, yl, e


def spk_mean_ecapa(corp, idx_by_spk):
    return {s: corp.ecapa[ix].mean(0) for s, ix in idx_by_spk.items()}
