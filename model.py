"""VAZHAKKU acoustic model.

Phoneme encoder + duration predictor (Glow/Matcha style monotonic alignment), an optimal-transport
conditional flow-matching (OT-CFM) mel decoder, per-sample low-rank adapters that can be produced by a
hypernetwork from a dialect support set, and the speaker/dialect disentanglement critics.
"""
from config import ROOT
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from numba import njit

N_MEL = 80


# ----------------------------------------------------------------------------- alignment
@njit(cache=True)
def _mas(logp, T_x, T_y):
    # logp: (T_x, T_y) log-likelihood of frame j under phoneme i; returns hard monotonic path
    v = np.full((T_x, T_y), -1e9, dtype=np.float32)
    for j in range(T_y):
        for i in range(max(0, T_x - T_y + j), min(T_x, j + 1)):
            if j == 0:
                v[i, j] = logp[i, j] if i == 0 else -1e9
            else:
                stay = v[i, j - 1]
                move = v[i - 1, j - 1] if i > 0 else -1e9
                v[i, j] = logp[i, j] + (stay if stay > move else move)
    path = np.zeros((T_x, T_y), dtype=np.float32)
    i = T_x - 1
    for j in range(T_y - 1, -1, -1):
        path[i, j] = 1.0
        if i > 0 and (i == j or v[i - 1, j - 1] > v[i, j - 1]):
            i -= 1
    return path


def mono_align(mu, y, x_len, y_len):
    """mu: (B,80,Tx) phoneme means; y: (B,80,Ty) mel. Returns attn (B,Tx,Ty)."""
    with torch.no_grad():
        # Gaussian log-likelihood with unit variance (constant terms dropped)
        logp = -0.5 * ((mu * mu).sum(1)[:, :, None]
                       - 2 * torch.einsum('bct,bcs->bts', mu, y) + (y * y).sum(1)[:, None, :])
        logp = logp.cpu().numpy().astype(np.float32)
        out = np.zeros_like(logp)
        for b in range(mu.shape[0]):
            tx, ty = int(x_len[b]), int(y_len[b])
            out[b, :tx, :ty] = _mas(np.ascontiguousarray(logp[b, :tx, :ty]), tx, ty)
    return torch.from_numpy(out)


def seq_mask(lens, T=None):
    T = T or int(lens.max())
    return (torch.arange(T)[None, :] < lens[:, None]).float()


# ----------------------------------------------------------------------------- adapters
class LoRAConv1x1(nn.Module):
    """Pointwise projection W x + (alpha/r) * B (A x), with A,B supplied per sample.
    When no adapter is given the layer is the frozen/plain backbone projection."""
    def __init__(self, c_in, c_out, rank=4, alpha=8.0):
        super().__init__()
        self.lin = nn.Conv1d(c_in, c_out, 1)
        self.c_in, self.c_out, self.rank, self.scale = c_in, c_out, rank, alpha / rank

    def n_lora(self):
        return self.rank * (self.c_in + self.c_out)

    def forward(self, x, lora=None):
        y = self.lin(x)
        if lora is not None:
            A = lora[:, : self.rank * self.c_in].view(-1, self.rank, self.c_in)
            B = lora[:, self.rank * self.c_in:].view(-1, self.c_out, self.rank)
            y = y + self.scale * torch.einsum('bor,brt->bot', B, torch.einsum('bri,bit->brt', A, x))
        return y


class AdaptedNet(nn.Module):
    """Mixin: registry of LoRA sites; a flat per-sample vector is split across sites."""
    def lora_sites(self):
        return [m for m in self.modules() if isinstance(m, LoRAConv1x1)]

    def lora_dim(self):
        return sum(m.n_lora() for m in self.lora_sites())

    def split_lora(self, flat):
        if flat is None:
            return {}
        out, k = {}, 0
        for m in self.lora_sites():
            n = m.n_lora(); out[id(m)] = flat[:, k:k + n]; k += n
        return out


# ----------------------------------------------------------------------------- encoder
class ConvBlock(nn.Module):
    def __init__(self, c, k=5, p=0.1):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=k // 2); self.norm = nn.GroupNorm(1, c); self.drop = nn.Dropout(p)

    def forward(self, x, m):
        return (x + self.drop(F.gelu(self.norm(self.conv(x * m))))) * m


class FFTLayer(nn.Module):
    def __init__(self, c, heads=2, ff=512, p=0.1, rank=4):
        super().__init__()
        self.att = nn.MultiheadAttention(c, heads, dropout=p, batch_first=True)
        self.n1 = nn.LayerNorm(c); self.n2 = nn.LayerNorm(c)
        self.ff1 = LoRAConv1x1(c, ff, rank); self.ff2 = LoRAConv1x1(ff, c, rank); self.drop = nn.Dropout(p)

    def forward(self, x, m, L):
        h = x.transpose(1, 2)
        a, _ = self.att(h, h, h, key_padding_mask=(m[:, 0] == 0))
        h = self.n1(h + self.drop(a)).transpose(1, 2)
        f = self.ff2(F.gelu(self.ff1(h * m, L.get(id(self.ff1)))), L.get(id(self.ff2)))
        return self.n2((h + self.drop(f)).transpose(1, 2)).transpose(1, 2) * m


class DurationPredictor(nn.Module):
    def __init__(self, c, h=192, rank=4):
        super().__init__()
        self.c1 = nn.Conv1d(c, h, 3, padding=1); self.n1 = nn.GroupNorm(1, h)
        self.c2 = nn.Conv1d(h, h, 3, padding=1); self.n2 = nn.GroupNorm(1, h)
        self.out = LoRAConv1x1(h, 1, rank); self.drop = nn.Dropout(0.1)

    def forward(self, x, m, L):
        x = self.drop(F.gelu(self.n1(self.c1(x * m))))
        x = self.drop(F.gelu(self.n2(self.c2(x * m))))
        return self.out(x * m, L.get(id(self.out))) * m  # log-duration


_MMS = None
def mms_text_encoder():
    """Frozen pretrained Tamil text encoder (MMS-TTS Tamil, VITS text encoder, 6.3 M parameters)."""
    global _MMS
    if _MMS is None:
        from transformers import VitsModel
        _MMS = VitsModel.from_pretrained(ROOT + '/models/facebook__mms-tts-tam').text_encoder.eval()
        for p in _MMS.parameters():
            p.requires_grad_(False)
    return _MMS


class TextEncoder(AdaptedNet):
    def __init__(self, n_sym, c=192, n_conv=1, n_fft=3, c_spk=192, rank=4):
        super().__init__()
        self.inp = nn.Conv1d(192, c, 1)
        self.convs = nn.ModuleList([ConvBlock(c) for _ in range(n_conv)])
        self.layers = nn.ModuleList([FFTLayer(c, rank=rank) for _ in range(n_fft)])
        self.spk = nn.Linear(c_spk, c)
        self.proj = LoRAConv1x1(c, N_MEL, rank)
        self.dur = DurationPredictor(c, rank=rank)

    def forward(self, x, x_len, s, L, h0=None):
        m = seq_mask(x_len, x.shape[1])[:, None]
        if h0 is None:
            with torch.no_grad():
                h0 = mms_text_encoder()(input_ids=x, padding_mask=m.transpose(1, 2), attention_mask=m[:, 0].long()).last_hidden_state
        h = self.inp(h0.transpose(1, 2)) * m
        for cb in self.convs:
            h = cb(h, m)
        h = (h + self.spk(s)[:, :, None]) * m
        for ly in self.layers:
            h = ly(h, m, L)
        mu = self.proj(h, L.get(id(self.proj))) * m
        logw = self.dur(h.detach(), m, L)
        return mu, logw, m


# ----------------------------------------------------------------------------- CFM decoder
class TimeEmb(nn.Module):
    def __init__(self, d):
        super().__init__(); self.d = d; self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half) / half)
        a = 1000 * t[:, None] * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class ResBlock(nn.Module):
    def __init__(self, c, c_cond, k=5, dil=1, rank=4):
        super().__init__()
        self.conv = nn.Conv1d(c, 2 * c, k, padding=dil * (k // 2), dilation=dil)
        self.film = nn.Linear(c_cond, 2 * c)
        self.pw = LoRAConv1x1(c, c, rank)
        self.norm = nn.GroupNorm(1, c)

    def forward(self, x, m, g, L):
        h = self.conv(self.norm(x) * m)
        a, b = h.chunk(2, 1)
        h = torch.tanh(a) * torch.sigmoid(b)
        sc, sh = self.film(g).chunk(2, -1)
        h = h * (1 + sc[:, :, None]) + sh[:, :, None]
        return (x + self.pw(h, L.get(id(self.pw)))) * m


class FlowDecoder(AdaptedNet):
    """Estimates the OT-CFM vector field v(t, x_t | mu, s)."""
    def __init__(self, c=160, n_blocks=8, c_spk=192, rank=4, regression=False):
        super().__init__()
        self.regression = regression
        c_in = N_MEL if regression else 2 * N_MEL
        self.inp = nn.Conv1d(c_in, c, 1)
        self.temb = TimeEmb(128); self.cond = nn.Linear(128 + c_spk, 256)
        self.blocks = nn.ModuleList([ResBlock(c, 256, dil=2 ** (i % 4), rank=rank) for i in range(n_blocks)])
        self.out = nn.Conv1d(c, N_MEL, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x_t, mu, m, t, s, L):
        h = self.inp((mu if self.regression else torch.cat([x_t, mu], 1)) * m)
        g = F.silu(self.cond(torch.cat([self.temb(t), s], -1)))
        for b in self.blocks:
            h = b(h, m, g, L)
        return self.out(h) * m


class Acoustic(nn.Module):
    """Backbone = encoder + decoder. `regression=True` gives the FastSpeech2-style deterministic decoder."""
    def __init__(self, n_sym, rank=4, regression=False, c_ecapa=192):
        super().__init__()
        self.spk_proj = nn.Sequential(nn.Linear(c_ecapa, 192), nn.SiLU(), nn.Linear(192, 192))
        self.enc = TextEncoder(n_sym, rank=rank)
        self.dec = FlowDecoder(rank=rank, regression=regression)
        self.regression = regression
        self.sigma_min = 1e-4

    def lora_dim(self):
        return self.enc.lora_dim() + self.dec.lora_dim()

    def split(self, flat):
        if flat is None:
            return {}
        ne = self.enc.lora_dim()
        L = self.enc.split_lora(flat[:, :ne]); L.update(self.dec.split_lora(flat[:, ne:]))
        return L

    def speaker(self, ecapa):
        return self.spk_proj(F.normalize(ecapa, dim=-1))

    def losses(self, x, x_len, y, y_len, ecapa, lora=None, crop=96, s=None, h0=None):
        """Returns dict of losses; y is (B,80,T)."""
        L = self.split(lora)
        if s is None:
            s = self.speaker(ecapa)
        mu_x, logw, xm = self.enc(x, x_len, s, L, h0)
        ym = seq_mask(y_len, y.shape[2])[:, None]
        attn = mono_align(mu_x.detach(), y, x_len, y_len)                      # (B,Tx,Ty)
        dur_tgt = torch.log(1e-8 + attn.sum(2)) * xm[:, 0]
        l_dur = ((logw[:, 0] - dur_tgt) ** 2).sum() / x_len.sum()
        mu_y = torch.einsum('bct,bts->bcs', mu_x, attn)
        l_prior = (0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * ym).sum() / (ym.sum() * N_MEL)
        # random crops for the decoder
        B = y.shape[0]; T = min(crop, int(y_len.min()))
        st = [np.random.randint(0, int(y_len[b]) - T + 1) for b in range(B)]
        yc = torch.stack([y[b, :, st[b]:st[b] + T] for b in range(B)])
        mc = torch.stack([mu_y[b, :, st[b]:st[b] + T] for b in range(B)])
        cm = torch.ones(B, 1, T)
        if self.regression:
            pred = self.dec(None, mc, cm, torch.zeros(B), s, L)
            l_dec = F.l1_loss(pred + mc, yc)
        else:
            t = torch.rand(B)
            x0 = mc.detach() + torch.randn_like(yc)   # source distribution centred on the encoder prior
            xt = (1 - (1 - self.sigma_min) * t[:, None, None]) * x0 + t[:, None, None] * yc
            u = yc - (1 - self.sigma_min) * x0
            v = self.dec(xt, mc, cm, t, s, L)
            l_dec = F.mse_loss(v, u)
        return dict(dur=l_dur, prior=l_prior, dec=l_dec, s=s)

    @torch.no_grad()
    def synthesize(self, x, x_len, ecapa, lora=None, n_steps=10, temp=0.667, length_scale=1.0, s=None, t_start=0.0):
        L = self.split(lora)
        if s is None:
            s = self.speaker(ecapa)
        mu_x, logw, xm = self.enc(x, x_len, s, L)
        w = torch.ceil(torch.exp(logw) * xm * length_scale).clamp(min=0)[:, 0]
        y_len = w.sum(1).clamp(min=8).long()
        Ty = int(y_len.max())
        ym = seq_mask(y_len, Ty)[:, None]
        cum = torch.cumsum(w, 1)
        frames = torch.arange(Ty)[None, None, :]
        attn = ((frames < cum[:, :, None]) & (frames >= (cum - w)[:, :, None])).float()
        mu_y = torch.einsum('bct,bts->bcs', mu_x, attn)
        if self.regression:
            return (self.dec(None, mu_y, ym, torch.zeros(x.shape[0]), s, L) + mu_y) * ym, y_len
        if n_steps == 0:                              # encoder prior only (no flow refinement)
            return mu_y * ym, y_len
        z0 = mu_y + torch.randn_like(mu_y) * temp     # prior-centred source
        # shallow start: interpolant at t_start with the prior standing in for the unknown data
        z = (1 - (1 - self.sigma_min) * t_start) * z0 + t_start * mu_y
        dt = (1.0 - t_start) / n_steps
        for k in range(n_steps):
            t = torch.full((x.shape[0],), t_start + k * dt)
            z = z + dt * self.dec(z, mu_y, ym, t, s, L)
        return z * ym, y_len


# ----------------------------------------------------------------------------- dialect side
class SupportEncoder(nn.Module):
    """Maps a set of K support mels (from several speakers of one dialect) to a dialect code z."""
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(N_MEL, 128, 5, padding=2), nn.GELU(), nn.GroupNorm(1, 128),
                                 nn.Conv1d(128, 128, 5, padding=2, stride=2), nn.GELU(), nn.GroupNorm(1, 128),
                                 nn.Conv1d(128, 128, 5, padding=2, stride=2), nn.GELU())
        self.utt = nn.Linear(256, d); self.set = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d, elementwise_affine=False)    # unit-scale dialect code

    def utterance(self, mels):
        out = []
        for m in mels:                                  # list of (80,T)
            m = m - m.mean(1, keepdim=True)              # per-utterance mean removal (channel / timbre tilt)
            h = self.net(m[None])[0]
            out.append(self.utt(torch.cat([h.mean(1), h.std(1)])))
        return torch.stack(out)

    def forward(self, support_sets):
        """support_sets: list (episodes) of lists of mels -> (E, d)"""
        return self.norm(torch.stack([self.set(self.utterance(S).mean(0)) for S in support_sets]))


class HyperLoRA(nn.Module):
    """z -> flat LoRA vector. Heads have no bias, so every adapter value is a function of the dialect code and the
    network cannot fall back to one region-independent adapter. B-heads are zero-initialised, so the adapted model
    starts exactly at the backbone."""
    def __init__(self, model, d=128, h=256):
        super().__init__()
        self.sites = [(m.rank * m.c_in, m.rank * m.c_out, m.c_in) for m in model.enc.lora_sites() + model.dec.lora_sites()]
        self.trunk = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.heads_A = nn.ModuleList([nn.Linear(h, a, bias=False) for a, _, _ in self.sites])
        self.heads_B = nn.ModuleList([nn.Linear(h, b, bias=False) for _, b, _ in self.sites])
        for hb in self.heads_B:
            nn.init.zeros_(hb.weight)
        for ha, (_, _, cin) in zip(self.heads_A, self.sites):
            nn.init.normal_(ha.weight, std=1.0 / math.sqrt(0.25 * h * cin))

    def forward(self, z):
        g = self.trunk(z)
        return torch.cat([torch.cat([ha(g), hb(g)], -1) for ha, hb in zip(self.heads_A, self.heads_B)], -1)


def lora_init_vector(model, std=None):
    """Plain LoRA initialisation (A Gaussian, B zero) as one flat vector."""
    parts = []
    for m in model.enc.lora_sites() + model.dec.lora_sites():
        parts += [torch.randn(m.rank * m.c_in) / math.sqrt(m.c_in), torch.zeros(m.rank * m.c_out)]
    return torch.cat(parts)


class CLUB(nn.Module):
    """Contrastive log-ratio upper bound on I(s; z) with a Gaussian variational q(z|s)."""
    def __init__(self, ds=192, dz=128, h=256):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(ds, h), nn.SiLU(), nn.Linear(h, dz))
        self.lv = nn.Sequential(nn.Linear(ds, h), nn.SiLU(), nn.Linear(h, dz), nn.Tanh())

    def loglik(self, s, z):
        mu, lv = self.mu(s), self.lv(s)
        return (-(mu - z) ** 2 / lv.exp() - lv).sum(-1).mean()

    def bound(self, s, z):
        mu, lv = self.mu(s), self.lv(s)
        pos = -((mu - z) ** 2) / lv.exp()
        neg = -((mu[:, None, :] - z[None, :, :]) ** 2).mean(1) / lv.exp()
        return (pos.sum(-1) - neg.sum(-1)).mean() / 2.0


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None
