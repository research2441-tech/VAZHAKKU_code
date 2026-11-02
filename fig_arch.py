"""Fig. 1: VAZHAKKU architecture (grayscale). Solid arrows: forward data flow. Dashed arrow: reverse-mode
gradient of the CLUB penalty, which reaches only the speaker projector. All connectors are orthogonal and
none cross: the speaker code enters encoder/decoder from above, LoRA updates from below."""
from config import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif'})
FS = 6.0
fig, ax = plt.subplots(figsize=(7.2, 4.15)); ax.set_xlim(0, 100); ax.set_ylim(-0.5, 58.5); ax.axis('off')
B = {}
def box(key, x0, x1, y0, y1, text, fill='white', ls='-', fs=FS):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.0,rounding_size=0.8', fc=fill, ec='black', lw=0.8, ls=ls))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center', fontsize=fs, linespacing=1.18)
    B[key] = (x0, x1, y0, y1)
def P(key, side, f=0.5):
    x0, x1, y0, y1 = B[key]
    return {'l': (x0, y0 + (y1 - y0) * f), 'r': (x1, y0 + (y1 - y0) * f),
            't': (x0 + (x1 - x0) * f, y1), 'b': (x0 + (x1 - x0) * f, y0)}[side]
def seg(p0, p1, ls='-'):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='black', lw=0.8, ls=ls, solid_capstyle='projecting')
def arr(p0, p1, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=7, lw=0.8, color='black',
                                 linestyle=ls, shrinkA=0, shrinkB=0.2))
def path(pts, ls='-'):
    for k in range(len(pts) - 2): seg(pts[k], pts[k + 1], ls)
    arr(pts[-2], pts[-1], ls)
def lab(x, y, t, ha='center', fs=5.8):
    ax.text(x, y, t, ha=ha, va='center', fontsize=fs, bbox=dict(fc='white', ec='none', pad=0.3))

G1, G2, G3 = '#efefef', '#d6d6d6', '#bdbdbd'
# ---------------- text / acoustic lane
yt0, yt1 = 42.5, 52.0
box('txt', 3, 14, yt0, yt1, 'Standard\n(literary)\ntext $\\mathbf{x}$')
box('br', 16.5, 30, yt0, yt1, 'Register bridge\nlattice $\\mathcal{C}(w_i)$\n+ context scorer', G1)
box('g2p', 32.5, 43.5, yt0, yt1, 'Spoken form\n$\\hat{\\mathbf{x}}$ →\ncharacters $\\mathbf{c}$')
box('enc', 46, 58, yt0, yt1, 'Text encoder\nfrozen $E_{\\mathrm{pre}}$ +\nadapted layers\n[LoRA sites]', G2)
box('exp', 60.5, 69.5, yt0, yt1, 'Duration\nexpansion\n$\\boldsymbol{\\mu}_y$')
box('dec', 72, 85.5, yt0, yt1, 'OT-CFM\ndecoder $v_\\theta$\n[LoRA sites]', G2)
box('voc', 88, 99, yt0, yt1, 'Mel $\\hat{\\mathbf{Y}}$ →\nHiFi-GAN →\nwaveform')
for a, b in [('txt', 'br'), ('br', 'g2p'), ('g2p', 'enc'), ('enc', 'exp'), ('exp', 'dec'), ('dec', 'voc')]:
    arr(P(a, 'r'), P(b, 'l'))
# ---------------- speaker lane
ys0, ys1 = 28.5, 36.5
box('ref', 3, 14, ys0, ys1, 'Reference\nvoice\n(3 utterances)')
box('ecapa', 16.5, 28.5, ys0, ys1, 'ECAPA-TDNN\n(frozen)\n$\\mathbf{e}_u$', G1)
box('proj', 31.5, 45, ys0, ys1, 'Speaker\nprojector $P_\\phi$\n→ $\\tilde{\\mathbf{s}}_u$', G2)
arr(P('ref', 'r'), P('ecapa', 'l')); arr(P('ecapa', 'r'), P('proj', 'l'))
# speaker code: up, left round the lanes, over the top, down into encoder and decoder
xs = 38.25; ytop = 56.0
seg((xs, ys1), (xs, 39.5)); seg((xs, 39.5), (1.0, 39.5)); seg((1.0, 39.5), (1.0, ytop)); seg((1.0, ytop), (78.75, ytop))
arr((52.0, ytop), P('enc', 't', 0.5)); arr((78.75, ytop), P('dec', 't', 0.5))
ax.plot([52.0], [ytop], 'o', ms=2.2, color='black')
lab(26.0, ytop, '$\\tilde{\\mathbf{s}}_u$  (additive in encoder, FiLM in decoder)')
# ---------------- CLUB critic between lanes
yc0, yc1 = 18.5, 25.0
box('club', 31.5, 45, yc0, yc1, 'CLUB critic\n$q_\\psi(\\mathbf{z}\\mid\\tilde{\\mathbf{s}})$', G3)
arr(P('proj', 'b', 0.3), P('club', 't', 0.3))                       # forward: s~ into the critic
arr(P('club', 't', 0.72), P('proj', 'b', 0.72), ls='--')            # reverse-mode gradient to P_phi only
lab(48.6, 26.8, '$-\\lambda_{\\mathrm{MI}}\\nabla_{\\phi}\\hat{I}$', fs=5.8)
# ---------------- dialect lane
yd0, yd1 = 6.5, 14.5
box('sup', 3, 14, yd0, yd1, 'Support set\n$\\mathcal{S}_r$ ($K$ utts,\n≥3 speakers)')
box('senc', 16.5, 28.5, yd0, yd1, 'Support\nencoder (CMN)\n→ $\\mathbf{z}_r$', G1)
box('hyp', 31.5, 45, yd0, yd1, 'Hypernetwork\n$H_\\omega(\\mathbf{z}_r)$', G2)
box('lora', 48.5, 63, yd0, yd1, '$\\Delta\\mathbf{W}_\\ell=\\frac{\\alpha}{\\rho}\\mathbf{B}_\\ell\\mathbf{A}_\\ell$\nfor every\nLoRA site', G1)
arr(P('sup', 'r'), P('senc', 'l')); arr(P('senc', 'r'), P('hyp', 'l')); arr(P('hyp', 'r'), P('lora', 'l'))
path([P('senc', 't', 0.55), (23.1, 21.75), P('club', 'l')])            # stop-gradient dialect code into critic
lab(23.1, 18.2, 'sg$(\\mathbf{z}_r)$')
arr((56.5, yd1), P('enc', 'b', 0.875))                                  # LoRA into encoder (from below)
path([P('lora', 'r'), (78.75, 10.5), P('dec', 'b', 0.5)])            # LoRA into decoder (from below)
# ---------------- noisy dialect prior
box('dist', 3, 62, 0.0, 4.6, 'District prior $\\pi(r\\mid\\mathrm{district})$ × acoustic dialect classifier → posterior $q(r\\mid u)$:\nepisode sampling weights for meta-training; region code $r$ of the register bridge', ls=':', fs=5.6)
arr((8.5, 4.6), P('sup', 'b', 0.5))
# ---------------- legend (own frame, clear of every element)
lx, ly = 67.5, 0.4
ax.add_patch(Rectangle((lx, ly), 31.5, 7.2, fc='white', ec='black', lw=0.5))
ax.add_patch(FancyArrowPatch((lx + 1.5, ly + 5.0), (lx + 8.0, ly + 5.0), arrowstyle='-|>', mutation_scale=7, lw=0.8, color='black'))
ax.text(lx + 9.5, ly + 5.0, 'forward data flow', fontsize=5.8, va='center')
ax.add_patch(FancyArrowPatch((lx + 1.5, ly + 2.0), (lx + 8.0, ly + 2.0), arrowstyle='-|>', mutation_scale=7, lw=0.8, color='black', linestyle='--'))
ax.text(lx + 9.5, ly + 2.0, 'reverse-mode gradient', fontsize=5.8, va='center')
fig.savefig(ROOT + '/work/figs/fig1_architecture.png', dpi=450, bbox_inches='tight', pad_inches=0.02)
print('ok')
