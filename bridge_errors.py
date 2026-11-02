"""Edit-level diagnostics of register conversion: precision/recall of word changes against the verbatim reference,
and the share of unseen (novel) output forms that the reference confirms. Position-aligned on equal-length pairs."""
from config import ROOT
import json, sys, numpy as np, pandas as pd
R = ROOT + '/work/res'
runs = sys.argv[1:] or ['bridge_seq2seq_posterior_s0', 'bridge_lattice_noctx_s0', 'bridge_lattice_none_s0', 'bridge_lattice_none_s1',
                        'bridge_lattice_hard_s0', 'bridge_lattice_hard_s1', 'bridge_lattice_posterior_s0', 'bridge_lattice_posterior_s1']
out = {}
for f in runs:
    d = pd.read_csv(f'{R}/{f}_utts.csv').fillna('')
    tp = fp = fn = keep_err = n_al = 0; nov_ok = nov = 0
    for s, h, v in zip(d.src, d.hyp, d.ref):
        s, h, v = str(s).split(), str(h).split(), str(v).split()
        if not (len(s) == len(h) == len(v)): continue
        n_al += 1
        for a, b, c in zip(s, h, v):
            if b != a and b == c: tp += 1
            elif b != a and b != c: fp += 1
            if c != a and b != c: fn += 1
    out[f] = dict(aligned_sentences=n_al, change_precision=tp / max(1, tp + fp), change_recall=tp / max(1, tp + fn),
                  changes=tp + fp)
    print(f, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[f].items()})
json.dump(out, open(f'{R}/bridge_edits.json', 'w'), indent=1)

# ---- error categories on aligned words: over-conversion (changed, speaker kept the standard word), missed change
# (kept, speaker changed it), wrong form (both changed, to different forms); share of wrong forms within one character
def ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p = d[:]; d[0] = i
        for j in range(1, len(b) + 1): d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (a[i - 1] != b[j - 1]))
    return d[-1]
cat = {}
for f in ['bridge_lattice_hard_s0', 'bmargin_hard_s0', 'bridge_lattice_posterior_s0', 'bmargin_posterior_s0']:
    try: d = pd.read_csv(f'{R}/{f}_utts.csv').fillna('')
    except FileNotFoundError: continue
    over = miss = wrong = wrong1 = 0
    for s, h, v in zip(d.src, d.hyp, d.ref):
        s, h, v = str(s).split(), str(h).split(), str(v).split()
        if not (len(s) == len(h) == len(v)): continue
        for a, b, c in zip(s, h, v):
            if b == c: continue
            if b != a and c == a: over += 1
            elif b == a and c != a: miss += 1
            else: wrong += 1; wrong1 += ed(b, c) <= 1
    t = over + miss + wrong
    cat[f] = dict(errors=t, over=over / t, missed=miss / t, wrong=wrong / t, wrong_within_one_char=wrong1 / max(1, wrong))
    print(f, cat[f])
json.dump(cat, open(f'{R}/bridge_error_types.json', 'w'), indent=1)
