"""Final training-loss summary of the dialect-stage runs: mean of the last five logged records (last 250 updates)."""
from config import ROOT
import json, os
L = ROOT + '/work/logs'; out = {}
for name in ['clone_s0', 'full_s0', 'collapsed_full_s0', 'hyper_nomi_s0', 'hyper_grl_s0', 'lodo_full_Western', 'lodo_full_Central', 'maml_s0']:
    p = f'{L}/{name}.log'
    if not os.path.exists(p): continue
    recs = [json.loads(l) for l in open(p) if l.startswith('{"step"')]
    if not recs: continue
    last = recs[-5:]
    out[name] = {k: sum(r.get(k, 0.0) for r in last) / len(last) for k in ['dur', 'prior', 'dec', 'mi', 'lz', 'zstd']}
    out[name]['last_step'] = recs[-1]['step']; out[name]['max_dur'] = max(r['dur'] for r in recs)
json.dump(out, open(ROOT + '/work/res/train_summary.json', 'w'), indent=1)
for k, v in out.items(): print(k, {a: round(b, 4) for a, b in v.items()})
