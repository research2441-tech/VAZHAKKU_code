from config import ROOT
import sys
import numpy as np, json
from data import Corpus, make_splits
from evaluate_utils import mcd_dtw
D=Corpus('dialect'); sp=np.array(make_splits()['dialect']); te=np.where(sp=='test')[0]
v=[]
for i in te:
    g=D.get_raw_mel(i); v.append(mcd_dtw(np.repeat(g.mean(1,keepdims=True),g.shape[1],1),g))
r=dict(flat_mcd_mean=float(np.mean(v)),flat_mcd_sd=float(np.std(v)),n=len(v))
print(r); json.dump(r,open(ROOT + '/work/res/mcd_reference.json','w'))
