"""Layer-wise mean+std pooled hidden states of the Tamil wav2vec2 CTC model (frozen), used as a
phonetically oriented representation for the dialect classifier."""
from config import ROOT
import numpy as np, torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor

P = ROOT + "/models/Harveenchadha__vakyansh-wav2vec2-tamil-tam-250"
LAYERS = (5, 6, 7, 8)
_m = None


def model():
    global _m
    if _m is None:
        _m = (Wav2Vec2FeatureExtractor.from_pretrained(P),
              Wav2Vec2ForCTC.from_pretrained(P, output_hidden_states=True).eval())
    return _m


def embed(wav, max_s=6.0):
    fe, m = model()
    wav = wav[: int(max_s * 16000)]
    x = fe(wav, sampling_rate=16000, return_tensors='pt').input_values
    with torch.no_grad():
        hs = m(x).hidden_states
    out = []
    for l in LAYERS:
        h = hs[l][0]
        out += [h.mean(0), h.std(0)]
    return torch.cat(out).numpy().astype(np.float32)
