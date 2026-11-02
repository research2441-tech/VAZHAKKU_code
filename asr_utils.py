"""Tamil CTC recogniser used only for evaluation (never for training)."""
from config import ROOT
import torch, numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
_m=None
def load():
    global _m
    if _m is None:
        p=ROOT + "/models/Harveenchadha__vakyansh-wav2vec2-tamil-tam-250"
        _m=(Wav2Vec2Processor.from_pretrained(p), Wav2Vec2ForCTC.from_pretrained(p).eval())
    return _m
def transcribe(wav):
    proc, model = load()
    x = proc(wav, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad(): lg = model(x).logits
    return proc.batch_decode(lg.argmax(-1))[0]
