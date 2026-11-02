"""Character tokenisation for the pretrained Tamil text encoder (MMS-TTS Tamil vocabulary, blank-interleaved)."""
from config import ROOT
import numpy as np
from transformers import AutoTokenizer
P = ROOT + '/models/facebook__mms-tts-tam'
_tok = None
def tok():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(P)
    return _tok
def text_to_ids(text):
    return np.array(tok()(text)['input_ids'], dtype=np.int64)
N_SYM = 59
