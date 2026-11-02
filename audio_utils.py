"""Audio front-end shared by every model: 16 kHz, 80-band log-mel matched to the HiFi-GAN vocoder."""
from config import ROOT
import io, numpy as np, torch, soundfile as sf, librosa
from speechbrain.lobes.models.HifiGAN import mel_spectrogram

SR = 16000
MEL_KW = dict(sample_rate=SR, hop_length=256, win_length=1024, n_mels=80, n_fft=1024,
              f_min=0.0, f_max=8000.0, power=1, normalized=False,
              norm="slaney", mel_scale="slaney", compression=True)

def decode(audio_dict):
    wav, sr = sf.read(io.BytesIO(audio_dict["bytes"]), dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)

def trim(wav, top_db=35):
    y, _ = librosa.effects.trim(wav, top_db=top_db)
    return y

def mel(wav):
    x = torch.from_numpy(wav).float()
    m = mel_spectrogram(audio=x, **MEL_KW)
    return m.numpy().astype(np.float32)  # (80, T)

_voc = None
def vocoder():
    global _voc
    if _voc is None:
        from speechbrain.inference.vocoders import HIFIGAN
        _voc = HIFIGAN.from_hparams(source=ROOT + "/models/speechbrain__tts-hifigan-libritts-16kHz",
                                    savedir=ROOT + "/models/hifigan_sb")
    return _voc

def vocode(m):
    with torch.no_grad():
        w = vocoder().decode_batch(torch.from_numpy(m)[None])
    return w.squeeze().numpy()
