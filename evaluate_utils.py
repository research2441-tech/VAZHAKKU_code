import numpy as np
def cer(ref, hyp):
    r, h = list(ref.replace(' ', '')), list(hyp.replace(' ', ''))
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev = d.copy(); d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))

def mcd_dtw(m1, m2):
    """mel-cepstral distortion (dB) between two log-mel spectrograms (80,T), DTW aligned, c0 excluded"""
    from scipy.fft import dct
    import librosa
    c1 = dct(m1.T, type=2, norm='ortho', axis=1)[:, 1:25]; c2 = dct(m2.T, type=2, norm='ortho', axis=1)[:, 1:25]
    Dm = ((c1[:, None, :] - c2[None, :, :]) ** 2).sum(-1) ** 0.5
    _, wp = librosa.sequence.dtw(C=Dm)
    return float(10 / np.log(10) * np.sqrt(2) * np.mean([Dm[i, j] for i, j in wp]))
