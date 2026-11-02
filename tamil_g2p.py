"""Deterministic Tamil grapheme-to-phoneme conversion at the akshara level.
Each consonant letter carries the inherent vowel /a/ unless followed by the virama (pulli)
or a dependent vowel sign. Output symbols are phonemic, not allophonic."""
import re, unicodedata
CONS = {'க':'k','ங':'ng','ச':'c','ஞ':'nj','ட':'tt','ண':'nn','த':'t','ந':'n','ப':'p','ம':'m',
        'ய':'y','ர':'r','ல':'l','வ':'v','ழ':'zh','ள':'ll','ற':'rr','ன':'n_','ஜ':'j','ஷ':'sh',
        'ஸ':'s','ஹ':'h','க்ஷ':'ksh','ஶ':'sh'}
VOW = {'அ':'a','ஆ':'aa','இ':'i','ஈ':'ii','உ':'u','ஊ':'uu','எ':'e','ஏ':'ee','ஐ':'ai','ஒ':'o','ஓ':'oo','ஔ':'au'}
SIGN = {'ா':'aa','ி':'i','ீ':'ii','ு':'u','ூ':'uu','ெ':'e','ே':'ee','ை':'ai','ொ':'o','ோ':'oo','ௌ':'au'}
VIRAMA = '்'; AYTHAM = 'ஃ'
PHONES = sorted(set(CONS.values()) | set(VOW.values()) | {'q', '|'})  # q = aytham, | = word boundary

def clean(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^஀-௿ ]', ' ', text)   # keep Tamil block only
    return re.sub(r'\s+', ' ', text).strip()

def g2p_word(w):
    out, i = [], 0
    while i < len(w):
        ch = w[i]
        if ch in CONS:
            out.append(CONS[ch]); nxt = w[i+1] if i+1 < len(w) else ''
            if nxt == VIRAMA: i += 2; continue
            if nxt in SIGN: out.append(SIGN[nxt]); i += 2; continue
            out.append('a'); i += 1; continue
        if ch in VOW: out.append(VOW[ch])
        elif ch == AYTHAM: out.append('q')
        i += 1
    return out

def g2p(text):
    seq = []
    for w in clean(text).split():
        p = g2p_word(w)
        if p: seq += p + ['|']
    return seq[:-1] if seq else seq
