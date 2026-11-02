# VAZHAKKU — spoken-register conversion and meta-learned dialect conditioning for Tamil speech synthesis

Code for the manuscript *VAZHAKKU*. Every file sits in this one folder; run the scripts from this folder.
Set `VZ_ROOT` to a working directory (default: this folder). Data, features, checkpoints and results go under it.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `prep.py {dialect,iv,slr}` | decode, trim, 80-band log-mel, ECAPA embeddings, metadata |
| 2 | `mel_stats.py iv,slr` | per-band normalisation statistics from training partitions |
| 3 | `precompute_mms.py {corpus}` | cache frozen MMS-TTS Tamil text-encoder states |
| 4 | `extract_w2v.py`, `dialect_probe.py`, `dialect_probe.py --iv_only` | wav2vec2 dialect instrument; district-prior posterior q(r\|u) |
| 5 | `bridge.py --mode {lattice,seq2seq} --region {posterior,hard,none} [--margins 0,0.5,1,1.5,2,3]` | register bridge; identity bias chosen on validation speakers |
| 6 | `bridge_errors.py`, `analyze_bridge.py` | edit precision/recall; seed means, bootstrap CIs, Wilcoxon–Holm, Friedman |
| 7 | `train_backbone.py` | stage S0 backbone (OT-CFM, prior-centred source) |
| 8 | `train_dialect.py --method {clone,maml,hyper} --disent {none,grl,club} [--holdout REGION]` | stage S1, episodic dialect stage |
| 9 | `select_decoding.py` | temperature, start time and Euler steps on validation speakers |
| 10 | `evaluate.py --protocol {seen,anchors}` | DCA, ALI, SECS, CER, MCD, RTF of synthesized speech with natural anchors |
| 11 | `repr_analysis.py out.json name:ckpt ...` | few-shot dialect identification from support codes; probe and EER of speaker codes |
| 12 | `adapt_cost.py`, `train_summary.py`, `mcd_reference.py` | adaptation timing, training-loss summary, constant-spectrum MCD reference |
| 13 | `analyze.py`, `fig_*.py` | synthesis statistics and all figures |

`scheduler.py` runs a job list on two CPU workers; `run_dialect.sh` wraps stage S1.

## Data (public)
* Tamil dialect corpus (DravidianLangTech 2026 shared task; Hugging Face mirror `VishalTheHuman/tamil-dialect-speech`)
* IndicVoices Tamil 10k sample (`PharynxAI/IndicVoices-tamil-10000`)
* OpenSLR SLR65 (`deepdml/openslr65-tamil`)

Speaker-disjoint splits used in the paper are provided as `split_dialect.json`, `split_iv.json`, `split_slr.json`
(one label per row of the corresponding `*_meta.csv` produced by `prep.py`).

## Pretrained components (Hugging Face)
`facebook/mms-tts-tam` (frozen text encoder), `speechbrain/spkrec-ecapa-voxceleb`, `speechbrain/tts-hifigan-libritts-16kHz`,
`microsoft/wavlm-base-plus-sv` (evaluation only), `Harveenchadha/vakyansh-wav2vec2-tamil-tam-250` (evaluation only).

## Requirements
See `requirements.txt`. Everything in the paper ran on a 2-core CPU without a GPU.
