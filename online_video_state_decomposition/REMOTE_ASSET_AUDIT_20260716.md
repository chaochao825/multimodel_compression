# Remote Asset Audit

Audit date: 2026-07-16. Commands were read-only. Missing Python packages may be
installed with the configured mirror when an experiment requires them.

## Recommended Execution Order

1. `210` for the first probes and online-video MVP.
2. `34` only when its GPUs and disk pressure clear.
3. `35` mainly for the deferred Wan generation transfer.

## Server 210

- Profile: `210`, host `172.25.5.210`, user `wangmeiqi`.
- GPU: four A800 80 GB; GPUs 0-2 were idle and GPU 3 was occupied during audit.
- Storage: `/home` had about 17 TB free.
- Preferred existing environment: named conda environment `Qwen3`, resolved to
  `/home/wangmeiqi/anaconda3/envs/Qwen3`.
- Verified environment core: Python 3.10, PyTorch 2.6.0+cu124,
  Transformers 4.51.0, Safetensors 0.7.0.
- Missing at audit time: `accelerate`, `qwen_vl_utils`, `flash_attn`, `cv2`,
  `scipy`, `sklearn`, and `matplotlib`.
- Installation policy: install required packages through the `ssh-dev` mirror
  workflow; create a separate conda environment if dependency conflicts appear.

Verified model assets:

| Path | Status | Intended use |
|---|---|---|
| `/home/wangmeiqi/dqy/Qwen3-VL-30B-A3B-FP8` | Four safetensor shards and index verified | Primary visual-tower probe |
| `/home/spco/models/llava-v1.5-7b` | Complete 13 GB model | Second visual encoder |
| `/home/wangmeiqi/zjh/meta-llama/Llama-2-7b-hf` | Complete | Normal-precision language backbone control |
| `/home/wangmeiqi/zjh/mistralai/Mistral-7B-v0.1` | Complete | Normal-precision language backbone control |
| `/home/wangmeiqi/zjh/nyu-visionx/Cambrian-S-7B-LFP` | Model present but directory is very large | Optional cross-model replication |
| `/home/spco/Monarch/MonarchRT` | Present | Deferred generation/structured baseline |
| `/home/spco/diff_bitnet` | Present | Separate parameter-compression evidence only |

Verified dataset assets:

| Path | Status | Intended use |
|---|---|---|
| `/home/wangmeiqi/.cache/huggingface/videomme` | About 95 GB and 900 MP4 files | Initial real-video probe pool |
| `/home/wangmeiqi/.cache/huggingface/videomme/data/fFYNmVb3NCQ.mp4` | Verified | Existing development clip |
| `/home/wangmeiqi/.cache/huggingface/videomme/data/VP4GtrEsefk.mp4` | Verified | Existing development clip |
| `/home/wangmeiqi/.cache/huggingface/videomme/data/HProiNnmGwI.mp4` | Verified | Existing development clip |
| `/home/wangmeiqi/.cache/huggingface/hub/datasets--OpenGVLab--MVBench` | About 15 GB | Secondary long-video task candidate |
| `/home/spco/BitVLA/lmms-eval` | Task definitions present | Reuse evaluation adapters where compatible |

## Server 34

- Profile: `434`, host `172.25.4.34`.
- GPU: three A100 80 GB devices were heavily utilized during audit.
- Storage was critically constrained: `/` about 97% used and `/data1` effectively full.
- Complete useful assets include Llama-2-13B-Chat, Qwen2-7B-Instruct,
  RoBERTa-base, and RoBERTa-large.
- `/data1/liaosiyu/Llama-2-7b-hf` is an incomplete placeholder.
- Qwen2.5-VL directories found under LLaMA-Factory appear to be small adapters,
  not complete base weights.
- Decision: do not schedule first online-video probes here.

## Server 35

- Profile: `435`, host `172.25.4.35`.
- GPU: A100 80 GB devices were heavily utilized during audit.
- Storage was highly constrained on `/data5` and `/data2`; `/data6` had limited
  remaining headroom.
- `/data6/user24215463/Wan2.2_modules/Wan2.2-T2V-A14B` is a complete roughly
  118 GB generation asset.
- The located Qwen2.5-VL-3B cache was only about 12 KB and is incomplete.
- Decision: reserve for deferred Wan experiments; do not use as the second
  online-understanding encoder.

## Dataset Gaps

The following remain unresolved and therefore keep research issue `R3` open:

- StreamingBench source, split, license, and local path.
- OVO-Bench source, split, license, and local path.
- OVBench/VStream-QA exact evaluation assets.
- Ego4D subset and license if used for event evaluation.
- A reproducible mapping from Video-MME videos to stable, motion, scene-cut,
  OCR, and rare-event probe strata.

## Operational Guardrails

- Check GPU occupancy immediately before every submission.
- Keep all output under a dedicated workspace on 210.
- Record model shard hashes, video hashes, package versions, GPU name, and
  command line in every result artifact.
- Do not transfer the large Wan or Cambrian directories unless the corresponding
  experiment is approved.
- Count CPU and disk offload in state-byte and latency measurements.
