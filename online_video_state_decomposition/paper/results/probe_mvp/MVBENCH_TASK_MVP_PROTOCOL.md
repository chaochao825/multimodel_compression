# MVBench Task-Level MVP Protocol

## Purpose

This stage tests whether the two-timescale memory retained by the
representation probes is useful for language-conditioned video decisions.
It deliberately separates two questions:

1. Does a low-rank long-term state improve multiple-choice evidence retrieval
   at a fixed state payload?
2. Does a recent-plus-history frame policy help a real VLM when the visual
   token budget is fixed?

The first question is evaluated with a CLIP task proxy. The second is
evaluated with a pooled multi-frame LLaVA-1.5 anchor. Results from the two
tracks must not be conflated.

## Track A: Matched-Byte CLIP Memory

### Data

- Benchmark: MVBench local snapshot.
- Tasks: `object_existence`, `state_change`, `scene_transition`,
  `action_sequence`, and `moving_direction`.
- Formal MVP sample: 40 deterministically selected examples per task.
- Video sampling: 32 uniformly spaced frames per video.

### Model and query

- Model: normal-precision `openai/clip-vit-large-patch14-336`.
- Each frame is represented by its normalized 768-dimensional CLIP image
  embedding.
- Each candidate is represented by the normalized CLIP text embedding of
  `Question: <question> Answer: <candidate>`.
- Candidate evidence is aggregated over stored or reconstructed prototypes
  with temperature-controlled log-mean-exp pooling.

### Fixed payload comparison

Each method receives the same vector-equivalent payload at a given capacity:

`payload_bytes = capacity * 768 * 16 / 8`.

The tested capacities are 4, 8, and 16. Metadata bytes are reported
separately.

Methods:

- `recent_window`: latest exact frame embeddings.
- `uniform_reservoir`: online reservoir of exact frame embeddings.
- `adaptive_slots`: online centroid slots.
- `oja_subspace`: one running mean plus `capacity - 1` Oja directions,
  decoded into virtual ellipsoid prototypes at read time.
- `instant_oja`: three exact recent slots plus a long-term Oja state in the
  remaining payload.
- `full_sequence`: all 32 frame embeddings, reported only as a larger-state
  reference.

### Primary metrics

- Macro task accuracy.
- Micro accuracy and Wilson 95% interval.
- Per-task accuracy.
- Gain versus `recent_window` at the same capacity.
- Total state bytes.
- Memory update/read wall time and operation proxy.

### Promotion gate

The two-timescale state is promoted only if `instant_oja`:

- improves macro task accuracy over `recent_window` by at least 0.03 at one
  matched capacity;
- is not worse on more than one of the five tasks;
- remains stable across at least two capacities; and
- does not rely on a lower parsed-sample count.

Failure to pass this gate means the reconstruction advantage did not transfer
to language-conditioned decisions under this scoring model.

## Track B: Pooled Multi-Frame LLaVA Anchor

### Setup

- Model: local normal-precision LLaVA-1.5-7B.
- Frames per sample: 8.
- Each frame's 24 by 24 projected visual grid is adaptively pooled to 4 by 4.
- Total visual tokens: 128 per sample.
- Generation is deterministic and constrained by a prompt requesting one
  option letter.

Frame policies:

- `uniform`: eight frames across the full video.
- `recent`: the final eight frames.
- `hybrid`: five historical uniform frames plus three exact recent frames.

### Metrics

- Answer parsing rate.
- Macro and micro accuracy.
- Per-task accuracy.
- Mean generation latency.
- Visual-token and selection-state byte proxies.

### Promotion gate

The anchor is considered usable only if:

- parsing rate is at least 0.95;
- the same prompt and token budget are used for all policies; and
- `hybrid` improves over both `uniform` and `recent` by at least 0.02 macro
  accuracy, or shows a clear task-specific gain on long-horizon categories.

This anchor does not directly evaluate Oja vectors inside LLaVA. It tests the
two-timescale frame-allocation hypothesis with in-distribution visual inputs.

## Claim Boundary

Allowed conclusions:

- matched-byte language-conditioned retrieval behavior in CLIP space;
- fixed-token frame-policy behavior in a real LLaVA inference path;
- state-size and latency trade-offs for the implemented proxies.

Disallowed conclusions:

- end-to-end superiority of Oja memory inside LLaVA;
- general MVBench state of the art;
- proof that low-rank memory improves all online video understanding models;
- hardware speedup without measured kernels or system-level implementation.
