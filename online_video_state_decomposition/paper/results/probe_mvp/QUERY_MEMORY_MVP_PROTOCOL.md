# Query-Conditioned Memory MVP Protocol

Status: frozen protocol under execution.

Evidence cutoff: 2026-07-17.

## 1. Research Question

The next experiment asks whether a bounded, query-conditioned episodic memory
can outperform exact recent evidence on language-conditioned video decisions.
The primary hypothesis is:

> An exact recent cache plus a query-conditioned historical archive should
> outperform an exact recent-only cache when both the persistent state and the
> final visual evidence budget are reported explicitly.

This is narrower than claiming a new general streaming-memory architecture.
The experiment is a promotion gate for a later learned VLM-integrated memory.

## 2. Closest Prior Work and Required Controls

The protocol is constrained by the following verified primary sources:

| Work | Mechanism relevant to this MVP | Required control |
|---|---|---|
| CausalMem, arXiv:2606.25658 | Query-free online semantic basis and fixed-budget residual-energy retention | Include a query-free semantic/diversity writer and do not call a semantic basis novel |
| SelectStream, arXiv:2606.16353 | Exact current observation plus query-conditioned historical evidence | Keep recent evidence directly visible in every primary query-memory policy |
| SAVEMem, arXiv:2605.07897 | Semantic write prior, three memory tiers, query-aware late interaction, recency gate | Separate write policy, retrieval scope, and query-time readout |
| FOLIO, arXiv:2607.13298 | Short-term buffer, entity-centered long-term memory, evidence cache, hybrid retrieval | Count or explicitly qualify the visual-evidence cache |
| QSVideo, arXiv:2607.04559 | Query relevance, diversity, and temporal alignment | Test relevance-only and relevance-plus-diversity/temporal controls |
| ReQuest, arXiv:2607.01737 | Learned question-aware selection and adaptive temporal NMS | Reserve a disjoint calibration split for learned or calibrated selectors |
| Query-based frame selection, arXiv:2601.07459 | Query-conditioned submodular selection on MVBench | Include a full-pool query-selection upper bound |
| DSCache, arXiv:2605.01858 | Separate cumulative-past and instant caches | Retain an exact recent tier rather than injecting history indiscriminately |
| MuKV, arXiv:2605.22269 | Multi-grained cache compression and hierarchical query retrieval | Treat frame-only CLIP retrieval as an MVP, not a complete memory system |

The checked CausalMem repository is pinned at commit
`640104b3786125c4918924f9b666ff7fe04d81de`. Its LLaVA implementation updates
an online basis, ranks retained tokens by reconstruction residual energy, and
merges discarded tokens into per-frame background means. Its Qwen2.5-VL
implementation uses residual energy plus an optional recency score. Neither
implementation conditions memory writing or token retention on the evaluation
question. Therefore, CausalMem is a close query-free writer baseline, not an
implementation of the proposed query-time readout.

## 3. Dataset Split and Leakage Boundary

The first task suite remains the five-task MVBench diagnostic:

1. `object_existence`
2. `state_change`
3. `scene_transition`
4. `action_sequence`
5. `moving_direction`

Each task has 200 records. The 40 records per task used by the completed formal
CLIP experiment are excluded before constructing this MVP split.

From the remaining 160 records per task:

- calibration: 20 records per task;
- evaluation: 40 records per task;
- unused reserve: 100 records per task.

The split is generated once with seed `20260717` and serialized as a manifest.
No hyperparameter, ranker, prompt, threshold, or promotion decision may use the
evaluation labels.

### No-answer-leakage rule

The retrieval query is generated from the question text only:

```text
Question: <question>
```

The correct answer and correct answer index are never passed to a write policy,
pool constructor, relevance function, MMR selector, or learned query transform
at evaluation time. Candidate answer texts are used only by the final
multiple-choice CLIP scorer. A secondary option-aware selector may be studied
later, but it must be labeled separately and must use all options symmetrically.

Calibration labels may train or select global readout parameters. The resulting
parameters are frozen before evaluation.

## 4. Stream, State, and Evidence Budgets

The CLIP proxy observes 32 uniformly spaced frames in chronological order. This
is the sampled input stream for the MVP; it is not a claim of full-frame-rate
processing.

Primary budgets:

- final evidence budget: 8 frames;
- exact recent anchors: 3 frames;
- bounded persistent pool: 16 CLIP frame vectors;
- CLIP vector dimension: detected from the model, normally 768;
- persistent vector precision for accounting: FP16;
- frame index metadata: 32-bit integer per retained vector.

All reports must separate:

1. persistent selection state;
2. final evidence frames/tokens passed to the answerer;
3. query-time retrieval FLOPs and latency;
4. visual-evidence cache or frame replay cost.

The CLIP proxy stores frame vectors only. A later LLaVA selection anchor may
re-decode source frames by exported indices. That experiment validates the
selection policy but is not a deployable bounded-memory result until the
compressed visual-evidence cache, host storage, or re-encoding cost is counted.

## 5. Policies

### Required baselines

- `exact_recent`: retain and answer from the last 8 sampled frames.
- `offline_uniform`: 8 uniformly spaced frames; this knows the observed prefix
  length and is not a strict unknown-horizon writer.
- `recent_pool_query_topk`: retain the last 16 vectors and query-select 8.
- `reservoir_recent_query_mmr`: 3 exact recent slots plus a bounded historical
  reservoir, followed by query-conditioned MMR.
- `diverse_recent_query_topk`: 3 exact recent slots plus a query-free diverse
  historical archive, followed by relevance-only retrieval.
- `diverse_recent_query_mmr`: the primary rule-based proposal.
- `offline_full_query_mmr`: query-select 8 from all 32 vectors. This is an
  offline upper bound and must not be presented as a bounded online method.

### Readout score

The primary greedy selector uses:

```text
score(i) =
    relevance(i, question)
  - lambda_diversity * max_similarity(i, selected)
  + lambda_temporal * temporal_coverage(i, selected)
```

The three most recent retained frames are forced into the selected evidence.
`lambda_diversity` and `lambda_temporal` are selected on calibration data only.
Pure query top-k is retained as an ablation with both weights set to zero.

### Learned extension

If the rule-based selector shows a positive calibration and evaluation trend, a
small positive diagonal query gate may be trained on calibration examples. Its
parameter bytes, training objective, regularization, and frozen evaluation
checkpoint must be reported. It is not required for the first smoke run.

## 6. Calibration and Evaluation

The embedding cache stores image vectors, a question-only vector, candidate
answer vectors, sampled frame indices, and immutable metadata. Video decoding
and CLIP encoding are performed once and may be sharded across GPUs.

Calibration performs a deterministic grid search:

- `lambda_diversity`: 0.0, 0.1, 0.25, 0.5;
- `lambda_temporal`: 0.0, 0.1, 0.25;
- primary writer: `diverse_recent`.

Selection criterion:

1. highest macro task accuracy;
2. highest micro accuracy;
3. lower total regularization weight;
4. deterministic lexical tie break.

The chosen weights are then shared by all MMR policy comparisons and frozen for
the evaluation split.

## 7. Metrics and Statistical Tests

Required outputs:

- micro and macro task accuracy;
- task-level accuracy;
- paired gain versus `exact_recent`;
- paired bootstrap 95% confidence interval;
- exact McNemar/binomial p-value;
- better/worse/tied sample counts;
- persistent state bytes;
- final evidence count;
- estimated retrieval FLOPs;
- measured pool update and query selection latency;
- selected frame indices for every sample and policy.

Plots:

1. evaluation accuracy versus persistent state bytes;
2. task-by-policy accuracy heatmap;
3. paired gain versus exact recent;
4. calibration surface over diversity and temporal weights;
5. calibration-only surface for the option-aware feature ranker;
6. calibration-to-evaluation method transfer;
7. selected-evidence overlap versus exact recent;
8. selected-frame temporal distribution by policy.

The analysis pipeline also emits `RESULTS_ANALYSIS.md`, generated from the
frozen aggregate CSV and JSON files.

## 8. Promotion Gates

The query-memory direction advances to the expensive VLM anchor only if a
strictly bounded policy, not only the offline full-pool upper bound, satisfies:

- evaluation macro gain at least `+0.03` versus `exact_recent`, or a positive
  paired gain with at least 3 of 5 tasks non-worse and a bootstrap lower bound
  above `-0.02`;
- no evidence of answer leakage or split overlap;
- all state and evidence budgets are reported;
- the selected policy is not chosen using evaluation labels.

The direction advances beyond the LLaVA anchor only if the selected bounded
policy has a positive point estimate versus exact recent on the same examples
and the gain is not confined to one task.

## 9. Claim Boundary

Positive CLIP results establish only that question-conditioned evidence
selection is promising under a frame-embedding proxy. Positive LLaVA results
establish only that the selected raw frames transfer to one VLM anchor. Neither
result alone establishes a real-time streaming system, an entity memory, a
token-level memory bank, or an algorithm-hardware co-design.

If only `offline_full_query_mmr` wins, the conclusion is that query-conditioned
retrieval helps when the full history remains available, not that a bounded
online memory has been solved. If exact recent remains best, the project should
not hide the negative result; it should redirect toward richer semantic/event
writers or native VLM/KV-level retrieval.

## 10. Untouched-Reserve Confirmation

The preregistered `diverse_recent_query_mmr` primary failed its evaluation
gate. `recent_pool_query_topk` produced a post-hoc positive point estimate, so
it is not promoted from the first evaluation. It receives one frozen
confirmation on previously unused reserve records:

- 40 reserve records per task, 200 total;
- no calibration records and no hyperparameter search;
- primary policy fixed to `recent_pool_query_topk`;
- reference fixed to `exact_recent`;
- evidence, pool, and recent-anchor budgets fixed to 8, 16, and 3;
- all five feature weights loaded from the frozen formal-run JSON;
- split SHA256:
  `406768bc85c4ffd9ebc5c99d002fcef7a24f3b69cf73697637fbe6aa2cc65e44`;
- hyperparameter SHA256:
  `c04db25ab9597c4079e675a2790f3cee56e40bf2d87b06b20b4215dcabf26681`;
- exploratory learned-ranker SHA256:
  `65226a3ef558bf0968d48fa8a482cec85a7a0d39af85f0226c785f613e666f12`.

The learned readout is frozen before reserve evaluation. It is a four-feature
ridge model fit only on the original 100 calibration records (1,600 frame
targets), occupies 48 FP32 parameter bytes, and never receives evaluation
answer labels. It remains an exploratory secondary policy; the confirmation
primary and promotion gate are unchanged.

The same promotion gate is applied. A failed confirmation ends the handcrafted
CLIP retrieval branch. A passed confirmation permits only a paired raw-frame
LLaVA anchor; it does not establish a bounded deployment or a learned memory.

## 11. Paired LLaVA Raw-Frame Anchor

The reserve confirmation passes the protocol's weak trend gate, so one paired
LLaVA-1.5-7B anchor is run without changing the sample set or selectors:

- the same 200 untouched-reserve samples, 40 per task;
- policies fixed to `exact_recent`, `recent_pool_query_topk`,
  `recent_pool_query_mmr`, and `learned_recent_query_topk`;
- 8 selected raw frames per sample;
- adaptive 8 by 8 visual pooling per frame, 512 visual tokens total;
- identical prompts, answer parsing, decoding, and generation settings for all
  policies;
- selection-manifest SHA256:
  `dbc763008af72a9e335039bc0ab61e08e4fc5375b3996b72a304c2ea9789945c`.

The direction advances beyond this anchor only if a frozen bounded policy has
a positive paired LLaVA point gain and positive task gains in at least two
tasks. Raw-frame replay validates evidence selection only; it does not count
the source video or decoded frame cache as bounded persistent state.

### Anchor result

All 200 checkpoints and 800 policy predictions complete with one configuration
fingerprint and 100% answer parsing. Against 47.0% exact recent:

- frozen recent-pool top-k: 50.0%, paired gain +3.0 percentage points,
  interval [-1.0, +7.0];
- recent-pool MMR: 50.5%, paired gain +3.5 points, interval [0.0, +7.0];
- frozen learned recent-pool top-k: 51.0%, paired gain +4.0 points,
  interval [+1.0, +7.5], 10 better / 2 worse, McNemar p=0.0386.

The learned selector improves action sequence, scene transition, and state
change, and does not reduce moving direction or object existence. It therefore
passes the raw-frame transfer gate. It is not significantly better than the
two recent-pool query-only controls, however: +1.0 point versus top-k and +0.5
point versus MMR, with both paired intervals crossing zero.

The deployment gate remains open. Query policies retain 24.14 KiB of selection
state versus 12.05 KiB for exact recent, and source-video replay remains
unbounded. Mean end-to-end latency is 4.71-4.88 seconds per sample, of which
only 0.28-0.30 seconds is `model.generate`; decoding and preprocessing dominate
this unoptimized anchor. The next experiment must match persistent bytes and
move selection into a native memory or VLM/KV interface.

## 12. Matched-State Native Feature-Memory Anchor

The raw-frame transfer gate permits one native-cache confirmation with all
selection policies and sample identities frozen:

- the same 200 untouched-reserve samples and four selection policies;
- the same selection-manifest SHA256:
  `dbc763008af72a9e335039bc0ab61e08e4fc5375b3996b72a304c2ea9789945c`;
- one 16-frame projected LLaVA feature cache written before the query;
- 64 projected tokens per frame, hidden size 4096, stored in FP16;
- 8 selected frames and 512 visual tokens read by every policy;
- the maximum selector allocation, 24,720 bytes, provisioned to every policy;
- total persistent state fixed to 8,413,328 bytes per policy;
- the visual cache included in state accounting and no source-video replay at
  query time.

The run has configuration fingerprint
`b5538473e2354f57af923385a3bd63fc91fffe27f1763cd81b14592080a38c48`
and configuration SHA256
`95f3c8a1d8f2ffd000a14822349edc35834df0b0912564012c151ab3f19df036`.

### Native result

All 200 checkpoints and 800 predictions complete with 100% parsing and no
failures. Exact recent reaches 47.5%. Frozen recent-pool top-k reaches 50.0%
(+2.5 points, interval [-1.0, +6.5]); MMR reaches 50.5% (+3.0 points,
interval [0.0, +6.5]); and the frozen learned readout reaches 51.0% (+3.5
points, interval [0.0, +7.0], 10 better / 3 worse, McNemar p=0.0923).

The learned readout is not statistically significant versus exact recent,
top-k, or MMR. Its positive task gains remain distributed across action
sequence, scene transition, and state change. Moving direction is unchanged,
and object existence is preserved by the learned selector but reduced by the
two query-only controls.

The native and raw paths use identical selected frames on all 800 policy
evaluations. Predictions agree on 797/800 and correctness agrees on 799/800.
The paths are therefore task-equivalent at this scale but not bit-exact; FP16
visual encoding with different batch shapes can move borderline generations.

Mean native-cache write time is 8.68 seconds: 7.06 seconds decoding, 1.23
seconds preprocessing, and 0.39 seconds visual encoding. Mean cached read plus
generation is 0.083 seconds; P95 and P99 cached-read times are 0.090 and 0.093
seconds. The implementation removes query-time replay but is not optimized for
online writing.

### Compression gate result

The low-rank projected-feature codec is fitted only on the original 100
calibration videos. Formal confirmation uses 200 disjoint reserve videos and
compares the full 8 MiB native feature cache, rank-256 latent state, and the
same latent state plus four highest-residual tokens per frame.

The sparse-residual state uses 1.024 MiB per stream versus 8.024 MiB for the
full cache. The shared codec is 2.008 MiB and is reported separately; including
it gives a 3.032 MiB cold start. Learned-selector accuracy is 50.5% versus
51.0% full, with 99% prediction agreement and one full-correct/compressed-wrong
event. The one-sided 95% exact loss-rate bound is 2.35%, so the 2% preservation
gate is not passed.

The matched-state query-memory effect is retained: learned sparse-residual
memory reaches 50.5% versus 46.5% exact recent, +4.0 points with interval
[+1.0, +7.0] and McNemar p=0.0215. This promotes adaptive sparse-event
allocation as the next mechanism probe, not PCA or low-rank-plus-sparse as a
novel method claim.
