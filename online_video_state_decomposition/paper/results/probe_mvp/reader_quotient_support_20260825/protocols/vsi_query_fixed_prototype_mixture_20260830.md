# Query-fixed positive prototype-mixture Gate

Date frozen: 2026-08-30

Role: exposed calibration-development capacity diagnostic

## Decision question

After Gaussian moments and exact regular pages fail, can a query-independent,
data-adaptive positive mixture preserve multimodal K/V bulk while a bounded
query-conditioned exact-cluster branch repairs the remaining innovation?

Each head clusters its 1,568 visual K/V leaves without using the question query,
answer, reader margin, or target output. A prototype stores cluster mass, mean K,
and mean V. The current query evaluates all prototypes with a shared positive
softmax numerator and denominator. Selected clusters replace their prototype with
their exact K/V leaves.

## Frozen scope

- LLaVA-OneVision Qwen2-7B eager attention replay.
- Exposed calibration positions 73--96 only; calibration positions 97--120,
  official selection, and official formal roles stay unread.
- Final generated-token query at language layers 0, 13, and 27.
- Eight frames and all 1,568 native visual tokens.
- Target-free writers: K-only and scale-normalized K+V k-means.
- Prototype counts: 32, 64, and 128; four deterministic Lloyd iterations.
- Active-read budget: at most 392 K/V vector pairs per head, equal to 25% dense
  visual leaves.
- Deployable selector: prototype attention mass per incremental exact-read cost.
- Capacity selector: target-visible local AV defect per incremental read cost.

The writer cost and cold exact storage are deliberately not hidden: they are
excluded only because this Gate asks for representation capacity. A pass cannot
support a writer, TTFT, memory, reader-accuracy, or latency claim.

## Guards

- Engineering smoke is restricted to calibration position 73 and cannot
  produce the registered decision. The decision-bearing run uses 73--96.
- Captured Q/K/V replay error is at most `1e-4`.
- Cluster assignment is deterministic and every visual leaf belongs to exactly
  one non-empty or explicitly ignored prototype.
- Prototype and exact branches share one numerator and denominator.
- Active K/V reads never exceed 392 vector pairs per head.
- Neither writer uses the current query or target; only the registered support
  selector may use current-query mass or the target-visible capacity score.
- The previous valid decisions remain `NO_POSITIVE_GAUSSIAN_MEASURE_PATH` and
  `NO_PROGRESSIVE_EXACT_PAGE_PATH`.

## Decision rule

Eligible candidates require mean active-read ratio at least `3.8x` and maximum
active token count at most 392.

`PROTOTYPE_MIXTURE_DEPLOYABLE_PATH` requires the mass selector to reach visual
mean/P95/worst at most `1%/2%/5%` and full-output mean/P95 at most `0.5%/1%`.

`PROTOTYPE_MIXTURE_CAPACITY_ONLY` requires the oracle selector to reach visual
mean/P95/worst at most `0.5%/1%/2%` and full-output mean/P95 at most
`0.25%/0.5%`.

If neither passes, return `NO_PROTOTYPE_MIXTURE_PATH` and do not train a writer
or router. A capacity-only result authorizes one separately frozen four-arm
training Gate on calibration data; it does not authorize official selection.
