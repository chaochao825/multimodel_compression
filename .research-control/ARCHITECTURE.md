# Research Architecture

Updated: 2026-08-07

## Stable layers

| Layer | Purpose | Change policy |
|---|---|---|
| `PROJECT.md` | North star, primary claim, scope, human authority | Change only by explicit researcher decision |
| `docs/decisions/` | Accepted protected decisions and their rationale | Append or supersede; do not silently rewrite |
| `docs/plans/active/` | One current execution plan | Exactly one active plan |
| `research/` | Claim graph and candidate-line state | Update at every gate |
| `experiments/` | Protocols and append-only experiment registry | Predeclare before runs |
| `results/` | Evidence records, including negative evidence | Preserve after code rollback |
| `STATUS.md` | Compact decision surface | Refresh after material evidence or decisions |

## Direction of dependency

`north star -> claims -> candidate lines -> protocols -> runs -> results -> gate decisions`

Do not reverse this chain by inventing a claim to justify an already attractive result.

## Work lanes

- **Explore:** cheap probes that reduce uncertainty; not claim-confirming evidence.
- **Prove:** frozen protocols that test predeclared claims and thresholds.
- **Integrate:** engineering work that packages already-decided evidence without changing its meaning.

## Extension rule

Add a new component only when it has:

1. a stable owner and interface;
2. a named claim or decision it supports;
3. an explicit source of truth;
4. a removal or retirement condition.

Prefer extending registries and templates over creating parallel status systems.
