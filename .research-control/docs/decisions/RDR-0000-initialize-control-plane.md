# RDR-0000: Initialize the research control plane

- Status: accepted
- Date: 2026-08-07
- Decider: researcher

## Context

The project needs a durable distinction between scientific direction, candidate tactics, experiment evidence, and implementation state.

## Decision

Use the files under `.research-control` as the project control plane. Maintain exactly one mainline, at most two bounded side probes, one active plan, explicit experiment gates, and durable negative evidence.

The Agent may optimize tactics inside an approved gate. It may not autonomously change the primary claim, mainline, protected architecture, resource envelope, canonical repository, or external release state.

## Consequences

- New work must map to a claim and a decision.
- A failed or plateaued run does not authorize a pivot.
- Valid negative evidence survives code rollback.
- Superseding this decision requires a new accepted RDR.
