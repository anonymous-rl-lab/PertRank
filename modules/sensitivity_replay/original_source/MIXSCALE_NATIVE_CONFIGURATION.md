# Native Mixscale configuration and scope

The completed comparison uses Mixscale 0.3.0 at upstream commit
`3d3a2cd12e62382df01c056536d967534b2de3a9`. It is a development comparison
on the opened D3 responsive 20-cell task, using 1,200 targets, 2,000 input
non-targeting control cells and the original 300 queries in both directions.
The fixed task, estimator settings and evaluation rules are specified in
`MIXSCALE_BOUNDED_PROTOCOL.md` and `MIXSCALE_EVAL_FREEZE.json`.

The completed native fitting used two workers, fixed per-target seeds and the
full 18,130-feature axis. The compact adapter preserves global preprocessing,
nearest-control selection, native perturbation scores, leave-one-gene-out
calculations and regression outputs. Execution-equivalence checks permitted
release of the unused PRTB and RNA scale-data slots before the native DE call;
cell and gene subsampling and dispersion approximations were not used.

All 1,200 target outputs must be complete before evaluation. The frozen ranking
readout is `beta_weight`. Queries omitted by the specified native expression
filter receive zero; fit errors and nonfinite coefficients do not. Memory-guard
interruptions were execution events, not scientific method failures or scored
comparisons. The completed coefficients, filter coverage and final contrasts
are retained in this module.

The portable entry reproduces rankings and uncertainty summaries from the
retained coefficients and reference inputs. It does not run native fitting.
The original generation scripts and frozen source identities document the
native computation; full native fitting additionally requires the original
raw-cell inputs and compatible R dependencies.
