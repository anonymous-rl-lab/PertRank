Native ASH and TRADE posterior execution
======================================

This directory prepares original R ashr 2.2-63 under isolated R 4.4.3.
It does not modify system R or the global Python environment. The exact runtime
comes from the previously verified Stim_Preaccess artifact. All source downloads
and the Posit binary are checked against that artifact's SHA-256 manifests.
Run environment/rebuild_runtime.py after relocation to rebuild the launcher and
libraries; installation paths in the R launch scripts are absolute.

run_posterior.py accepts one NPZ:
  effects: rows x full declared gene axis, float64
  ses: same shape, finite strictly positive values
  row_ids: unique unicode IDs in exactly the row order
  genes: unique unicode gene IDs in exactly the column order

No filtering, standard-error flooring, empirical seed search, benchmark-reference
read or scoring takes place in this runner. Any SE flooring or common gene-axis
definition must be specified and performed by the authorized input builder.

Each row is one perturbation/input repetition; the mixture is estimated using
all its declared genes. Posterior means retain their signs and gene dependence.
They are never compressed into scalar weights.

Prespecified ASH developmental configurations:
  Default uniform/nullbiased (the actual ashr default):
    python run_posterior.py --input INPUT.npz --output NEW_DIR --retry-mixem
  Normal/nullbiased:
    python run_posterior.py --input INPUT.npz --output NEW_DIR --mixcompdist normal --retry-mixem
  Normal/shrink (documented ashr method=shrink, no point mass and uniform prior):
    python run_posterior.py --input INPUT.npz --output NEW_DIR --mixcompdist normal --method shrink --retry-mixem

TRADE full posterior adaptation:
    python run_posterior.py --input INPUT.npz --output NEW_DIR --mode trade --retry-mixem

TRADE primary fitting calls the unmodified official fit_ash() from commit
3432d108c43325e9df65276563d65144a26b69ce, using halfuniform components, uniform
prior and the native adaptive grid. get_pm(fit) returns each coordinate's full
posterior mean. Input scale is common normalized log1p mean difference; this is
a TRADE-core posterior adaptation, not native end-to-end DESeq2 TRADE.

Numerical rule: Primary optimizer is native mixSQP. With --retry-mixem, and only
after an explicit error or convergence=FALSE, the native mixEM optimizer retries
the exact same model with maxiter=10000. For TRADE, this retry is explicitly
implemented by the same ashr call/options plus optmethod=mixEM; the official
TRADE source is never changed. Every original and fallback result and warning
is retained, including per-row solver identity. Unconverged/failed rows remain
NaN; a complete unsuccessful arm cannot be scored as zero or silently subsetted.
ASH requests fitted_g, PosteriorMean, fit_details and loglik through the native
outputlevel option. Removing unused derived statistics was checked on all 32
prespecified D3 smoke targets: posterior binaries, mixtures, loglik and solver
diagnostics were identical to outputlevel=3 (OUTPUT_SELECTION_EQUIVALENCE.json).
TRADE retains its original outputlevel=3. The old complete-output R runner and
the patch lock are retained. This changes no fitting model or optimizer.

Outputs:
  posterior.npz: ordered full posterior matrix and identifiers
  fit_rows.jsonl: per-row diagnostics, mixture parameters and solver paths
  primary_*.rds: native primary fit summaries and exact posterior means
  fit_*.rds: final valid fit summaries
  unconverged_*.rds: any last unconverged fit before declaring failure
  runtime.json / execution.json / config.json: runtime, hashes and arguments
  run.log: complete execution messages

Frozen protocol and ASH grid amendment are owned by the parent experiment.
This directory alone does not claim a completed comparative experiment.
