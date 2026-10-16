NATIVE FR-PERTURB ADAPTER — IMPLEMENTATION RECORD

Official source: https://github.com/douglasyao/FR-Perturb
Commit: 24e653b6a5c050ab17f3600ebaf222bef32e2d25
Official GPL-3.0 LICENSE retained under FR-Perturb/LICENSE.
The vendored source is unmodified. official_manifest.json pins source hashes;
fr_native_runner verifies them before fitting. It imports factorize_recover
and scale_effs directly. No alternative SVD, regression or PCA replaces them.

Preprocessing and computation
1. Accept the full raw CSR gene matrix and shallow cells only.
2. Reject missing/duplicate IDs, noninteger/negative counts, empty cells or
   fewer than 100 NTCs rather than silently altering the eligible cells.
3. Official Scanpy input-only all-zero-gene filter; normalize_total(target_sum
   10000), log1p; center on the supplied NTC mean. Full library totals, not
   query/HVG totals, define the per-cell normalization.
4. Official SPAMS trainDL and lasso projection, then sklearn Lasso recovery.
   Defaults: rank20, spca_alpha0.1, lasso_alpha0.0001, trainDL iter50.
   Resource wrappers set numThreads=2 and capture elapsed time; a sklearn
   subclass captures convergence diagnostics without changing fit arguments.
5. Official scale_effs LOWESS rescaling. Save output before CLI signif(...,3).
   Full native float precision is retained. A positive gene-specific scale
   cannot change target ranking; all gene rankings are checked explicitly.
6. Control is an explicit final design column. Its fitted coefficient is
   retained in native arrays and excluded from target output; no second
   subtraction of the control coefficient is performed.

Input format A
  python fr_native_runner.py --input input.npz --out fresh_directory
NPZ contains data, indices, indptr, shape (raw CSR), target_index (-1 for NTC,
0..T-1 targets), targets, genes, and optional cell_ids.

Input format B
  python run_prepared_input.py --input-dir input_directory --out fresh_directory
Directory contains scipy counts.npz and axes.npz with labels, gene_ids,
target_ids, cell_ids. Input file hashes are written in the fit report.

Use OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2. Runner prepends local deps/ and
the vendored repository to sys.path. On another machine, install requirements
and retain the vendor relative directory. Fitting is sequential because the
native factorization requires a dense centered matrix.
The actual cgroup limit is8 GiB despite a larger host memory inventory.
run_prepared_input.py acquires fr_benchmark/heavy_memory.lock before loading
counts and holds it through the complete fit, mutually excluding raw-input
exports. This resource wrapper does not change inference. The transition from
the initial legacy fits is recorded in HEAVY_MEMORY_COORDINATED.json.

Outputs
effects.npz: effect and effect_unscaled, target x original full gene axis.
Input-absent genes are exactly zero; retained_gene_index discloses the filter.
native_unscaled.npz: B (targets plus final NTC), U, U_tilde, W, gene names,
perturbation_names (including final __NTC__), NTC mean and scaling inputs.
fit_report.json: source/input hashes, all version/parameter/runtime/warning
records, LASSO dual gaps/iterations and independent one-hot analytic audit.
fit.log: stage progression. Failed runs remain marked FAILED, never zero-scored.

Numerical validation completed before real-data evaluation
- Synthetic 360 cells x600 genes, default native parameters.
- Adapter vs original CLI (intercepted only before output rounding): maximum
  absolute full-precision difference 1.11e-16; threshold1e-12 PASS.
- Repeated fresh-process synthetic fits produced bitwise identical effects.
- One-hot design allows independent recovery audit against coordinatewise
  soft-thresholded latent means; error <5e-15 in synthetic and first real smoke.
- Real smoke: fixed hash32 targets x20 cells plus NTC2000; all18130 raw genes,
  13564 input-expressed genes. 12.95 seconds; maxRSS1,147,240 KiB; no warnings.
- Full395target engineering pilot:9900cells,14653 input-expressed genes,
  25.93 seconds; maxRSS3,634,476 KiB; no warnings. No reference scoring was
  performed by the implementation agent.
- Before D3 fitting, sparse-minus-dense centering was implemented as dense
  allocation followed by in-place subtraction, avoiding a second full dense
  temporary. The full395target repeat produced bitwise identical scaled and
  unscaled effects (maximum difference0); maxRSS fell to2,499,612 KiB, time
  24.25 seconds. MEMORY_EQUIVALENCE.json and both run reports are retained.

Limits and failed engineering attempts
- Official environment.yml specifies Python3.8 and older package binaries.
  Execution uses Python3.12 with recorded compatible current packages;
  original CLI equivalence is verified in this same environment, not across
  historical package versions.
- Initial newest Scanpy/AnnData installation was incompatible with available
  transitive packages. The functioning versions are Scanpy1.10.4/AnnData0.10.9.
- synthetic_smoke_02 failed while saving an added string label via numpy.r_.
  No scientific computation changed. np.concatenate corrected serialization,
  synthetic_smoke_03 passed; the failed report is retained.
- SPAMS exposes no seed parameter. Python/NumPy are seeded and empirical
  fresh-process reproducibility is checked; this does not promise bitwise
  reproducibility on different hardware or library builds.
- trainDL executes the official fixed50-iteration budget, not a demonstrated
  dictionary convergence test. LASSO convergence is monitored separately.
- For this one-hot design, LASSO thresholds latent group means by alpha*N/n.
  AMENDMENT_04_FR_EFFECTIVE_PENALTY.json separately registers preservation of
  that threshold during transfer, while the fixed nominal-alpha comparison
  remains primary. run_locked_transfer.py executes the two arms separately,
  after FR_D3_WINNER_LOCK.json exists, using complete native inference.
- P values/permutation testing are not requested because ranking uses effect
  coefficients; skipping those leaves coefficient estimation unchanged.

2026-10-12 pre-performance numerical audit correction: original runner and failed results retained unchanged. New fr_native_runner_audit_v2.py uses native solver dual-gap acceptance plus independent exact one-hot objective audit; coefficient error remains a diagnostic. No official parameters/tolerance or native arrays changed. Two identical-parameter reruns reproduced B/U/U_tilde/W bitwise. Effective 21-grid lives in fr_outputs/D3_Rest_audit_v2; original incomplete ledger is preserved in D3_Rest. See NATIVE_SOLVER_AUDIT_AMENDMENT_V2.json and independent audit/NUMERICAL_AUDIT_FINAL.json.
