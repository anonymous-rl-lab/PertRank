GPerturb real D3 resource gate, 2026-10-14

The predeclared width-700 configuration did not complete within its 6.5 GiB
worker-RSS execution budget. The worker intentionally stopped at 6.6322 GiB
(29.58 s; worker exit 125). This was a resource-guard stop, not a recorded OS
OOM, a convergence result, or a loss against PertRank. Width 1000 was not run.
No gene-panel change or automatic rescue fit was attempted. The official model
allows filtered gene panels and different covariates; other legitimate adapters
remain unevaluated (see the independent fairness/scope review).

The gate retained the original D3 rep-0 26,000 cells, 1,200 targets, 2,000
controls and all 300 queries on the 15,616 input-positive gene axis. Raw data
have 18,130 genes. The adapter used real metadata-derived 26 covariates, the
unchanged pinned native Gaussian objective, and the native whole-validation
logic. The frozen request was 20 training batches followed by one full
validation call. Stage completion was not persisted before the guard: no claim
is made that all 20 batches or validation completed, and no 250-epoch runtime
projection can be certified. The tiny implementation smoke is a separate test.

Reproduction: extract at the original project root (archive paths start with
fr_benchmark/). Original D3 counts.npz and axes.npz are external prerequisites,
identified by path and SHA-256 in REAL_D3_INPUT_MANIFEST.json; large expression
inputs and the reconstructible torch wheel/venv are deliberately omitted.
Official source, environment versions/wheel hash, preparation and resource
harnesses, exact metadata/covariates, protocol, worker/supervisor logs and the
separately verified result are included. The command in the frozen protocol
requires a rebuilt environment and a new unique output path. Re-execution was
not authorized as an automatic continuation after the guard. Files retain their
historical statuses: the frozen protocol says NOT_EXECUTED because it was
written before execution; the RESOURCE_GATE_RESULT is the outcome.

GPerturb_REAL_D3_WIDTH700_RESOURCE_GATE_20261014.json was not created by the
interrupted worker. Its absence is expected. The child log is empty; the worker
guard and supervisor files carry the observed measurements. Proxy-PID /proc
RSS is not the model worker's RSS and is explicitly excluded from inference.
