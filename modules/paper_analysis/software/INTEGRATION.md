# Software integration

This tree supplies PertRank `0.2.0rc3` for the current review manuscript. The software version and reproduction-bundle version identify different deliverables.

The numerical API, five control directions, attenuation 0.9 and tutorial arrays are retained. This source revision checks documentation, distribution contents and tutorial execution paths. Source reliability statistics remain distinct from the effects used for ranking.

`tests/reference_core.py` is a frozen regression oracle; its content hash is recorded in `tests/REFERENCE_PROVENANCE.json`. It provides an independent implementation for synthetic cell-level software regression. Current validation records describe the delivered code.

The integrated archive carries this software source once. Wheel and source distributions can be built from an external copy. PertRank-Q is a separate optional checkpoint-replay module in the full bundle; it is not imported by the core Python package.
