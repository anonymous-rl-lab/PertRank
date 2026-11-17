# Verification records

These records come from checks of the delivered computational sources and lossless
stores. The default dispatcher completed all nine workflows; three additional
workflows and full PertRank-Q checkpoint replay passed separately. Paths beginning
with `WORKSPACE` are normalized execution paths, not required input locations.

`DELIVERY_CHECKS.json` gives the coverage summary. Storage, native-material
restoration, scientific recomputation, figure rendering and checkpoint inference
have distinct records. Original negative scientific findings remain in those
outputs. No model was retrained or native R estimator refitted during these checks.

The root file manifest covers the delivered package. Execution receipts record
the input inventory checked before these receipts were added; they are not a
self-referential checksum of this validation directory.
