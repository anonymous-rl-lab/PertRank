# Community integration proposal

Status: prepared for author review; no issue, pull request, registry entry or public release has been submitted.

## Proposed place in the workflow

PertRank addresses candidate prioritization in shallow Perturb-seq: given a response gene, nominate perturbations for a fixed number of follow-up positions. It combines profile-wide effect and variance information with matched control directions to assess target reliability, retaining a gene-aligned readout. The same weights can be applied to an externally estimated effect matrix. This complements screen annotation, differential-expression analysis and global perturbation-distance tools.

The standalone package is the initial delivery. A future pertpy contribution can present the public operation through the namespace selected by its maintainers while retaining the same input checks and numerical tests. This package does not alter `pertpy.tl` or install a monkey patch.

## Proposed upstream discussion text

**Title:** Add query-specific perturbation prioritization with PertRank

PertRank provides target-level reliability weighting for selecting perturbations associated with a specified response gene. The proposed feature accepts AnnData with explicit perturbation, perturbed-gene, matched-control and stratum metadata, and returns gene-specific up/down candidate tables. It also supports reusing fitted weights with external effect estimates.

The implementation uses the manuscript's fixed five-direction geometry and attenuation 0.9. It supplies numerical regression tests against the frozen research engine, checks count normalization before gene filtering, and makes control matching explicit. Replogle and T-cell examples replay the existing real-data analysis statistics, with the processing level of the compact fixtures disclosed.

The proposed contribution is an analysis operation and its documentation, rather than a new data-download framework. Maintainer guidance is requested on the preferred API shape, dependency boundary and return conventions before opening an implementation PR. The standalone package can remain independently maintained if that is the preferred integration model.

## Deliverables to attach to an implementation PR

1. A small API implementation using existing AnnData conventions and explicit scale/matching arguments.
2. A documented result object and exportable candidate tables.
3. Regression tests covering weights, scores, target alignment, self-target exclusion, ties and supported input storage types.
4. Two reproducible tutorials and fixture provenance; tutorial data contain no evaluation inputs used to fit weights.
5. API documentation and a concise explanation of the scientific task.
6. A rights-holder-approved software license and completed attribution.

The pertpy contribution guide calls for corresponding tests and documentation and uses its own development, formatting and CI procedures. Before preparing the actual PR, apply the then-current repository checks. Source: [pertpy contributing guide](https://pertpy.scverse.org/en/latest/contributing.html).

## scverse ecosystem registration specification

The public handoff should verify licensing, versioned registry installation, automated tests, API documentation, scverse data structures and maintainer agreement against the registry's current checklist. Ecosystem listing and scientific evaluation are separate matters. The proposed metadata below is a preparation checklist; the live schema and requirements must be checked when submitting. Source: [official ecosystem registry](https://github.com/scverse/ecosystem-packages).

Proposed descriptive content:

| Field | Proposed value or required evidence |
|---|---|
| Display name | PertRank |
| Description | Candidate prioritization using control geometry for reliability with gene-aligned effect readout |
| Primary category | `scRNA-seq` |
| Tags | `perturbation`, `denoising` |
| Language | `Python` |
| Data-structure use | AnnData inputs, aligned cell/gene metadata, labeled effect matrices and exportable ranking tables |
| Code, documentation and tutorial URLs | Actual public URLs after the approved release |
| Registry installation | Verified package name and published version |
| Maintainers | Actual responsible maintainer identity and contact |
| License | Rights-holder-approved OSI license |
| Inventory | `null` for README-only documentation; actual inventory URL if a compatible documentation site is built |

This is a submission specification, not a validated `meta.yaml`: unresolved identity and URL fields have deliberately not been replaced with invented values. At submission time, validate against the live registry schema and controlled vocabulary.

## Public handoff sequence

Finalize rights, attribution and maintainership; release the tested standalone package; verify installation from the selected registry; submit the ecosystem registration; then discuss the scoped pertpy contribution. These are separate deliverables, each with its own acceptance decision. The current private review package establishes the installable and testable implementation before those public actions.
