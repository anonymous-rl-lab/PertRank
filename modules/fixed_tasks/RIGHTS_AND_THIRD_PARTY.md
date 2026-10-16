# Rights and third-party provenance

This package records the completed PertRank analyses and preserves their source provenance. The packaging step does not assign a new license or imply ownership of the original experimental data. An explicit repository-wide license should be selected by the relevant rights holder; in its absence, no additional license is granted here. Original data and third-party materials remain subject to their own terms.

## TRADE fitting core

The experiments used the official fitting helper from:

- Repository: https://github.com/ajaynadig/TRADEtools
- Commit: `3432d108c43325e9df65276563d65144a26b69ce`
- Source: `R/TRADE_univariate.R`
- Recorded source SHA256: `b2fc72c0e5d7c2cde9631e73e33a95cdfed23a31b3a93c5d7141332452426320`

The local execution record found a placeholder license field in the upstream package. Consequently, this public bundle does **not** redistribute that upstream source. It retains the project-written scalar adapter, source identity and fitted sufficient quantities. To inspect the upstream implementation, retrieve the pinned file from the official repository and check the recorded hash and applicable license. The offline reproduction in this package requires neither the upstream source nor R, because it does not refit the mixture.

The comparator is a common-coordinate TRADE-core scalar adaptation. It is not the complete native DESeq2-based TRADE workflow. Retaining the original serialized TRADE weights avoids changing rankings because of cross-language rounding; the scalar formula is independently checked to numerical precision.

## Public experimental source

The original T-cell Perturb-seq dataset is attributed through the source object URLs and study/registration records listed in the README and provenance files. Compact arrays preserve public-data-derived effects and statistics, without creating a new claim of data ownership. This package contains no private OSF reviewer tokens or account credentials.
