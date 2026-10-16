"""Complete AnnData input path; synthetic software example, not biology."""
from pathlib import Path
import pertrank as pr
from pertrank.datasets import software_example

adata = software_example()
result = pr.rank(
    adata, ["G20", "G21"], input_scale="counts", full_gene_axis=True,
    strata_keys=["batch"], top_k=3,
)
assert result.effects.shape == (8, 60)
assert len(result.nominations) == 2 * 2 * 3 * 3
print(adata.uns["data_identity"])
print(result.nominations.query("method == 'PertRank'").to_string(index=False))
out = Path("outputs/quickstart")
out.mkdir(parents=True, exist_ok=True)
result.nominations.to_csv(out / "nominations.csv", index=False)
