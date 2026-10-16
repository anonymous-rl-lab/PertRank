"""Small, offline tutorial fixtures with explicit source identities."""
from importlib.resources import files
import hashlib
import io
import json

import numpy as np


def load_example(name):
    """Return (archived_statistics, provenance) for ``tcell`` or ``replogle``.

    These are authentic saved sufficient statistics, not reconstructed cells.
    Full-axis geometry is retained with a subset of gene-effect columns.
    Every candidate in each archived task remains present. Loading performs
    a SHA-256 check, reads no network data and opens no evaluation reference.
    """
    names = {"tcell": "tcell_d3_rest", "replogle": "replogle_k562"}
    if name not in names:
        raise ValueError("name must be 'tcell' or 'replogle'")
    stem = names[name]
    root = files("pertrank").joinpath("data")
    metadata = json.loads(root.joinpath(stem + ".provenance.json").read_text())
    payload = root.joinpath(stem + ".npz").read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["fixture"]["sha256"]:
        raise ValueError("Tutorial fixture integrity check failed")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, metadata


def software_example(seed=2026):
    """Build a tiny synthetic AnnData for installation and interface checks.

    This example carries no biological performance or validation claim.
    It has 120 controls, 8 targets with 20 cells each and 60 measured genes.
    """
    import anndata as ad
    import pandas as pd
    rng = np.random.default_rng(seed)
    counts = rng.poisson(3, (280, 60)).astype(np.int32)
    for j in range(8):
        counts[120 + 20*j:140 + 20*j, 20 + j] += 4
    targets = np.array(["NTC"] * 120 + [f"T{j}" for j in range(8) for _ in range(20)])
    own = np.array(["NTC"] * 120 + [f"G{j}" for j in range(8) for _ in range(20)])
    obs = pd.DataFrame({"target_ids": targets, "target_gene_ids": own,
                        "is_control": targets == "NTC", "batch": "example"},
                       index=[f"cell{i}" for i in range(280)])
    adata = ad.AnnData(counts, obs=obs,
                      var=pd.DataFrame(index=[f"G{i}" for i in range(60)]))
    adata.uns["data_identity"] = "Synthetic software example; not experimental evidence"
    return adata
