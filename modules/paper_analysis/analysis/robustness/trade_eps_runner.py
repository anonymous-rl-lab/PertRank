"""Portable archive-path adapter around the exact executed numerical runner."""
import os
from pathlib import Path
import trade_eps_runner_execution as _engine
if os.environ.get('PERT_RANK_STIM_ASSET_ROOT'):
    _engine.RUN=Path(os.environ['PERT_RANK_STIM_ASSET_ROOT']).resolve()
    _engine.SOURCES=_engine.RUN/'src'/'trade_frozen'
    _engine.R_EXECUTABLE=_engine.RUN/'environment'/'bin'/'R'
    _engine.RUNTIME_FILE=_engine.RUN/'environment'/'runtime_versions.json'
from trade_eps_runner_execution import fit_trade_batch,sha256,atomic_json
