"""Run unchanged compact exporter with atomic NPZ publication."""
from pathlib import Path
import runpy,sys,shutil
import atomic_npz
atomic_npz.install()
runpy.run_path(str(Path(__file__).resolve().with_name('export_compact_fit.py')),run_name='__main__')
fit=Path(sys.argv[sys.argv.index('--fit-dir')+1])
out=Path(sys.argv[sys.argv.index('--out')+1])
certificate=fit/'ATOMIC_SERIALIZATION_INPUT_STABILITY.json'
if certificate.exists():shutil.copy2(certificate,out/certificate.name)
