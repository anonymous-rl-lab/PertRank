"""Execute an unchanged Python entry point with atomic NPZ publication enabled."""
import runpy,sys
import atomic_npz
script=sys.argv.pop(1)
sys.argv[0]=script
atomic_npz.install()
runpy.run_path(script,run_name='__main__')
