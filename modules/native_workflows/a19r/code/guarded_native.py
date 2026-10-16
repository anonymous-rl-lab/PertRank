#!/usr/bin/env python3
"""In-process RSS/time guard; works when a parent sees only a transport proxy PID."""
import argparse,json,os,resource,runpy,sys,threading,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--record',required=True,type=Path)
    ap.add_argument('--rss-gib',required=True,type=float);ap.add_argument('--seconds',required=True,type=float)
    ap.add_argument('script');ap.add_argument('script_args',nargs=argparse.REMAINDER);a=ap.parse_args()
    start=time.monotonic();stop=threading.Event();state={'status':'RUNNING','guard':'native process self ru_maxrss and wall time','peak_RSS_bytes':0}
    def publish():
        a.record.write_text(json.dumps(state,indent=2)+'\n')
    def sample():
        while not stop.wait(.2):
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            state.update(peak_RSS_bytes=max(state['peak_RSS_bytes'],rss),seconds=time.monotonic()-start)
            if rss>a.rss_gib*(1<<30) or state['seconds']>a.seconds:
                state['status']='RSS_GUARD' if rss>a.rss_gib*(1<<30) else 'TIME_GUARD'
                publish();os._exit(125)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    sys.path.insert(0,str(Path(a.script).resolve().parent));sys.argv=[a.script,*a.script_args]
    try:runpy.run_path(a.script,run_name='__main__')
    finally:
        stop.set();thread.join()
        state.update(status='ENDED',peak_RSS_bytes=max(state['peak_RSS_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),seconds=time.monotonic()-start)
        publish()

if __name__=='__main__':main()
