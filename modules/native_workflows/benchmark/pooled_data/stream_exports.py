#!/usr/bin/env python3
"""Export each frozen role when its exact raw recovery is complete.

This only shortens output latency; it neither reads source data nor changes
roles, normalization, or method settings. export_inputs serializes and validates
existing outputs, so the original restartable recovery driver remains safe.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['D4_Rest','D4_Stim8hr']);a=p.parse_args()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for rep in range(1,5):
        ready=HERE/a.task/'raw/raw_batches'/f'input_group{rep}'/'COMPLETE.json'
        start=time.monotonic()
        while not ready.exists():
            if time.monotonic()-start>10800:raise TimeoutError(f'Raw role {rep} has not completed')
            time.sleep(5)
        print(json.dumps({'phase':'EXPORT_COMPLETE_RAW_ROLE','task':a.task,'rep':rep,'time':time.time()}),flush=True)
        subprocess.run([sys.executable,str(HERE/'export_inputs.py'),'--task',a.task,'--rep',str(rep)],check=True,env=env)
    print(json.dumps({'status':'STREAM_EXPORTS_COMPLETE','task':a.task}),flush=True)

if __name__=='__main__':main()
