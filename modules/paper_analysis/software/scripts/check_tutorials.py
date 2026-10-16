"""Execute every plain Python tutorial cell in a fresh namespace.

This socket-free check needs only the installed PertRank package. Open the
notebooks in Jupyter for rich interactive tables. No cell is skipped.
"""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'tutorials'
for path in sorted(root.glob('*.ipynb')):
    notebook = json.loads(path.read_text())
    namespace = {'__name__': '__main__', 'display': lambda value: None}
    previous = os.getcwd()
    os.chdir(root)
    count = 0
    try:
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            count += 1
            source = cell['source']
            if isinstance(source, list): source = ''.join(source)
            exec(compile(source, f'{path.name}:cell{count}', 'exec'), namespace)
    finally:
        os.chdir(previous)
    print(f'PASS {path.name}: {count} complete code cells')
