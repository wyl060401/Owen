"""CPU regression tests for the current Small-only workflow."""
import argparse
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['unit'],default='unit')
p.add_argument('--output',type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
command=[sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-p','test_small_gptq.py','-v']
if a.output:
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'unit.log').open('w',encoding='utf-8') as log:
        code=subprocess.call(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
else:
    code=subprocess.call(command,cwd=root)
raise SystemExit(code)
