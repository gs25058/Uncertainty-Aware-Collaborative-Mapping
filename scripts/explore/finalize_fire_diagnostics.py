#!/usr/bin/env python3
"""Run explanation-only probes after all policy selections are complete."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_closed_loop_20261107');p.add_argument('--report',default='reports/fire_closed_loop_20261107');a=p.parse_args();root=Path(a.root);report=Path(a.report)
while not (root/'post_complete.json').exists():time.sleep(30)
assert json.load(open(root/'post_complete.json'))['audit_passed']
commands=[['scripts/explore/analyze_fire_decisions.py','--root',str(root/'main'),'--out',str(report/'data')]]
for scene,condition in [(1,'clean'),(5,'strong')]:
 for policy in ('unknown','entry'):
  name=f'scene{scene}_seed0_{condition}_{policy}';commands.append(['scripts/explore/diagnose_fire_paths.py','--run',str(root/'main'/name),'--out',str(root/'diagnostics'/name)])
for command in commands:
 print('RUN',command,flush=True);subprocess.run([sys.executable,*command],check=True)
(root/'diagnostics_complete.json').write_text(json.dumps(dict(completed=True,commands=commands))+'\n');print('FINAL DIAGNOSTICS COMPLETE',flush=True)
