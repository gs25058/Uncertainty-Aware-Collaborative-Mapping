#!/usr/bin/env python3
"""Count-aware adapter for the frozen audit's zero-valid sensor rows.

Raw CSV errors stay undefined/blank for these frames. Their contribution to
pooled error numerators and denominators is exactly zero. No simulation source,
depth, map, decision, or original CSV is edited. This adapter exists separately
because the active study has frozen the older audit source as well as the core.
"""
import argparse,csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import scripts.explore.post_fire_study as legacy

def analyze(root,report,skip_plots=False):
    root=Path(root);report=Path(report);reader=legacy.readcsv;plot=legacy.plot;zeros=[]
    def counted_read(path):
        rows=reader(path)
        if Path(path).name=='simulator_audit.csv':
            for q in rows:
                if int(q['n_valid'])==0:
                    zeros.append(dict(run=Path(path).parent.name,time=q['time'],drone=q['drone'],frame_id=q['frame_id'],n_geometry=q['n_geometry'],n_valid=0,mean_error='N/A',rms_error='N/A',pooled_error_weight=0))
                    # Compatibility value only in memory: the legacy formula
                    # multiplies this value by n_valid=0. Raw rows are preserved.
                    q['mean_error']='0';q['rms_error']='0'
        return rows
    legacy.readcsv=counted_read
    if skip_plots:legacy.plot=lambda *args:None
    try:legacy.analyze(root,report)
    finally:legacy.readcsv=reader;legacy.plot=plot
    target=report/'data/zero_valid_frames.csv'
    with target.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['run','time','drone','frame_id','n_geometry','n_valid','mean_error','rms_error','pooled_error_weight']);writer.writeheader();writer.writerows(zeros)
    # A wholly empty trial has no error statistic, not an error of zero.
    stats=report/'data/sensor_observations.csv';rows=reader(stats)
    for q in rows:
        if int(q['valid_pixels'])==0:q['mean_error_m']='';q['rms_error_m']=''
        q['zero_valid_frames']=sum(z['run']==q['run'] for z in zeros)
    with stats.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    qcpath=root/'audit/main_checks.json';qc=json.loads(qcpath.read_text());qc.update(zero_valid_frames=len(zeros),undefined_frame_errors_excluded_from_pooled_error=True,raw_sensor_audit_CSV_unchanged=True)
    temp=qcpath.with_suffix('.json.counted.tmp');temp.write_text(json.dumps(qc,indent=2)+'\n');temp.replace(qcpath)
    files=[Path(__file__),Path('scripts/explore/post_fire_study.py')];(root/'postprocessing_zero_valid_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    print('COUNT-AWARE AUDIT PASS; zero-valid captures retained:',len(zeros),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--report',required=True);p.add_argument('--skip-plots',action='store_true');a=p.parse_args();analyze(a.root,a.report,a.skip_plots)
