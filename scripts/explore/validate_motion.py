#!/usr/bin/env python
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.motion import FlightMap,Trajectory,Hover,reserve,reservation_conflict
from covor.explore.runner import csv_write,write_json
cfg=json.load(open('configs/closed_loop_20261105.json'));c=cfg['motion'];out=Path('results/closed_loop_20261105/validation/crossing');out.mkdir(parents=True,exist_ok=True)
# Fully observed small crossing + narrow doorway, no target/truth data in planner.
lab=np.zeros((70,70,30),np.uint8);lab[2:68,2:68,2:28]=1;lab[34:36,2:68,2:28]=2;lab[34:36,30:40,7:23]=1
f=FlightMap(lab,np.zeros(3,int),.1,c)
s1=dict(p=np.array([2.,3.4,1.25]),yaw=0.,pitch=-.65);s2=dict(p=np.array([5.,3.4,1.25]),yaw=np.pi,pitch=-.65)
# Opposing traversal through same aperture. Distinct waiting endpoints permit
# serial transit. Both robots keep their arrival hover reserved.
p1=[s1['p'],np.array([3.5,3.4,1.25]),np.array([4.5,4.3,1.25])];p2=[s2['p'],np.array([3.5,3.4,1.25]),np.array([2.5,4.3,1.25])]
assert all(f.segment(a,b) for path in (p1,p2) for a,b in zip(path,path[1:]))
tr1=reserve(p1,s1,dict(yaw=0.,pitch=-.65),0,[Hover(s2['p'])],c);assert tr1
tr2=reserve(p2,s2,dict(yaw=np.pi,pitch=-.65),0,[tr1],c)
# If serial scheduling still cannot guarantee endpoint clearance, safe hover is
# an explicit accepted outcome; never permit crossing through each other.
assert tr2 is None or not reservation_conflict(tr2,[tr1],c)
if tr2 is None:
 outcome='safe_hover_then_replan';tstart=tr1.end+1
 tr2=reserve(p2,s2,dict(yaw=np.pi,pitch=-.65),tstart,[tr1],c)
else:outcome='waited_then_crossed'
assert tr2 is not None
rows=[]
for t in np.arange(0,max(tr1.end,tr2.end)+2,.05):
 a=tr1.sample(t);b=tr2.sample(t)
 assert np.linalg.norm(a-b)>=2*(c['radius']+c['margin'])
 for i,tr in enumerate([tr1,tr2]):
  p=tr.sample(t);assert f.point_clear(p)
  rows.append(dict(time=t,drone=i,x=p[0],y=p[1],z=p[2],speed=np.linalg.norm(tr.sample(t,1)),acceleration=np.linalg.norm(tr.sample(t,2))))
csv_write(out/'trajectories.csv',rows);write_json(out/'summary.json',dict(outcome=outcome,wait=tr2.wait,first=tr1.record(),second=tr2.record(),minimum_separation=min(np.linalg.norm(tr1.sample(t)-tr2.sample(t)) for t in np.arange(0,max(tr1.end,tr2.end)+2,.01)),collision=False,backend='FUEL native Polynomial + observed-FREE A* + timed reservations'))
print('crossing passed',outcome,'wait',tr2.wait)
