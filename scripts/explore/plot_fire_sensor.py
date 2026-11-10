#!/usr/bin/env python3
"""Fixed-pose sensor comparison; simulator truth is used only for these figures."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from covor.explore.fire_world import FireWorld
from covor.explore.depth_observation import RayDepthSensor
p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_closed_loop_20261107');p.add_argument('--out',default='reports/fire_closed_loop_20261107/figures');a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cfg=json.load(open(root/'main/fixed_config.json'));b=FireWorld(0,cfg);arrays={};audits={}
for c in ('clean','strong'):
 with np.load(root/'sensor_validation_v2'/(c+'.npz')) as q:T=q['T'];z=q['Z'];valid=q['valid']
 s=RayDepthSensor(b,cfg,0,c);(rz,_,rv),audit=s.measure(T,0,0,audit=True);assert np.array_equal(z,rz,equal_nan=True);assert np.array_equal(valid,rv);arrays[c]=(z,valid);audits[c]=audit
ideal=np.where(audits['clean']['geometric'],audits['clean']['true_z'],np.nan)
fig,axs=plt.subplots(2,4,figsize=(15,6),constrained_layout=True)
items=[(ideal,'Ideal visible Z (m)','viridis',.6,6),(arrays['clean'][0],'Clean measured Z (m)','viridis',.6,6),(arrays['strong'][0],'Strong measured Z (m)','viridis',.6,6),(arrays['strong'][0]-arrays['clean'][0],'Strong - clean (m)','coolwarm',-.08,.08),(arrays['clean'][1],'Clean valid','gray',0,1),(arrays['strong'][1],'Strong valid','gray',0,1),((arrays['clean'][0]-ideal)*1000,'Clean error (mm)','coolwarm',-60,60),((arrays['strong'][0]-ideal)*1000,'Strong error (mm)','coolwarm',-60,60)]
for ax,(v,title,cm,lo,hi) in zip(axs.ravel(),items):
 im=ax.imshow(v,cmap=cm,vmin=lo,vmax=hi);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(im,ax=ax,fraction=.045)
fig.savefig(out/'source_local_depth.png',dpi=150);plt.close(fig)
field=s.fire_smoke.field;r=s.fire_smoke.res;lab,ijk=b.labels(r);extent=[ijk[0]*r,(ijk[0]+field.shape[0])*r,ijk[1]*r,(ijk[1]+field.shape[1])*r]
fig,axs=plt.subplots(1,b.levels,figsize=(6*b.levels,5),squeeze=False,constrained_layout=True)
for f,ax in enumerate(axs[0]):
 z=f*3.3+2.5;k=int(z/r)-ijk[2];im=ax.imshow(np.ma.masked_where(lab[:,:,k].T==2,field[:,:,k].T),origin='lower',extent=extent,cmap='magma',vmin=0,vmax=max(.1,float(field.max())));ax.set_facecolor('#7b7b7b')
 for fire in b.fires:
  if int(fire['floor'])==f:ax.plot(fire['position'][0],fire['position'][1],'cx',ms=8)
 ax.set(title=f'Fixed source field, floor {f+1}, z={z:.1f}m',xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=ax,label='Dimensionless simulated density')
fig.savefig(out/'source_local_smoke_field.png',dpi=150);plt.close(fig);print('sensor and source-field figures saved',flush=True)
