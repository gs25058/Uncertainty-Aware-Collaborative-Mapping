#!/usr/bin/env python
"""Scientific depth/mask/map figures; no synthetic camera PNGs in main runs."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
p=argparse.ArgumentParser();p.add_argument('--root',default='results/depth_model_20261106');p.add_argument('--out',default='reports/depth_model_20261106/figures');a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
q=np.load(root/'validation/step_representative.npz');fig,axs=plt.subplots(1,4,figsize=(15,3.7),constrained_layout=True)
for ax,key,title in zip(axs[:3],('ideal','observed','valid'),('Ideal visible Z [m]','Observed Z [m]','Valid measurement')):
 im=ax.imshow(q[key],cmap='gray' if key=='valid' else 'viridis',vmin=0 if key=='valid' else 1.95,vmax=1 if key=='valid' else 2.22);ax.set_title(title);fig.colorbar(im,ax=ax,fraction=.045)
im=axs[3].imshow((q['observed']-q['ideal'])*1000,cmap='coolwarm',vmin=-30,vmax=30);axs[3].set_title('Depth error [mm]');fig.colorbar(im,ax=axs[3],fraction=.045)
for ax in axs:ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Optimistic virtual sensor: fixed step fixture, not a physical D455 capture');fig.savefig(out/'depth_and_valid.png',dpi=150);plt.close(fig)
replay=root/'replay_full';fig,axs=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
for ax,backend in zip(axs,('ideal_depth','depth_error_model')):
 q=np.load(replay/backend/'final_map.npz');lab=q['labels'];ijk=q['ijk'];k=int(np.floor(1.25/.1))-ijk[2]
 extent=[ijk[0]*.1,(ijk[0]+lab.shape[0])*.1,ijk[1]*.1,(ijk[1]+lab.shape[1])*.1]
 ax.imshow(lab[:,:,k].T,origin='lower',extent=extent,cmap=ListedColormap(['#9e9e9e','#f5f5f5','#212121']),vmin=0,vmax=2,interpolation='nearest');ax.set_title(backend+' at recorded poses');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
fig.suptitle('Same recorded trajectory / new camera: gray unknown, white free, black occupied; z=1.25 m');fig.savefig(out/'recorded_map_comparison.png',dpi=150);plt.close(fig)
print(out)
