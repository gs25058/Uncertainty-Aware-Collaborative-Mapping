#!/usr/bin/env python3
"""Metric diagram from the scene generator's exact landing/hole coordinates."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
ax.add_patch(Rectangle((-.3,-.7),3.3,1.25,fc='#b9d6c2'))
ax.add_patch(Rectangle((-.15,-.27),2.85,1.3,fc='#fafafa',ec='#999',ls='--'))
ax.add_patch(Rectangle((1.45,-.58),1.25,.58,fc='#84b79a',ec='#294d39'))
ax.add_patch(Rectangle((0,0),1.25,.29,fc='#91bad8',ec='#345d7b'))
ax.add_patch(Rectangle((0,.29),1.25,.29,fc='#91bad8',ec='#345d7b'))
ax.add_patch(Rectangle((0,-.27),1.25,.27,fc='#edb9ad',ec='#c34b36',hatch='///'))
ax.text(.625,-.135,'Missing support',ha='center',va='center',fontsize=10)
ax.text(2.075,-.29,'Previous flight\nupper landing',ha='center',va='center',fontsize=10)
ax.text(.625,.15,'Next flight: first tread',ha='center',va='center',fontsize=10)
ax.text(.625,-.55,'Corridor floor',ha='center',va='center',fontsize=10)
ax.annotate('',(.65,.1),(.65,-.5),arrowprops=dict(arrowstyle='->',color='#a53b29',lw=2))
ax.set(xlim=(-.3,3),ylim=(-.7,.65),xlabel='x relative to left stair (m)',ylabel='y relative to first tread (m)',title='2nd-floor plan: hole removes the bottom landing');ax.set_aspect('equal');ax.grid(alpha=.12)
bx.add_patch(Rectangle((-.65,3.08),.38,.22,fc='#84b79a',ec='#294d39'))
bx.add_patch(Rectangle((0,3.315),.29,.15,fc='#91bad8',ec='#345d7b'))
bx.add_patch(Rectangle((.29,3.48),.29,.15,fc='#91bad8',ec='#345d7b'))
bx.axhline(3.3,color='#777',ls=':',lw=.8)
bx.annotate('',(0,3.25),(-.27,3.25),arrowprops=dict(arrowstyle='<->',color='#b1412c'))
bx.text(-.135,3.20,'0.27 m gap',ha='center',color='#b1412c')
bx.annotate('',(.25,3.465),(.25,3.3),arrowprops=dict(arrowstyle='<->',color='#345d7b'))
bx.text(.27,3.38,'0.165 m\nriser',va='center',color='#345d7b')
bx.annotate('No floor at z=3.3 m\nRay hits ground floor',xy=(-.13,3.1),xytext=(-.55,2.85),arrowprops=dict(arrowstyle='->',color='#b1412c'),color='#b1412c')
bx.set(xlim=(-.65,.6),ylim=(2.75,3.75),xlabel='y relative to first tread (m)',ylabel='World z (m)',title='Section along the next flight');bx.grid(alpha=.15)
out=Path('reports/fire_closed_loop_20261107/figures/third_floor_landing_gap.png');out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=160);print(out)
