"""Map-only candidate pool, occlusion predictions and marginal greedy utility."""
import numpy as np
from scipy import ndimage
from covor.occupancy import _dda_batch
from covor.entry.assignment import visible_fraction
from .sensing import camera_pose
from .motion import FlightMap,reserve,Hover


def predicted_unknown(lab,ijk,res,T,sd,max_range,interest=None):
    # Same deterministic sparse pixel budget in both utility functions.
    h,w=sd.calib.size[1],sd.calib.size[0]
    vv,uu=np.meshgrid(np.linspace(0,h-1,12),np.linspace(0,w-1,18),indexing='ij')
    d=np.c_[(uu.ravel()-sd.cx)/sd.fx,(vv.ravel()-sd.cy)/sd.fy,np.ones(uu.size)]@T[:3,:3].T
    d/=np.linalg.norm(d,axis=1)[:,None]
    ray,vox=_dda_batch(T[:3,3],T[:3,3]+d*max_range,res);q=vox-ijk
    inside=((q>=0)&(q<np.array(lab.shape))).all(1);vals=np.full(len(q),255,np.uint8);vals[inside]=lab[tuple(q[inside].T)]
    distance=np.linalg.norm((vox+.5)*res-T[:3,3],axis=1)
    stop=np.full(len(d),np.inf);np.minimum.at(stop,ray[(vals==2)|(vals==255)],distance[(vals==2)|(vals==255)])
    keep=(vals==0)&(distance<stop[ray])&(distance>.3)
    if interest is not None:
        allowed=np.zeros(len(q),bool);allowed[inside]=interest[tuple(q[inside].T)];keep&=allowed
    pts=q[keep]
    return set(map(int,np.ravel_multi_index(pts.T,lab.shape))) if len(pts) else set()


def interest_mask(lab,ijk,res,prior):
    lo,hi=np.array(prior['exploration_box']);x,y,z=np.ogrid[:lab.shape[0],:lab.shape[1],:lab.shape[2]]
    return ((x+ijk[0]+.5)*res>=lo[0])&((x+ijk[0]+.5)*res<=hi[0])&((y+ijk[1]+.5)*res>=lo[1])&((y+ijk[1]+.5)*res<=hi[1])&((z+ijk[2]+.5)*res>=lo[2])&((z+ijk[2]+.5)*res<=hi[2])

def candidates(lab,ijk,res,layer,flight,states,cfg,epoch,runseed,prior):
    cc,num=ndimage.label(flight.safe);comps=[]
    for state in states:
        q=flight.index(state['p'])
        if ((q>=0)&(q<np.array(lab.shape))).all():comps.append(cc[tuple(q)])
    reachable=flight.safe&np.isin(cc,[c for c in comps if c])
    interest=interest_mask(lab,ijk,res,prior)
    weak=(layer['votes']>0)&~layer['support']&(layer['evidence']>0)&interest
    frontier_distance=ndimage.distance_transform_edt(~((lab==0)&interest))*res
    frontier=reachable&(frontier_distance<=.8)
    # Air waypoints sampled in reachable observed-free volume, NEVER true rooms.
    points=[];stride=4
    for h in cfg['candidates']['heights']:
        k=int(np.floor(h/res))-ijk[2]
        if 0<=k<lab.shape[2]:
            for i,j in np.argwhere(frontier[::stride,::stride,k])*stride:
                points.append(flight.world((i,j,k)))
    points.extend(np.asarray(s['p']) for s in states if flight.point_clear(s['p']))
    if not points:return []
    points=np.unique(np.round(points,4),axis=0)
    # Spatial coverage, deterministic seed controls first anchor, no utility or GT.
    rng=np.random.default_rng(runseed*100003+epoch)
    order=[int(rng.integers(len(points)))];dist=np.linalg.norm(points-points[order[0]],axis=1)
    for _ in range(min(len(points),cfg['candidates']['max_total'])-1):
        idx=int(np.argmax(dist));order.append(idx);dist=np.minimum(dist,np.linalg.norm(points-points[idx],axis=1));dist[order]=-1
    raw=[];unk=(lab==0)&interest
    for idx in order:
        p=points[idx];q=flight.index(p);r=12
        lo=np.maximum(q-r,0);hi=np.minimum(q+r+1,lab.shape)
        cloud=np.argwhere(unk[tuple(slice(a,b) for a,b in zip(lo,hi))])+lo
        if len(cloud):
            xy=flight.world(cloud)-p
            # Four azimuth sectors, choose largest current unknown volume.
            az=np.arctan2(xy[:,1],xy[:,0]);counts=[int((abs((az-y+np.pi)%(2*np.pi)-np.pi)<np.pi/4).sum()) for y in (0,np.pi/2,np.pi,3*np.pi/2)]
            yaw=(0,np.pi/2,np.pi,3*np.pi/2)[int(np.argmax(counts))]
        else:yaw=np.pi/2
        pitch=cfg['candidates']['pitches'][(epoch+len(raw))%len(cfg['candidates']['pitches'])]
        raw.append(dict(position=p.tolist(),yaw=yaw,pitch=pitch,kind='frontier'))
    # Re-observation candidates face weak support or current uncertain segments;
    # existing candidates may be repurposed but their air positions stay safe.
    surf=np.argwhere(weak)
    if len(surf):
        n=min(cfg['candidates']['max_reobserve'],len(raw)//2)
        for c in raw[-n:]:
            p=np.array(c['position']);world=(surf+ijk+.5)*res
            d=np.linalg.norm(world-p,axis=1);dest=world[int(np.argmin(d))];delta=dest-p
            c.update(yaw=float(np.arctan2(delta[1],delta[0])),pitch=float(np.clip(np.arctan2(delta[2],np.linalg.norm(delta[:2])),-.85,.25)),kind='reobserve')
    for i,c in enumerate(raw):c['id']=f'c{i:03d}'
    return raw[:cfg['candidates']['max_total']]


def score_pool(pool,analysis,lab,layer,ijk,res,sd,cfg,prior):
    g=analysis['graph'];segments={};limit=cfg['candidates']['observation_samples']
    for sid,goals in sorted(analysis['impact'].items()):
        (feet,nf),(body,nb)=g.region_observation_cells(sid,max_cells=limit)
        segments[sid]=(feet,body,goals)
    scored=[]
    for c in pool:
        c['pitch_program']='common_6s_floor_upper_scan'
        c['pitch']=-.65
        T=camera_pose(np.array(c['position']),c['yaw'],c['pitch'],cfg['sensor']['baseline'])
        views=[camera_pose(np.array(c['position']),c['yaw'],pitch,cfg['sensor']['baseline']) for pitch in cfg['candidates']['pitches']]
        unknown=set().union(*(predicted_unknown(lab,ijk,res,v,sd,cfg['sensor']['range'],interest_mask(lab,ijk,res,prior)) for v in views));targets={};contrib=[];evidence={}
        for sid,(feet,body,goals) in segments.items():
            f=max((visible_fraction(feet,lab,ijk,res,v,sd,True,max_range_m=cfg['sensor']['range']) for v in views),key=lambda x:x['fraction']);b=max((visible_fraction(body,lab,ijk,res,v,sd,max_range_m=cfg['sensor']['range']) for v in views),key=lambda x:x['fraction'])
            value=min(f['fraction'],b['fraction']);contrib.append(dict(segment=sid,goals=goals,foot=f,body=b,value=value))
            for goal in goals:targets[goal]=max(targets.get(goal,0),value)
            if value>0:
                for cell in feet:
                    q=tuple(cell);needed=max(0,2-int(layer['votes'][q]));evidence[int(np.ravel_multi_index(q,lab.shape))]=(value,min(2,needed))
        scored.append(dict(candidate=c,unknown=unknown,targets=targets,evidence=evidence,contributions=contrib))
    return scored


def marginal(item,policy,allocated,cfg,entry_active=False):
    used=set().union(*(x['unknown'] for x in allocated)) if allocated else set()
    u=len(item['unknown']-used)/cfg['utility']['unknown_normalizer']
    fallback=not entry_active
    if policy=='unknown' or fallback:return u,policy=='entry'
    gain=0.
    for target,value in item['targets'].items():
        prev=max([x['targets'].get(target,0) for x in allocated]+[0.]);gain+=max(0.,value-prev)
    # Complementary frame evidence is capped at the missing votes. Each action
    # contains two distinct sensor ticks; duplicate actions give no new support.
    extra=0.;den=max(1,len(item['evidence']))
    for cell,(value,needed) in item['evidence'].items():
        prev=max([x['evidence'].get(cell,(0,0))[0] for x in allocated]+[0.])
        extra+=max(0.,value-prev)*(min(needed,2)/2)
    return cfg['utility']['completion_weight']*gain+cfg['utility']['progress_weight']*extra/den,False
