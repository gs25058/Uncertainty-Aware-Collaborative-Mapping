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
    z_range=getattr(sd,'range_metric',None)=='optical_z'
    if not z_range:d/=np.linalg.norm(d,axis=1)[:,None]
    ray,vox=_dda_batch(T[:3,3],T[:3,3]+d*max_range,res);q=vox-ijk
    inside=((q>=0)&(q<np.array(lab.shape))).all(1);vals=np.full(len(q),255,np.uint8);vals[inside]=lab[tuple(q[inside].T)]
    distance=np.linalg.norm((vox+.5)*res-T[:3,3],axis=1)
    stop=np.full(len(d),np.inf);np.minimum.at(stop,ray[(vals==2)|(vals==255)],distance[(vals==2)|(vals==255)])
    keep=(vals==0)&(distance<stop[ray])&(distance>.3)
    if z_range:
        depth=(((vox+.5)*res-T[:3,3])@T[:3,:3])[:,2]
        keep&=(depth>=sd.cfg.z_min)&(depth<=sd.cfg.z_max)
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
    clearance=None
    if cfg['candidates'].get('min_view_clearance_m',0):
        # Viewpoint stand-off, not a larger vehicle: transit paths still use
        # the unchanged physical flight map. Account for the camera blind range.
        clearance=np.maximum(0.,ndimage.distance_transform_edt(lab==1)*res-res*np.sqrt(3)/2)
        frontier&=clearance>=cfg['candidates']['min_view_clearance_m']
    # Air waypoints sampled in reachable observed-free volume, NEVER true rooms.
    points=[];stride=4
    heights=cfg['candidates']['heights']
    if cfg['candidates'].get('height_mode')=='mission_floor_offsets':
        # Public mission floor intervals only; no true floors/stairs/rooms.
        # Lower viewing heights avoid repeatedly selecting ceiling-adjacent
        # poses whose near surfaces fall inside the sensor's blind range.
        floors=sorted({sum(goal['floor_z_bounds'])/2 for goal in prior['goals']})
        heights=[floor+offset for floor in floors for offset in heights]
    if cfg['candidates'].get('height_mode')=='observed_free':
        ks=np.flatnonzero(frontier.any(axis=(0,1)));stride_z=cfg['candidates']['height_stride_rows']
        heights=(ks[ks%stride_z==0]+ijk[2]+.5)*res
    for h in heights:
        k=int(np.floor(h/res))-ijk[2]
        if 0<=k<lab.shape[2]:
            for i,j in np.argwhere(frontier[::stride,::stride,k])*stride:
                points.append(flight.world((i,j,k)))
    for state in states:
        if clearance is not None and state.get('failed'):continue
        if not flight.point_clear(state['p']):continue
        if clearance is not None and points and clearance[tuple(flight.index(state['p']))]<cfg['candidates']['min_view_clearance_m']:continue
        points.append(np.asarray(state['p']))
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
        if clearance is not None:raw[-1]['observed_view_clearance_m']=float(clearance[tuple(q)])
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
    interest=interest_mask(lab,ijk,res,prior)
    for sid,goals in sorted(analysis['impact'].items()):
        (feet,nf),(body,nb)=g.region_observation_cells(sid,max_cells=limit)
        foot_points=None
        if g.xy_padding and len(feet):
            nodes=g.ids[tuple(feet.T)];foot_points=np.c_[g.node_xy[nodes],g.foot_z[tuple(feet.T)]-g.foot_contact_offset]
        offset=getattr(g,'map_offset',np.zeros(3,int));feet=feet+offset;body=body+offset
        segments[sid]=(feet,body,goals,foot_points)
    scored=[]
    for c in pool:
        c['pitch_program']='common_6s_floor_upper_scan'
        c['pitch']=-.65
        T=camera_pose(np.array(c['position']),c['yaw'],c['pitch'],cfg['sensor']['baseline'])
        views=[camera_pose(np.array(c['position']),c['yaw'],pitch,cfg['sensor']['baseline']) for pitch in cfg['candidates']['pitches']]
        unknown=set().union(*(predicted_unknown(lab,ijk,res,v,sd,cfg['sensor']['range'],interest) for v in views));targets={};contrib=[];evidence={}
        for sid,(feet,body,goals,foot_points) in segments.items():
            f=max((visible_fraction(feet,lab,ijk,res,v,sd,True,points_override=foot_points,max_range_m=cfg['sensor']['range']) for v in views),key=lambda x:x['fraction']);b=max((visible_fraction(body,lab,ijk,res,v,sd,max_range_m=cfg['sensor']['range']) for v in views),key=lambda x:x['fraction'])
            value=min(f['fraction'],b['fraction']);contrib.append(dict(segment=sid,goals=goals,foot=f,body=b,value=value))
            for goal in goals:targets[goal]=max(targets.get(goal,0),value)
            if value>0:
                for cell in feet:
                    q=tuple(cell);needed=max(0,2-int(layer['votes'][q]));evidence[int(np.ravel_multi_index(q,lab.shape))]=(value,min(2,needed))
        scored.append(dict(candidate=c,unknown=unknown,targets=targets,evidence=evidence,contributions=contrib))
    if cfg['utility'].get('entry_objective','legacy')!='legacy':
        from .entry_objective import augment
        scored=augment(scored,analysis,lab,layer,ijk,res,sd,cfg,prior)
    return scored


def marginal(item,policy,allocated,cfg,entry_active=False,*,remaining=None,cost=None):
    used=set().union(*(x['unknown'] for x in allocated)) if allocated else set()
    u=len(item['unknown']-used)/cfg['utility']['unknown_normalizer']
    fallback=not entry_active
    if policy=='unknown' or fallback:return u,policy=='entry'
    if cfg['utility'].get('entry_objective')=='completion_opportunity_v1':
        return sum(max(0.,v-max([q['targets'].get(g,0.) for q in allocated]+[0.])) for g,v in item['targets'].items()),False
    gain=0.
    for target,value in item['targets'].items():
        prev=max([x['targets'].get(target,0) for x in allocated]+[0.]);gain+=max(0.,value-prev)
    if cfg['utility'].get('entry_objective','legacy')!='legacy':
        opt=cfg['utility']['task_progress']
        if opt.get('joint_evidence'):
            from .entry_objective import joint_marginal
            gain=joint_marginal(item,allocated,cfg)
        gain*=opt.get('task_gain_scale',1.)
        gain+=opt.get('exploration_weight',.1)*u*item.get('view_novelty',1.)
        if opt.get('deadline_discount') and remaining is not None and cost is not None:
            dwell=cfg['capture_dwell'];gain*=float(np.clip((remaining-cost+dwell)/dwell,0.,1.))
        return gain,False
    # Complementary frame evidence is capped at the missing votes. Each action
    # contains two distinct sensor ticks; duplicate actions give no new support.
    extra=0.;den=max(1,len(item['evidence']))
    for cell,(value,needed) in item['evidence'].items():
        prev=max([x['evidence'].get(cell,(0,0))[0] for x in allocated]+[0.])
        extra+=max(0.,value-prev)*(min(needed,2)/2)
    return cfg['utility']['completion_weight']*gain+cfg['utility']['progress_weight']*extra/den,False
