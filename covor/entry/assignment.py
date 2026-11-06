"""Single observation allocation from current labels and proposal metadata only."""
import numpy as np

from covor.occupancy import _voxel_traverse, _dda_batch


def visible_fraction(cells,labels,ijk,res,T,sd,support_surface=False,points_override=None,max_range_m=5.):
    if len(cells)==0:return dict(fraction=0.,through_unknown=0.,n=0)
    points=(cells+ijk+.5)*res
    if support_surface:points[:,2]+=res*.5+1e-5
    if points_override is not None:points=np.asarray(points_override,float)
    raw=(points-T[:3,3])@T[:3,:3]
    rect=raw@sd.R1.T
    z=rect[:,2]
    with np.errstate(divide="ignore",invalid="ignore"):
        u=sd.fx*rect[:,0]/z+sd.cx;v=sd.fy*rect[:,1]/z+sd.cy
    distance=np.linalg.norm(points-T[:3,3],axis=1)
    w,h=sd.calib.size
    seen=(z>.3)&(distance<=max_range_m)&(u>=0)&(u<w)&(v>=0)&(v<h)
    uncertain=np.zeros(len(cells),bool)
    # Exact voxel DDA excludes the endpoint: an occupied support can be seen,
    # while any observed occupied voxel before it occludes it.
    ids=np.flatnonzero(seen)
    if len(ids):
        ray,vox=_dda_batch(T[:3,3],points[ids],res)
        ind=vox-ijk
        inside=((ind>=0)&(ind<np.array(labels.shape))).all(1)
        vals=np.zeros(len(ind),np.uint8);vals[inside]=labels[tuple(ind[inside].T)]
        blocked=np.zeros(len(ids),bool);missing=np.zeros(len(ids),bool)
        np.logical_or.at(blocked,ray,vals==2)
        np.logical_or.at(missing,ray,vals==0)
        seen[ids]&=~blocked;uncertain[ids]=missing
    return dict(fraction=float(seen.mean()),through_unknown=float((seen&uncertain).mean()),n=len(cells))


def unknown_visibility(labels,ijk,res,T,sd):
    w,h=sd.calib.size
    vv,uu=np.meshgrid(np.linspace(0,h-1,16),np.linspace(0,w-1,24),indexing="ij")
    dirs=np.stack(((uu-sd.cx)/sd.fx,(vv-sd.cy)/sd.fy,np.ones_like(uu)),-1).reshape(-1,3)
    dirs=dirs@sd.R1@T[:3,:3].T;dirs/=np.linalg.norm(dirs,axis=1)[:,None]
    ray,vox=_dda_batch(T[:3,3],T[:3,3]+dirs*5.,res)
    ind=vox-ijk;inside=((ind>=0)&(ind<np.array(labels.shape))).all(1)
    vals=np.full(len(ind),255,np.uint8);vals[inside]=labels[tuple(ind[inside].T)]
    d=dirs[ray]
    with np.errstate(divide="ignore",invalid="ignore"):
        t0=(vox*res-T[:3,3])/d;t1=((vox+1)*res-T[:3,3])/d
    small=np.minimum(t0,t1);small[d==0]=-np.inf
    entry=np.maximum(small.max(1),0.)
    stop=np.full(len(dirs),np.inf);np.minimum.at(stop,ray[vals==2],entry[vals==2])
    unknown=vox[(vals==0)&(entry>=.3)&(entry<stop[ray])]
    return len(np.unique(unknown,axis=0))


def choose(graph,analysis,candidates,calibrations,max_move_m=6.,max_cells=512,*,single_view=False,surface_cfg=None):
    """Candidates need ID, robot, T, move_cost; no depth/truth/callback API."""
    positive={sid:goals for sid,goals in analysis["impact"].items() if goals}
    if single_view:
        affected={goal for goals in positive.values() for goal in goals}
        for d in analysis['decisions']:
            if d['goal'] in affected:
                for sid in d['uncertain_segments']:positive.setdefault(sid,[])
    samples={sid:graph.region_observation_cells(sid,max_cells) for sid in positive}
    rows=[]
    for candidate in candidates:
        T=np.asarray(candidate["T"]);cost=float(candidate["move_cost"])
        eligible=cost<=max_move_m
        row=dict(id=int(candidate["id"]),robot=candidate["robot"],move_cost=cost,
                 eligible=eligible,goal_scores={},segments=[],priority=0.,unknown_voxels=0)
        if eligible:
            sd=calibrations[candidate["robot"]]
            row["unknown_voxels"]=unknown_visibility(graph.lab,graph.ijk,graph.res,T,sd)
            for sid,goals in positive.items():
                (feet,nf),(body,nb)=samples[sid]
                points=None
                if getattr(graph,'surface_geometry',False):
                    points=(feet+graph.ijk+.5)*graph.res
                    points[:,2]=graph.foot_z[tuple(feet.T)]+1e-5
                f=visible_fraction(feet,graph.lab,graph.ijk,graph.res,T,sd,support_surface=True,points_override=points)
                b=visible_fraction(body,graph.lab,graph.ijk,graph.res,T,sd)
                visibility=min(f["fraction"],b["fraction"])
                readiness=1.
                if single_view:
                    if graph.support_layer is None or surface_cfg is None:raise ValueError('single-view score requires current surface evidence and cfg')
                    # Can ONE additional independent frame meet the declared
                    # support evidence rule? Unseen cells need two frames and
                    # cannot become confirmed in one, however visible they are.
                    layer=graph.support_layer;ind=tuple(feet.T)
                    positions=points if points is not None else (feet+graph.ijk+.5)*graph.res
                    optical=((positions-T[:3,3])@T[:3,:3])@sd.R1.T
                    sigma=optical[:,2]**2/(sd.fx*sd.calib.baseline)
                    expected_weight=np.exp(-sigma**2/surface_cfg.depth_beta)
                    ready=graph.support[ind]|((layer['votes'][ind]+1>=surface_cfg.min_frames)&
                        (layer['evidence'][ind]+expected_weight>=surface_cfg.min_evidence))
                    readiness=float(ready.mean()) if len(ready) else 0.
                    visibility=min(f['fraction']*readiness,b['fraction'])
                row["segments"].append(dict(id=int(sid),goals=goals,support=f,body=b,
                       support_population=nf,body_population=nb,observability=visibility,
                       one_frame_evidence_readiness=readiness))
                for goal in goals:row["goal_scores"][goal]=max(row["goal_scores"].get(goal,0.),visibility)
            if single_view:
                by_id={r['id']:r['observability'] for r in row['segments']}
                # Serial uncertain sections must ALL be observable; the cut
                # impact score alone never promises completion after one frame.
                dependencies={d['goal']:d['uncertain_segments'] for d in analysis['decisions']}
                for goal in list(row['goal_scores']):
                    deps=dependencies.get(goal,[])
                    row['goal_scores'][goal]=min((by_id.get(sid,0.) for sid in deps),default=0.)
            row["priority"]=float(sum(row["goal_scores"].values()))
        rows.append(row)
    eligible=[r for r in rows if r["eligible"]]
    if not eligible:raise ValueError("no candidates satisfy common move limit")
    nearest=min(eligible,key=lambda r:(r["move_cost"],r["id"]))
    unknown=min(eligible,key=lambda r:(-r["unknown_voxels"],r["move_cost"],r["id"]))
    proposed=min(eligible,key=lambda r:(-r["priority"],r["move_cost"],r["id"]))
    fallback=proposed["priority"]<=0
    if fallback:proposed=nearest
    return dict(nearest=nearest["id"],unknown=unknown["id"],decision=proposed["id"]),rows,dict(
        fallback=fallback,reason="nearest: no positive dependency/visibility score" if fallback else "maximum distinct-goal observability sum",
        max_move_m=max_move_m,score_model='single-frame evidence readiness and serial visibility bottleneck' if single_view else 'cut impact visibility')
