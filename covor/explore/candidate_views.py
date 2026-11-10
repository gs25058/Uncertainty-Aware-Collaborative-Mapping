"""Optional COMMON camera-yaw repair, separately ablated from utility changes.

Candidate positions/counts, permitted pitch program, and flight constraints are
unchanged. Every policy receives this identical map-only orientation procedure.
"""
import copy
import numpy as np
from scipy import ndimage
from .entry_objective import TaskContext
from .sensing import camera_pose


def refine_body_views(pool,analysis,lab,layer,ijk,res,sd,cfg,prior,*,flight=None,states=None):
    if not cfg['candidates'].get('body_reobserve_views'):return pool
    context=analysis.get('_candidate_task_context')
    if context is None:
        query_cfg=copy.deepcopy(cfg);query_cfg['utility']['task_progress']=cfg['candidates']['evidence_query_profile']
        context=TaskContext(analysis,lab,layer,ijk,res,query_cfg,prior);analysis['_candidate_task_context']=context
    queries=[q for q in context.queries if q.kind in ('boundary','entry')]
    if not queries:return pool
    if cfg['candidates'].get('evidence_viewpoint_positions'):
        return evidence_positions(pool,queries,context,sd,cfg,flight,states)
    result=[dict(c) for c in pool];n=min(cfg['candidates']['max_reobserve'],len(result))
    for candidate in result[-n:]:
        position=np.asarray(candidate['position']);old_yaw=candidate['yaw']
        def views(yaw):return [camera_pose(position,yaw,p,cfg['sensor']['baseline']) for p in cfg['candidates']['pitches']]
        before=max(context.observation_value(q,views(old_yaw),sd)[0] for q in queries)
        options=[]
        for q in queries:
            centre=q.body.mean(0);yaw=float(np.arctan2(centre[1]-position[1],centre[0]-position[0]))
            value,_=context.observation_value(q,views(yaw),sd);options.append((value,q.key,yaw))
        value,key,yaw=sorted(options,key=lambda x:(-x[0],x[1]))[0]
        if value>before+1e-8:
            candidate.update(yaw=yaw,kind='body_reobserve',orientation_source='common_observed_human_body_frontier',previous_yaw=old_yaw,target_query=key,visibility_before=before,visibility_after=value)
    return result


def evidence_positions(pool,queries,context,sd,cfg,flight,states):
    """Same finite safe viewpoint generator for every observation utility."""
    if flight is None or states is None:raise ValueError('observed flight map and current states required')
    labels,_=ndimage.label(flight.safe);components=[]
    for state in states:
        q=flight.index(state['p'])
        if ((q>=0)&(q<flight.safe.shape)).all():components.append(labels[tuple(q)])
    reachable=flight.safe&np.isin(labels,[c for c in components if c])
    proposals=[];budget=cfg['candidates'].get('evidence_proposal_budget',96);count=0
    for query in queries:
        centre=query.body.mean(0);surface=float(np.median(query.feet[:,2]));origin=centre.copy();origin[2]=surface
        for radius in (1.2,2.):
            for angle in (0.,np.pi/2,np.pi,-np.pi/2):
                for height in (.45,1.15,1.65):
                    if count>=budget:break
                    count+=1;p=origin+[radius*np.cos(angle),radius*np.sin(angle),height];q=flight.index(p)
                    if not ((q>=0)&(q<flight.safe.shape)).all() or not reachable[tuple(q)]:continue
                    p=flight.world(q);yaw=float(np.arctan2(centre[1]-p[1],centre[0]-p[0]))
                    views=[camera_pose(p,yaw,pitch,cfg['sensor']['baseline']) for pitch in cfg['candidates']['pitches']]
                    value,_=context.observation_value(query,views,sd)
                    if value>0:proposals.append((value,query.key,p,yaw))
                if count>=budget:break
            if count>=budget:break
        if count>=budget:break
    selected=[];quota=min(cfg['candidates']['max_reobserve'],max(1,len(pool)//2))
    for value,key,p,yaw in sorted(proposals,key=lambda x:(-x[0],x[1],tuple(x[2]),x[3])):
        if any(np.linalg.norm(p-np.asarray(c['position']))<.5 and abs((yaw-c['yaw']+np.pi)%(2*np.pi)-np.pi)<np.pi/3 for c in selected):continue
        selected.append(dict(position=p.tolist(),yaw=yaw,pitch=-.65,kind='body_reobserve',orientation_source='common_observed_evidence_viewpoint',target_query=key,predicted_enquiry_gain=value,raw_proposal_budget=budget))
        if len(selected)>=quota:break
    if not selected:return pool
    # Keep the common frontier quota; no utility may add private candidates.
    result=[dict(c) for c in pool[:max(0,len(pool)-len(selected))]]+selected
    for i,c in enumerate(result):c['id']=f'c{i:03d}'
    return result
