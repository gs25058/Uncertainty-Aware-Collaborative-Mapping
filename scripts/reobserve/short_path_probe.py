#!/usr/bin/env python3
"""Separate a short-path evaluator control from sensor-map route generation.

Positive-control mesh annotations never enter the system graph. Perfect-pose
and ideal-depth arms are diagnostics, not deployable performance estimates.
"""
import argparse
import csv
from dataclasses import asdict
import json
import shutil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'scripts/reobserve',ROOT/'scripts/smoke'):sys.path.insert(0,str(p))
import gtsam  # before open3d
import cv2
import numpy as np
from covor.entry.surfaces import SupportSurfaces
from covor.entry.decision import DecisionGraph
from covor.entry.path_eval import ReferenceValidator,motion_path
from covor.entry import traverse as TV
from covor.occupancy import StereoDepth,load_stereo_calib
from common import write_json,write_csv,digest,make_builder,sparse_map,dense_map
from run import reference_scene
import repair as R


def prepare(out):
    cases=R.frozen();goal=next(g for g in cases['goals'] if g['old_id']=='south_0')
    ref=next(r for r in json.load(open(R.STUDY/'references.json')) if r['case_id']==goal['id'])
    assert ref['reference_state']=='reachable'
    config=dict(scene='corridor915f',seed=0,entry=cases['entry'],goal=goal,
        positive_xy=[[-2.35,y] for y in (-19.55,-19.45,-19.35,-19.25)],
        positive_posture=1,positive_width=1,positive_contact_offset_m=.005,
        positive_role='evaluator control only; not an input to planning or mapping',
        body=cases['body'],path_eval=cases['path_eval_cfg'],
        initial=json.load(open(R.SOURCE/'design.json'))['initial'],
        reference_goal_sha256=digest(R.STUDY/'references.json'),
        threshold_y_bounds=[-19.5,-19.4],threshold_z_bounds=[-.05,.15],
        threshold_first_view_rule='first initial frame in fixed robot/time order with planar support in the fixed threshold band',
        candidate_set='same 24 robot/frame IDs as archived study; ideal localization control; all 24 posthoc, move<=6m subset separate',
        max_move_m=6.)
    write_json(out/'case.json',config)


def clip_triangle_box(triangle,low,high):
    """Independent exact convex clipping for the positive-control envelope."""
    polygon=[p.copy() for p in triangle]
    for axis in range(3):
        for bound,sign in ((low[axis],1),(high[axis],-1)):
            result=[]
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                da=sign*(a[axis]-bound);db=sign*(b[axis]-bound)
                if da>=-1e-9:result.append(a)
                if (da>=0)!=(db>=0):result.append(a+(b-a)*da/(da-db))
            polygon=result
            if not polygon:return False
    return True


def positive(out):
    import trimesh
    cfg=json.load(open(out/'case.json'));scene,mc=reference_scene()
    gt=np.load(ROOT/'results/synth_corridor915f/gt_voxel_fs.npz')
    validator=ReferenceValidator(gt['labels'],gt['ijk_min'],float(gt['res']),scene)
    mesh=trimesh.load(mc['obj'],force='mesh',process=False);mesh.apply_transform(np.asarray(mc['T_mesh_world']))
    # Independently compute the floor height from triangle barycentric
    # coordinates, not from the occupancy graph or the path evaluator.
    triangles=np.asarray(mesh.triangles)
    floor=triangles[(triangles[:,:,2].min(1)<.3)&(triangles[:,:,2].max(1)>-.2)]
    path=[];annotations=[]
    for x,y in cfg['positive_xy']:
        t=floor[(floor[:,:,0].min(1)<=x)&(floor[:,:,0].max(1)>=x)&
                (floor[:,:,1].min(1)<=y)&(floor[:,:,1].max(1)>=y)]
        a=t[:,1]-t[:,0];b=t[:,2]-t[:,0];d=np.array([x,y])-t[:,0,:2]
        det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        with np.errstate(divide='ignore',invalid='ignore'):
            u=(d[:,0]*b[:,1]-d[:,1]*b[:,0])/det
            v=(a[:,0]*d[:,1]-a[:,1]*d[:,0])/det
        z=t[:,0,2]+u*a[:,2]+v*b[:,2]
        good=(abs(det)>1e-10)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(z>=-.2)&(z<=.3)
        assert good.any(),f'no independently annotated support at {x,y}'
        h=float(z[good].max());point=dict(xyz=[x,y,h+cfg['positive_contact_offset_m']],
            posture=cfg['positive_posture'],width=cfg['positive_width'],observed_support=True)
        path.append(point);annotations.append(dict(x=x,y=y,mesh_floor_z=h,covering_triangles=int(good.sum())))
    radius=cfg['body']['w_side']/2;height=cfg['body']['h_stoop']
    xyz=np.array([p['xyz'] for p in path]);low=xyz.min(0);high=xyz.max(0)
    low[:2]-=radius;high[:2]+=radius;low[2]+=cfg['body']['step'];high[2]+=height
    possible=triangles[((triangles.max(1)>=low)&(triangles.min(1)<=high)).all(1)]
    intersections=sum(clip_triangle_box(t,low,high) for t in possible)
    # The box encloses the complete swept torso disc, so no triangle inside
    # the box independently certifies this fixed posture's torso clearance.
    assert intersections==0,'fixed positive-control torso envelope intersects mesh'
    check=validator.validate(path,cfg['entry'],cfg['goal'])
    write_json(out/'positive_control.json',dict(path=path,mesh_annotations=annotations,
        independent_torso_envelope=dict(low=low,high=high,intersecting_triangles=intersections),validation=check,
        not_used_by_generator=True,method='triangle barycentric support + exact triangle/AABB swept torso certificate'))
    print('POSITIVE',check,flush=True)


def build_ideal(out):
    from covor.synth.render import DepthRenderer
    from covor import data as D
    from run_corridor import teammates
    cfg=json.load(open(out/'case.json'));scene,_=reference_scene()
    poses={r:dict(np.load(R.SOURCE/'poses'/f'{r}.npz')) for r in cfg['initial']}
    truth={r:dict(t=p['t'],p_body=(p['T_truth']@np.linalg.inv(D.load_body_T_cam(r,0)))[:,:3,3]) for r,p in poses.items()}
    builders={name:make_builder() for name in ('true','estimated')}
    surfaces={name:SupportSurfaces() for name in builders}
    for r,indices in cfg['initial'].items():
        renderer=DepthRenderer(scene,r);p=poses[r]
        for n,i in enumerate(indices):
            z,sz,v=renderer.depth_ideal(p['T_truth'][i]);pc,sp=renderer.sd.backproject(z,sz,v,downsample=4)
            directory=out/'ideal_depth';directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/f'{r}_{i:05d}.npz',Z=z,sigma_Z=sz,valid=v,T_render=p['T_truth'][i])
            for arm in builders:
                T=p['T_truth'][i] if arm=='true' else p['T'][i]
                mates=teammates(truth if arm=='true' else poses,r,p['t'][i])
                if v.sum()>=100:builders[arm].integrate_frame(T,float(p['tr'][i]),pc,sp,teammates=mates)
                surfaces[arm].integrate(renderer.sd,z,sz,v,T,mates)
            if (n+1)%25==0:print('IDEAL',r,n+1,flush=True)
    gt=np.load(ROOT/'results/synth_corridor915f/gt_voxel_fs.npz')
    for arm,b in builders.items():
        b.finalize();lab=dense_map(sparse_map(b),gt['ijk_min'],gt['labels'].shape)
        np.savez_compressed(out/f'map_ideal_{arm}.npz',labels=lab,ijk_min=gt['ijk_min'],res=gt['res'])
        b.tree.write(str(out/f'map_ideal_{arm}.ot').encode());surfaces[arm].save(out/f'surfaces_ideal_{arm}.npz')
        prior=ROOT/'results/diagnosis_20261104'/f'map_ideal_{"true_pose" if arm=="true" else "estimated_pose"}.npz'
        assert np.array_equal(lab,np.load(prior)['labels']),'ideal map replay differs from archived control'


def mode_evidence(g,node,validator):
    """Explain the existing body slab rule without changing its decisions."""
    i,j,k=node;z=float(g.foot_z[i,j,k]);result=[]
    for posture,height in enumerate((g.body.h_stand,g.body.h_stoop,g.body.h_crawl)):
        top=int(np.ceil((z+height+.02)/g.res-1e-9))-g.ijk[2]
        low=max(k+1,int(np.floor((z+g.body.step)/g.res+1e-9))-g.ijk[2])
        for width,diameter in enumerate((g.body.w,g.body.w_side)):
            radius=diameter/2;r=int(np.ceil(radius/g.res+.5))
            cells=[(i,j,kk) for kk in range(k+1,top)]
            for di in range(-r,r+1):
                for dj in range(-r,r+1):
                    distance=np.hypot(max(abs(di)*g.res-g.res*.5,0),max(abs(dj)*g.res-g.res*.5,0))
                    if distance<=radius+1e-9:cells.extend((i+di,j+dj,kk) for kk in range(low,top))
            ind=np.unique(np.asarray(cells,int).reshape(-1,3),axis=0)
            inside=((ind>=0)&(ind<g.lab.shape)).all(1)
            values=np.zeros(len(ind),np.uint8);values[inside]=g.lab[tuple(ind[inside].T)]
            occupied=ind[inside&(values==2)];unknown=ind[(values==0)|~inside]
            xyz=[float((i+g.ijk[0]+.5)*g.res),float((j+g.ijk[1]+.5)*g.res),z]
            check=validator.body_clear(xyz,posture,width)
            result.append(dict(posture=posture,width=width,n_body_cells=len(ind),n_free=int((values==1).sum()),
                n_occupied=len(occupied),n_unknown=len(unknown),
                observed_body_clear=len(occupied)==0 and len(unknown)==0,
                geometric_possibility=len(occupied)==0,reference_body_status=check[0],reference_reason=check[1],
                occupied_world_ijk=(occupied[:12]+g.ijk).tolist(),unknown_world_ijk=(unknown[:12]+g.ijk).tolist(),
                first_reference_failure=check[2]))
    if g.support[i,j,k]:assert any(m['observed_body_clear'] for m in result)==bool(g.confirmed[i,j,k])
    return result


def route_query(g,entry,goal):
    """Current-map single-goal decision, with the same verdict definitions."""
    starts=g.mission_nodes(entry,True);targets=g.mission_nodes(goal,True)
    confirmed=g.search(starts,True,target_nodes=targets)
    cp,_=g.path(confirmed,targets)
    if cp:return dict(goal=goal['id'],verdict='passable',reason='observed_support_and_body_path',path=cp,possible_path=[],uncertain_segments=[])
    potential=g.mission_nodes(goal)
    possible=g.search(g.mission_nodes(entry),target_nodes=potential)
    pp,deps=g.path(possible,potential);proof=None
    if pp:reason='short_uncertain_connections'
    elif not starts:reason='entry_not_observed_in_fixed_region'
    elif not g._region_inside(goal):reason='goal_outside_evaluation_scope'
    else:
        proof=g.blockage_proof(entry,goal)
        reason=proof['reason'] if proof else 'insufficient_coverage_or_possible_scope_escape'
    return dict(goal=goal['id'],verdict='blocked' if proof else 'unverified',reason=reason,
        path=[],possible_path=pp,uncertain_segments=deps,blockage_certificate=proof)


def diagnose(out):
    cfg=json.load(open(out/'case.json'));scene,_=reference_scene()
    gt=np.load(ROOT/'results/synth_corridor915f/gt_voxel_fs.npz')
    validator=ReferenceValidator(gt['labels'],gt['ijk_min'],float(gt['res']),scene)
    annotations=json.load(open(out/'positive_control.json'))['mesh_annotations']
    summaries=[];traces=[];mode_rows=[]
    for arm in ('true','estimated'):
        q=np.load(out/f'map_ideal_{arm}.npz');s=SupportSurfaces.load(out/f'surfaces_ideal_{arm}.npz')
        layer=s.dense(q['ijk_min'],q['labels'].shape)
        g=DecisionGraph(q['labels'],q['ijk_min'],float(q['res']),support_layer=layer,surface_geometry=True)
        search=g.search(g.mission_nodes(cfg['entry'],True),True)
        path,_=g.path(search,g.mission_nodes(cfg['goal'],True));path=motion_path(path)
        check=validator.validate(path,cfg['entry'],cfg['goal']) if path else dict(status='not_proposed',reason='no_confirmed_path')
        write_json(out/f'generated_{arm}.json',dict(path=path,validation=check))
        first=None
        for n,a in enumerate(annotations):
            ij=np.floor(np.array([a['x'],a['y']])/.1).astype(int)-q['ijk_min'][:2]
            i,j=ij;h=a['mesh_floor_z'];rows=np.arange(q['labels'].shape[2])
            physical=(rows+q['ijk_min'][2]+1)*.1
            candidates=rows[(physical>=cfg['entry']['floor_z_bounds'][0])&(physical<=cfg['entry']['floor_z_bounds'][1])]
            if len(candidates):k=int(min(candidates,key=lambda k:(not layer['support'][i,j,k],abs(float(layer['height'][i,j,k])-h) if np.isfinite(layer['height'][i,j,k]) else 1e3,k)))
            else:k=0
            support=bool(g.support[i,j,k]);z=float(g.foot_z[i,j,k]);posture=int(g.cp[i,j,k]);possible=int(g.pp[i,j,k])
            stage=None
            if not support:stage='support_evidence'
            elif abs(z-(h+.005))>.075:stage='foot_height'
            elif posture<0:stage='body_space' if possible>=0 else 'possible_posture'
            else:
                node=int(g.ids[i,j,k]);outgoing=list(g.edges(node,True)) if node>=0 else []
                if n<len(annotations)-1 and not outgoing:stage='next_support_connection'
            modes=mode_evidence(g,(i,j,k),validator)
            mode_rows.extend(dict(arm=arm,path_index=n,**m) for m in modes)
            record=dict(arm=arm,index=n,xy=[a['x'],a['y']],world_ijk=(np.array([i,j,k])+q['ijk_min']).tolist(),
                mesh_floor_z=h,support=support,votes=int(layer['votes'][i,j,k]),evidence=float(layer['evidence'][i,j,k]),
                surface_height=float(layer['height'][i,j,k]) if np.isfinite(layer['height'][i,j,k]) else None,foot_z=z,confirmed_posture=posture,possible_posture=possible,
                failure_stage=stage,volume_label=int(g.lab[i,j,k]))
            traces.append(record)
            if first is None and stage:first=record
        summaries.append(dict(arm=arm,verdict='passable' if path else 'unverified',path_status=check['status'],
            n_path_points=len(path),entry_confirmed_nodes=len(g.mission_nodes(cfg['entry'],True)),
            confirmed_reached_nodes=len(search['best']),first_failure=first))
        write_json(out/f'trace_{arm}.json',dict(summary=summaries[-1],trace=[t for t in traces if t['arm']==arm],
            body_modes=[m for m in mode_rows if m['arm']==arm]))
        print('DIAGNOSE',summaries[-1],flush=True)
    write_json(out/'stage1_summary.json',summaries);write_csv(out/'stage1_trace.csv',traces)
    write_csv(out/'stage1_body_modes.csv',mode_rows)


def threshold_case(out):
    """A controlled floor-depth withholding intervention, not GT map seeding.

    Integrate the same 300 real simulated observations. In a fixed floor strip,
    retain endpoints in exactly one INITIAL view and mark other such depth
    pixels invalid. Every other ray goes through the unchanged map pipeline.
    """
    from run_corridor import teammates
    from covor import data as D
    cfg=json.load(open(out/'case.json'))
    first=next(r for r in json.load(open(out/'stage1_summary.json')) if r['arm']=='true')
    if first['path_status']!='valid':
        write_json(out/'stage2_status.json',dict(executed=False,reason='stage1_system_path_not_valid'))
        return
    route=json.load(open(out/'generated_true.json'))['path']
    lo,hi=cfg['threshold_y_bounds'];zlo,zhi=cfg['threshold_z_bounds'];row=int(np.floor((lo+hi)*.5/.1))
    target=next(p for p in route if int(np.floor(p['xyz'][1]/.1))==row)
    key=tuple(np.floor((np.array(target['xyz'])-[0,0,.005])/.1).astype(int))
    poses={r:dict(np.load(R.SOURCE/'poses'/f'{r}.npz')) for r in cfg['initial']}
    truth={r:dict(t=p['t'],p_body=(p['T_truth']@np.linalg.inv(D.load_body_T_cam(r,0)))[:,:3,3]) for r,p in poses.items()}
    calibrations={r:StereoDepth(load_stereo_calib(r)) for r in poses}
    keep=None
    for r,indices in cfg['initial'].items():
        sd=calibrations[r];p=poses[r]
        for i in indices:
            q=np.load(out/'ideal_depth'/f'{r}_{i:05d}.npz');s=SupportSurfaces()
            s.integrate(sd,q['Z'],q['sigma_Z'],q['valid'],p['T_truth'][i],teammates(truth,r,p['t'][i]))
            if key in s.cells and s.cells[key][0]>0:
                keep=dict(robot=r,index=i,evidence=float(s.cells[key][0]));break
        if keep:break
    assert keep is not None,'fixed threshold has no initial support observation'
    builder=make_builder();surface=SupportSurfaces();frames=[]
    for r,indices in cfg['initial'].items():
        sd=calibrations[r];p=poses[r]
        for n,i in enumerate(indices):
            q=np.load(out/'ideal_depth'/f'{r}_{i:05d}.npz');z=q['Z'].copy();sz=q['sigma_Z'].copy();v=q['valid'].copy()
            vv,uu=np.mgrid[:z.shape[0],:z.shape[1]]
            rect=np.stack(((uu-sd.cx)*z/sd.fx,(vv-sd.cy)*z/sd.fy,z),-1)
            world=(rect@sd.R1)@p['T_truth'][i,:3,:3].T+p['T_truth'][i,:3,3]
            strip=(world[:,:,1]>=lo)&(world[:,:,1]<hi)&(world[:,:,2]>=zlo)&(world[:,:,2]<=zhi)&v
            retained=(r==keep['robot'] and i==keep['index'])
            removed=0 if retained else int(strip.sum())
            if not retained:z[strip]=np.nan;sz[strip]=np.nan;v[strip]=False
            pc,sp=sd.backproject(z,sz,v,downsample=4);mates=teammates(truth,r,p['t'][i])
            if v.sum()>=100:builder.integrate_frame(p['T_truth'][i],0,pc,sp,teammates=mates)
            surface.integrate(sd,z,sz,v,p['T_truth'][i],mates)
            frames.append(dict(robot=r,index=i,retained_threshold_view=retained,withheld_pixels=removed))
            if (n+1)%25==0:print('THRESHOLD',r,n+1,flush=True)
    base=np.load(out/'map_ideal_true.npz');lab=dense_map(sparse_map(builder),base['ijk_min'],base['labels'].shape)
    np.savez_compressed(out/'threshold_initial.npz',labels=lab,ijk_min=base['ijk_min'],res=base['res'])
    builder.tree.write(str(out/'threshold_initial.ot').encode());surface.save(out/'threshold_surfaces.npz')
    g=DecisionGraph(lab,base['ijk_min'],float(base['res']),support_layer=surface.dense(base['ijk_min'],lab.shape),surface_geometry=True)
    decision=g.decide(cfg['entry'],[cfg['goal']],True)
    d=decision['decisions'][0];positive=surface.cells.get(key,np.zeros(5))
    dependencies=d['uncertain_segments']
    outside=[p for p in d['possible_path'] if not lo<=p['xyz'][1]<hi and not p['confirmed']]
    accepted=bool(d['verdict']=='unverified' and bool(d['possible_path']) and len(dependencies)==1
        and bool(g.mission_nodes(cfg['entry'],True)) and not outside and positive[3]==1)
    write_json(out/'threshold_case.json',dict(accepted=accepted,retained_initial_view=keep,target_support_key=list(map(int,key)),
        target_statistics=positive,initial_decision=d,n_uncertain_segments=len(dependencies),
        unconfirmed_path_nodes_outside_strip=outside,case_type='controlled depth endpoint withholding; no truth labels inserted into map',
        inputs_unchanged_except_fixed_initial_depth_strip=True))
    write_csv(out/'threshold_input_order.csv',frames)
    print('THRESHOLD_CASE',accepted,d['verdict'],'dependencies',dependencies,'target stats',positive,flush=True)


def select(out):
    """Current map + ideal-localization metadata only; no candidate depth API."""
    from covor.entry.assignment import choose
    cfg=json.load(open(out/'case.json'));fixture=json.load(open(out/'threshold_case.json'))
    if fixture['accepted'] is not True:raise ValueError('fixed case does not isolate one once-observed uncertain threshold')
    old=json.load(open(R.SOURCE/'design.json'));candidates=[]
    for c in old['candidates']:
        r=c['robot'];p=np.load(R.SOURCE/'poses'/f'{r}.npz');T=p['T_truth'][c['index']]
        reference=p['T_truth'][cfg['initial'][r][-1],:3,3]
        assert c['index'] not in cfg['initial'][r]
        candidates.append(dict(c,T=T,move_cost=float(np.linalg.norm(T[:3,3]-reference)),
            pose_scope='perfect-localization diagnostic catalog; offline simulator poses known before depth'))
    assert len(candidates)==24
    q=np.load(out/'threshold_initial.npz');s=SupportSurfaces.load(out/'threshold_surfaces.npz')
    graph=DecisionGraph(q['labels'],q['ijk_min'],float(q['res']),support_layer=s.dense(q['ijk_min'],q['labels'].shape),surface_geometry=True)
    analysis=graph.decide(cfg['entry'],[cfg['goal']],True)
    calibrations={r:StereoDepth(load_stereo_calib(r)) for r in cfg['initial']}
    picks,scores,reason=choose(graph,analysis,candidates,calibrations,cfg['max_move_m'])
    readiness,ready_scores,ready_reason=choose(graph,analysis,candidates,calibrations,cfg['max_move_m'],single_view=True,surface_cfg=s.cfg)
    picks=dict(nearest=picks['nearest'],unknown=picks['unknown'],decision_impact=picks['decision'],decision_readiness=readiness['decision'])
    write_json(out/'selection.json',dict(picks=picks,scores=scores,readiness_scores=ready_scores,reason=reason,
        readiness_reason=ready_reason,candidates=candidates,impact=analysis['impact'],
        candidate_observation_access=False,reference_geometry_access=False,
        initial_map_sha256=digest(out/'threshold_initial.npz'),initial_surface_sha256=digest(out/'threshold_surfaces.npz'),
        case_sha256=digest(out/'threshold_case.json'),pose_scope='perfect localization oracle, not field selection performance'))
    print('SELECT',picks,flush=True)


def certify_threshold(out):
    """Re-check the saved fixture using current-map data only."""
    cfg=json.load(open(out/'case.json'));fixture=json.load(open(out/'threshold_case.json'))
    q=np.load(out/'threshold_initial.npz');s=SupportSurfaces.load(out/'threshold_surfaces.npz')
    graph=DecisionGraph(q['labels'],q['ijk_min'],float(q['res']),support_layer=s.dense(q['ijk_min'],q['labels'].shape),surface_geometry=True)
    decision=route_query(graph,cfg['entry'],cfg['goal']);lo,hi=cfg['threshold_y_bounds']
    key=tuple(map(int,fixture['target_support_key']));statistics=s.cells[key]
    outside=[p for p in decision['possible_path'] if not lo<=p['xyz'][1]<hi and not p['confirmed']]
    valid=bool(decision['verdict']=='unverified' and decision['possible_path'] and len(decision['uncertain_segments'])==1
        and graph.mission_nodes(cfg['entry'],True) and not outside and statistics[3]==1)
    fixture.update(accepted=valid,target_support_key=list(key),target_statistics=statistics,
        initial_decision=decision,n_uncertain_segments=len(decision['uncertain_segments']),
        unconfirmed_path_nodes_outside_strip=outside)
    write_json(out/'threshold_case.json',fixture)
    print('CERTIFIED_THRESHOLD',valid,'once-observed support',key,statistics,flush=True)


def all_candidates(out):
    """After selection, add EACH candidate once to independent initial clones."""
    import octomap
    from covor.synth.render import DepthRenderer
    from covor import data as D
    from run_corridor import teammates
    cfg=json.load(open(out/'case.json'));selection=json.load(open(out/'selection.json'))
    assert not selection['candidate_observation_access'] and not selection['reference_geometry_access']
    assert digest(out/'threshold_initial.npz')==selection['initial_map_sha256']
    assert digest(out/'threshold_surfaces.npz')==selection['initial_surface_sha256']
    selection_hash=digest(out/'selection.json');q=np.load(out/'threshold_initial.npz')
    scene,_=reference_scene();gt=np.load(ROOT/'results/synth_corridor915f/gt_voxel_fs.npz')
    validator=ReferenceValidator(gt['labels'],gt['ijk_min'],float(gt['res']),scene)
    renderers={r:DepthRenderer(scene,r) for r in cfg['initial']}
    poses={r:dict(np.load(R.SOURCE/'poses'/f'{r}.npz')) for r in cfg['initial']}
    truth={r:dict(t=p['t'],p_body=(p['T_truth']@np.linalg.inv(D.load_body_T_cam(r,0)))[:,:3,3]) for r,p in poses.items()}
    rows=[]
    for c in selection['candidates']:
        r=c['robot'];i=c['index'];T=poses[r]['T_truth'][i];sd=renderers[r].sd
        z,sz,v=renderers[r].depth_ideal(T);pc,sp=sd.backproject(z,sz,v,downsample=4)
        directory=out/'candidate_observations';directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/f"candidate{c['id']}.npz",Z=z,sigma_Z=sz,valid=v,P_cam=pc,sZ=sp,T=T,robot=r,index=i)
        builder=make_builder();builder.tree=octomap.OcTree.read(str(out/'threshold_initial.ot').encode())
        assert np.array_equal(dense_map(sparse_map(builder),q['ijk_min'],q['labels'].shape),q['labels'])
        mates=teammates(truth,r,poses[r]['t'][i]);surface=SupportSurfaces.load(out/'threshold_surfaces.npz')
        n_frames=surface.n_frames
        if v.sum()>=100:builder.integrate_frame(T,0,pc,sp,teammates=mates)
        surface.integrate(sd,z,sz,v,T,mates);assert surface.n_frames==n_frames+1
        lab=dense_map(sparse_map(builder),q['ijk_min'],q['labels'].shape)
        graph=DecisionGraph(lab,q['ijk_min'],float(q['res']),support_layer=surface.dense(q['ijk_min'],lab.shape),surface_geometry=True)
        d=route_query(graph,cfg['entry'],cfg['goal']);path=motion_path(d['path'])
        check=validator.validate(path,cfg['entry'],cfg['goal']) if path else dict(status='not_proposed',reason=d['reason'])
        directory=out/'candidate_maps';directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/f"candidate{c['id']}.npz",labels=lab,ijk_min=q['ijk_min'],res=q['res'])
        surface.save(directory/f"candidate{c['id']}_surfaces.npz")
        write_json(out/'candidate_paths'/f"candidate{c['id']}.json",dict(decision=d,path=path,validation=check))
        fixture=json.load(open(out/'threshold_case.json'));key=tuple(map(int,fixture['target_support_key']))
        local=tuple(np.asarray(key)-q['ijk_min'])
        failure_stage=('none' if check['status']=='valid' else
            'threshold_support_not_confirmed' if not graph.support[local] else
            'threshold_body_space_not_confirmed' if not graph.confirmed[local] else
            'other_connection_or_body_not_confirmed')
        rows.append(dict(id=c['id'],robot=r,index=i,move_cost=c['move_cost'],eligible=c['move_cost']<=cfg['max_move_m'],
            additional_observations=1,before='unverified',after=d['verdict'],path_status=check['status'],path_reason=check['reason'],
            valid_path_confirmed=check['status']=='valid',correct_blocked=False,reference='reachable',
            target_votes=float(surface.cells.get(key,np.zeros(5))[3]),target_evidence=float(surface.cells.get(key,np.zeros(5))[0]-surface.cells.get(key,np.zeros(5))[4]),
            failure_stage=failure_stage,n_path_points=len(path),first_failure=json.dumps(check.get('first_failure')),selected_by=','.join(p for p,cid in selection['picks'].items() if cid==c['id'])))
        print('CANDIDATE',c['id'],'eligible',rows[-1]['eligible'],d['verdict'],check['status'],flush=True)
        write_csv(out/'candidate_outcomes.csv',rows)
    assert digest(out/'selection.json')==selection_hash
    success=[r['id'] for r in rows if r['valid_path_confirmed']]
    eligible_success=[r['id'] for r in rows if r['valid_path_confirmed'] and r['eligible']]
    comparisons=[]
    for policy,cid in selection['picks'].items():
        result=next(r for r in rows if r['id']==cid)
        diagnosis=('succeeded' if result['valid_path_confirmed'] else
                   'selection_missed_feasible_one_view' if eligible_success else
                   'success_only_outside_move_limit' if success else 'no_one_view_success_in_this_24_candidate_set')
        comparisons.append(dict(policy=policy,selected_id=cid,diagnosis=diagnosis,**{k:v for k,v in result.items() if k!='id'}))
    write_csv(out/'policy_comparison.csv',comparisons)
    write_json(out/'stage2_status.json',dict(executed=True,n_candidates=len(rows),n_eligible=sum(r['eligible'] for r in rows),
        success_ids=success,eligible_success_ids=eligible_success,selection_unchanged_after_oracle=True,
        oracle_not_used_for_selection=True,scope='clean ideal depth + true poses; fixed withheld floor strip; reachable goal only'))


def contributors(out):
    """Posthoc provenance of the first estimated-height failure, not selection."""
    from run_corridor import teammates
    cfg=json.load(open(out/'case.json'));summary=json.load(open(out/'trace_estimated.json'))
    failure=summary['summary']['first_failure']
    if not failure:write_csv(out/'estimated_entry_contributors.csv',[]);return
    key=tuple(failure['world_ijk']);rows=[]
    poses={r:dict(np.load(R.SOURCE/'poses'/f'{r}.npz')) for r in cfg['initial']}
    for r,indices in cfg['initial'].items():
        sd=StereoDepth(load_stereo_calib(r));p=poses[r]
        for i in indices:
            q=np.load(out/'ideal_depth'/f'{r}_{i:05d}.npz');s=SupportSurfaces()
            s.integrate(sd,q['Z'],q['sigma_Z'],q['valid'],p['T'][i],teammates(poses,r,p['t'][i]))
            if key not in s.cells:continue
            a=s.cells[key];relative=p['T'][i,:3,:3]@p['T_truth'][i,:3,:3].T
            angle=np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
            rows.append(dict(robot=r,index=i,timestamp=float(p['t'][i]),weight=float(a[0]),surface_z=float(a[1]/a[0]),
                camera_dz=float(p['T'][i,2,3]-p['T_truth'][i,2,3]),rotation_error_deg=float(angle),
                mesh_floor_z=failure['mesh_floor_z']))
    write_csv(out/'estimated_entry_contributors.csv',rows)
    print('CONTRIBUTORS',rows,flush=True)


def report(out):
    from evaluation import metrics
    cfg=json.load(open(out/'case.json'));positive=json.load(open(out/'positive_control.json'))
    stage1=json.load(open(out/'stage1_summary.json'));stage2=json.load(open(out/'stage2_status.json'))
    fixture=json.load(open(out/'threshold_case.json'));selection=json.load(open(out/'selection.json'))
    outcomes=list(csv.DictReader(open(out/'candidate_outcomes.csv')))
    policies=list(csv.DictReader(open(out/'policy_comparison.csv')))
    assert len(outcomes)==24 and fixture['accepted'] is True
    assert positive['validation']['status']=='valid'
    assert not selection['candidate_observation_access'] and not selection['reference_geometry_access']
    assert digest(out/'threshold_case.json')==selection['case_sha256']
    assert len({int(r['id']) for r in outcomes})==24
    assert all(int(r['additional_observations'])==1 for r in outcomes)
    before=SupportSurfaces.load(out/'threshold_surfaces.npz');crossings=[];path_metrics=[]
    initial_unknown=[]
    for p in fixture['initial_decision']['possible_path']:
        if p['confirmed']:continue
        key=tuple(p['ijk']);a=before.cells[key]
        assert a[3]==1
        initial_unknown.append(dict(world_ijk=list(key),votes=float(a[3]),evidence=float(a[0]-a[4])))
    write_json(out/'initial_threshold_supports.json',initial_unknown)
    for row in outcomes:
        cid=int(row['id']);q=json.load(open(out/'candidate_paths'/f'candidate{cid}.json'))
        path=q['path'];p=np.array([point['xyz'] for point in path]).reshape(-1,3)
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()) if len(p) else None
        after=np.load(out/'candidate_maps'/f'candidate{cid}.npz')['labels'];base=np.load(out/'threshold_initial.npz')['labels']
        path_metrics.append(dict(id=cid,path_length_m=length,n_path_points=len(path),
            newly_known_voxels=int(((base==0)&(after!=0)).sum()),path_status=q['validation']['status']))
        if q['validation']['status']!='valid':continue
        surface=SupportSurfaces.load(out/'candidate_maps'/f'candidate{cid}_surfaces.npz');seen=set()
        for point in path:
            if not cfg['threshold_y_bounds'][0]<=point['xyz'][1]<cfg['threshold_y_bounds'][1]:continue
            key=tuple(point['ijk'])
            if key in seen:continue
            seen.add(key);a=before.cells.get(key,np.zeros(5));b=surface.cells[key]
            assert a[3]==1 and b[3]>=2 and b[0]-b[4]>=surface.cfg.min_evidence
            crossings.append(dict(candidate_id=cid,world_ijk=list(key),xyz=point['xyz'],
                before_votes=float(a[3]),before_evidence=float(a[0]-a[4]),after_votes=float(b[3]),after_evidence=float(b[0]-b[4])))
    write_csv(out/'candidate_path_metrics.csv',path_metrics)
    write_json(out/'successful_threshold_crossings.json',crossings)
    combined=[]
    for p in policies:
        metric=next(m for m in path_metrics if m['id']==int(p['selected_id']))
        row=dict(case_id=cfg['goal']['id'],reference_state='reachable',verdict=p['after'],path_status=p['path_status'])
        before_row=dict(row,verdict='unverified',path_status='not_proposed')
        combined.append(dict(p,path_length_m=metric['path_length_m'],newly_known_voxels=metric['newly_known_voxels'],
            **metrics([row],[before_row])))
    write_csv(out/'policy_summary.csv',combined)
    test_counts={}
    for name,prefix in (('tests','PASS '),('existing_tests','PASS '),('support_tests','PASS '),('traverse_tests','ok   ')):
        path=Path(f'/tmp/short_probe_{name}.log')
        if path.exists():
            content=path.read_text();assert 'Traceback' not in content
            test_counts[name]=sum(line.startswith(prefix) for line in content.splitlines())
    write_json(out/'validation.json',dict(test_counts=test_counts,total_passed=sum(test_counts.values()),
        candidate_count=24,independent_one_observation_budget=True,
        actual_threshold_crossings_have_one_prior_view=True,reference_goal_count=1))
    preserved=0
    for folder in ('smoke_20261103','decision_paths_20261103','diagnosis_20261104','fire_mapping_resume_20261104'):
        root=ROOT/'reports'/folder
        for name,expected in json.load(open(root/'artifacts.json')).items():
            assert digest(root/name)==expected['sha256'];preserved+=1
    files=[Path(__file__),ROOT/'covor/entry/decision.py',ROOT/'covor/entry/path_eval.py',ROOT/'covor/entry/surfaces.py',
           ROOT/'covor/entry/assignment.py',ROOT/'tests/test_short_path_probe.py']
    write_json(out/'integrity.json',dict(prior_report_artifacts_unchanged=preserved,
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in files},
        case_sha256=digest(out/'case.json'),selection_sha256=digest(out/'selection.json'),
        selection_not_modified_by_candidate_evaluation=stage2['selection_unchanged_after_oracle']))
    write_json(out/'complete.json',dict(complete=True,positive_control_status=positive['validation']['status'],
        stage1=stage1,stage2=stage2,goal=cfg['goal']['id'],one_observation_per_independent_candidate=True,
        no_positive_control_path_in_generator=True,no_evaluator_threshold_change=True))
    def table(rows,keys):
        lines=['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']
        for row in rows:
            vals=[]
            for key in keys:
                v=row.get(key,'')
                try:
                    x=float(v)
                    if np.isfinite(x) and not x.is_integer():v=f'{x:.4f}'
                except (ValueError,TypeError):pass
                vals.append(str(v))
            lines.append('| '+' | '.join(vals)+' |')
        return '\n'.join(lines)
    estimated=next(r for r in stage1 if r['arm']=='estimated');failure=estimated['first_failure']
    keep=fixture['retained_initial_view']
    stage1_rows=[dict(arm=r['arm'],verdict=r['verdict'],path_status=r['path_status'],n_path_points=r['n_path_points'],
        first_failure='none' if r['first_failure'] is None else r['first_failure']['failure_stage']) for r in stage1]
    pose_rows=list(csv.DictReader(open(out/'estimated_entry_contributors.csv')))
    report_text=f'''# 짧은 경로 생성/평가 분리와 한 번 재관측의 사후 상한 검사

## 고정 사례와 평가기 양성 대조

이전에 메시 검증으로 도달 가능한 10개 목표 중 가까운 `south_0`를 실행 전에 고정했다.
입구 (-2.35, -19.55), 목표 중심 (-2.35, -19.25), 직선 거리 0.30 m다.
기존 입구/목표 영역·층 허용 범위·인체 크기·평가 접촉 허용오차를 그대로 사용했다.

양성 대조는 경로 그래프에서 꺼낸 경로를 재활용하지 않았다. 고정한 네 XY 위치에서 메시
삼각형의 무게중심 좌표로 발판 높이를 별도 계산했고, 숙이기 1.40 m·옆으로 폭 0.45 m를 고정했다.
연속 몸통 원판을 포함하는 보수적인 swept AABB와 메시 삼각형의 정확한 클리핑으로 몸통 공간도
독립 확인했다. 지면은 Z=0.05006477 m, 발은 5 mm 위 Z=0.05506477 m였다.
평가기 결과는 **{positive['validation']['status']}**다. 이 경로는 `positive_control.json`에만 저장했고
지도 생성이나 시스템 탐색의 입력으로 넣지 않았다. 평가기의 좌표·접촉·충돌 문턱을 변경하지 않았다.
평가기는 기존 메시 발판·자세별 몸 공간·경로 사이 swept-body 검사를 그대로 수행한다.

## 1. 같은 관측에서 시스템이 스스로 생성한 경로

같은 초기 3대×100프레임, seed 0, 무처리 이상 깊이를 참 포즈에서 생성했다. 같은 깊이를
참 포즈와 기존 추정 포즈에 각각 적분했다. 기존 이상 깊이 대조 지도 두 개와 라벨 배열이
정확히 일치함도 확인했다. 표면층은 관측 평면으로만 생성했고 메시 발판 주석을 넣지 않았다.
탐색 입력은 현재 지도·표면층·고정 입구/목표뿐이다.

{table(stage1_rows,['arm','verdict','path_status','n_path_points','first_failure'])}

참 포즈에서 시스템이 스스로 만든 경로는 유효했다. 추정 포즈에서는 경로를 만들었으나 평가에서
거절됐다. **경로 생성 실패가 아니라 잘못된 높이의 경로 제안**이라는 차이가 중요하다.
첫 끊김은 `{failure['failure_stage']}`다. 입구의 지지 표는 {failure['votes']}개,
순 증거 {failure['evidence']:.6f}로 지지 확인 조건을 만족했지만, 표면 높이
{failure['surface_height']:.6f} m, 발 높이 {failure['foot_z']:.6f} m였다.
메시 지면 {failure['mesh_floor_z']:.6f} m보다 발이 약 10 cm 아래에서 시작해 몸이 바닥과 충돌했다.

`stage1_trace.csv`/`trace_*.json`은 발판 증거 → 발 높이 → 가능한 자세 → 몸 공간 → 다음 연결을
차례로 기록한다. `stage1_body_modes.csv`는 3자세×2폭의 free/occupied/unknown 셀 수와
처음 막힌 복셀, 독립 메시 몸 검증을 제공한다. 성공 경로는 `generated_true.json`,
거절 경로와 평가기의 첫 실패 위치는 `generated_estimated.json`에 있다.

잘못된 입구 발판에 기여한 초기 관측:

{table(pose_rows,['robot','index','weight','surface_z','camera_dz','rotation_error_deg'])}

이 관측은 모두 첫 드론 ifo001에서 왔다. 따라서 다음 우선순위는 포즈/공통 좌표계의 수직
일관성이다. 다만 다른 두 드론에 상수 Z 보정만 넣는 것으로 충분하다고 볼 수 없다.
기준 드론 자신의 국소 고도 및 자세 오차도 함께 확인해야 한다. 본 실행에서 새로운 높이 보정을
적용하거나 참 높이를 추정 지도에 주입하지 않았다.

## 2. 한 번 관측으로 바뀔 수 있는 단일 문턱 사례

참 포즈 경로의 성립을 확인한 뒤, 사전에 고정한 Y=[-19.5,-19.4) m, Z=[-0.05,0.15] m
바닥 띠에서 초기 깊이 끝점만 통제해 보류했다. 이 띠를 실제로 관측한 첫 초기 프레임
**{keep['robot']} index {keep['index']}**는 그대로 남겼다. 나머지 초기 영상에서는 해당 끝점 픽셀을
invalid 처리하고, 몸 공간을 포함한 다른 광선은 기존 파이프라인에 정상 적분했다.
이는 자연스럽게 발생한 연기 관측 사례가 아닌 **깊이 관측 누락을 통제한 사례**다.
GT 라벨로 확인 영역을 채우거나 관측 표 수를 수동 변경한 사례가 아니다.

초기 상태는 `unverified`, 입구는 확인됨, 가능한 경로의 미확인 국소 구간은 1개,
문턱 밖 미확인 경로 노드는 0개다. 대표 발판의 관측 표는 1개, 증거는 {keep['evidence']:.6f}다.
프레임별 가림 픽셀 수와 순서는 `threshold_input_order.csv`, 검증은 `threshold_case.json`에 있다.
최소 2프레임·순 증거 1.2·기존 몸/걸음 조건은 바꾸지 않았다.

### 후보 선택을 먼저 확정

기존 24개 드론/프레임 ID를 유지했다. 이 진단은 완벽한 위치추정 조건이므로 후보와 현재 위치에도
시뮬레이터 참 포즈를 사용했다. 이는 **오프라인 perfect-localization 대조**이며 현장 온라인 선택
성능으로 해석하지 않는다. 이동 비용은 각 드론의 마지막 초기 위치에서 후보까지의 직선 거리이고
공통 제한은 6 m다. 24개 중 제한 안 후보는 {stage2['n_eligible']}개다.

현재 지도와 카메라 시야만으로 `selection.json`을 먼저 저장했다. 후보 깊이·사후 결과·참 형상은
선택 함수의 입력이 아니다. 이후 24개 후보의 이상 깊이를 각각 생성해 **같은 초기 전체 log-odds
지도와 표면층의 독립 복제본**에 한 번씩 적분했다. 후보 결과를 점수 조정이나 재선택에 쓰지 않았다.
선택 파일의 해시가 사후 검사 뒤에도 바뀌지 않음을 확인했다.

### 사후 상한과 정책 결과

성공 후보는 전체 24개 중 **{len(stage2['success_ids'])}개: {stage2['success_ids']}**이고,
이동 제한 안에서도 **{len(stage2['eligible_success_ids'])}/{stage2['n_eligible']}개: {stage2['eligible_success_ids']}**다.
이들은 같은 목표 하나를 각각 확인한 대체 관측이지 서로 다른 목표가 아니다.
따라서 이 사례에서 한 번의 관측 예산 자체가 부족했던 것은 아니다.

{table(combined,['policy','selected_id','move_cost','after','path_status','path_length_m','newly_known_voxels','diagnosis'])}

모든 방식의 추가 관측 수는 1이다. 중요 구간 우선(`decision_impact`)과 단일 프레임 증거 대조
(`decision_readiness`)는 성공 후보를 골랐다. 최근접과 미관측 공간 우선은 미확인에 머물렀다.
이 사례에서는 **제안 정책이 성공 후보를 놓친 문제가 관찰되지 않았다.**
원래 단순 중요도 점수는 후보 2와 3이 동점이어서 이동 비용이 낮은 3을 골랐고,
증거 대조는 2를 골랐다. 사후 성공 여부는 이 선택에 쓰지 않았다.
미관측 공간 점수는 현재 지도에서 예상한 가시 셀 수이며 실제 깊이 반환을 보장하는 정보 이득은
아니다. 후보 4의 예상 미관측 셀은 5184개였지만 목표 구간의 예상 관측 점수는 0이었다.
따라서 이 한 사례를 광범위한 지도 완성도 비교나 정책의 일반적 우위로 확대하지 않는다.

성공 경로가 실제로 건넌 문턱 발판:

{table(crossings,['candidate_id','world_ijk','before_votes','before_evidence','after_votes','after_evidence'])}

대표 복셀 (-24,-195,0) 하나의 증거만 보면 성공을 놓친다. 후보 2/3은 같은 문턱 구간의 다른
발판을 두 번째로 관측해, 각각 약 1.58/1.44 m의 유효한 우회 경로를 제시했다. 목표와 입구는
동일하며 평가 과정에서 기준 경로로 대체하지 않았다. `candidate_paths/`에 실제 제시 경로의
XYZ·자세·폭, 검증과 첫 실패 위치를 저장했다.

## 범위와 다음 수정 방향

이 결과는 한 목표·한 시드·무처리 이상 깊이·참 포즈의 통제 실험이다. 앞서 SGBM/추정 포즈에서
0/10이었던 결과를 대체하지 않으며, 실제 연기·화재에서의 성능이나 여러 목표의 평균 우위를
입증하지 않는다. 참 포즈와 이상 깊이는 원인 분리용 진단 상한이다.

이번 목표의 기준 상태는 도달 가능이므로 올바른 차단 확인은 0개이며 차단 판단 능력은 검증하지
않았다. 성공 정책의 무효 경로는 0/1, 경로를 제시하지 않은 정책의 잘못 안내 비율은 N/A다.
따라서 이번 사례로 잘못된 안내 감소를 주장하지 않는다.

현재 우선순위는 **추정 포즈의 높이/자세와 공통 좌표계의 일관성을 개선하고 같은 짧은 경로를
다시 검증하는 것**이다. 평가기가 양성 경로를 거절하거나, 완벽한 센서/포즈에서도 표면층과
몸 검사 사이가 항상 끊기는 상황은 아니었다. 이번에는 평가 문턱이나 재관측 점수를 튜닝하지 않았다.

## 재실행·산출물

수정: `covor/entry/decision.py`에 단일 목표를 찾으면 멈추는 선택적 Dijkstra 인자를 추가했다.
기본 전체 도달 탐색은 그대로이며, 목표 경로가 완전 탐색과 같음을 테스트했다.
추가: `scripts/reobserve/short_path_probe.py`, `tests/test_short_path_probe.py`.
이외 지도/평가기 기본 규칙은 변경하지 않았다. 기존 보고서 산출물 {preserved}개의 해시를 확인했다.
새 단위 검증과 기존 회귀 테스트는 총 {sum(test_counts.values())}건 통과했다.

`candidate_outcomes.csv`: 24개 전체 결과. `policy_comparison.csv`/`policy_summary.csv`: 정책별 비교.
`selection.json`: 선택 전 점수·후보·입력 해시. `successful_threshold_crossings.json`: 실제 건넌 발판.
`candidate_maps/`, `candidate_observations/`, `candidate_paths/`: 모든 후보의 재생 자료.
큰 입력/지도/경로는 `{out}`에 있고, 설정·상세 CSV·로그는 보고서 옆에도 복사했다.

```bash
export MPLCONFIGDIR=/tmp/covor_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/src/gs25058/miniconda3/envs/covor/bin/python
O=results/short_path_probe_20261104_replay
for S in prepare positive ideal diagnose threshold certify select all-candidates contributors report; do
  $PY scripts/reobserve/short_path_probe.py --out $O --stage $S
done
$PY tests/test_short_path_probe.py
$PY tests/test_decision_paths.py
$PY tests/test_support_surfaces.py
$PY tests/test_entry_traverse.py
```
'''
    dest=ROOT/'reports'/out.name;dest.mkdir(parents=True,exist_ok=True)
    (dest/'RESULTS.md').write_text(report_text)
    for pattern in ('*.json','*.csv'):
        for p in out.glob(pattern):shutil.copy2(p,dest/p.name)
    for name in ('tests','existing_tests','support_tests','traverse_tests','positive','diagnose','all_candidates'):
        p=Path(f'/tmp/short_probe_{name}.log')
        if p.exists():shutil.copy2(p,dest/p.name)
    manifest={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='artifacts.json'}
    write_json(dest/'artifacts.json',manifest)
    print('REPORT',dest/'RESULTS.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    parser.add_argument('--stage',required=True,choices=('prepare','positive','ideal','diagnose','threshold','certify','select','all-candidates','contributors','report'))
    args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    {'prepare':prepare,'positive':positive,'ideal':build_ideal,'diagnose':diagnose,
     'threshold':threshold_case,'certify':certify_threshold,'select':select,'all-candidates':all_candidates,
     'contributors':contributors,'report':report}[args.stage](out)


if __name__=='__main__':main()
