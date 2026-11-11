"""Synchronous four-drone closed loop. Ground truth stays on simulator/eval side."""
from pathlib import Path
from time import perf_counter
import csv,json
import numpy as np
from .scene import Building
from .fire_world import make_world
from .sensing import Sensor,SharedMap
from .depth_observation import backend_name
from .human import Human,complete_support,ensure_current_analysis
from .motion import FlightMap,Trajectory,Hover,reserve,truth_collision,reservation_conflict,safe_rest_stop
from .policy import candidates,score_pool,marginal
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg,motion_path


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')

def csv_write(path,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def evaluate(analysis,truth,validator,prior,t,version):
    rows=[];index={g['id']:g for g in prior['goals']}
    for d,ref in zip(analysis['decisions'],truth):
        v=validator.validate(d['path'],prior['entry'],index[d['goal']]) if d['verdict']=='passable' else dict(status='not_proposed',reason=None)
        rows.append(dict(time=t,map_version=version,goal=d['goal'],reference_reachable=ref['reachable'],verdict=d['verdict'],reason=d['reason'],path_status=v['status'],path_reason=v.get('reason'),first_failure=v.get('first_failure'),path=d['path'],possible_path=d['possible_path'],uncertain_segments=d['uncertain_segments'],certificate=d.get('certificate')))
    R=sum(r['reference_reachable'] for r in rows);B=len(rows)-R
    V=sum(r['reference_reachable'] and r['path_status']=='valid' for r in rows);I=sum(r['path_status']=='invalid' for r in rows)
    valid=sum(r['path_status']=='valid' for r in rows);P=sum(r['verdict']=='passable' for r in rows);N=sum(r['path_status']=='unverifiable' for r in rows)
    C=sum(not r['reference_reachable'] and r['verdict']=='blocked' for r in rows);FB=sum(r['reference_reachable'] and r['verdict']=='blocked' for r in rows)
    D=sum(r['verdict']!='unverified' for r in rows)
    summary=dict(time=t,map_version=version,valid_confirmed=V,reachable_denominator=R,valid_path_confirmation_rate=V/R if R else None,correct_blocked=C,blocked_denominator=B,correct_blocked_rate=C/B if B else None,balanced_score=.5*(V/R+C/B) if R and B else None,invalid_paths=I,verified_path_denominator=valid+I,wrong_path_rate=I/(valid+I) if valid+I else None,proposed_paths=P,unverifiable_paths=N,unknown=len(rows)-D,decided=D,total_goals=len(rows),decision_rate=D/len(rows),false_blocked=FB,reference_undecidable=0,n_entry=analysis['n_entry'],n_confirmed=analysis['n_confirmed'],n_possible=analysis['n_possible'])
    return rows,summary


def validate_building(cfg,seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);b=make_world(cfg,seed);b.save(out)
    lab,ijk=b.labels(cfg['map_resolution']);truth,_,_=b.reference(cfg['map_resolution']);write_json(out/'truth.json',truth)
    layer=complete_support(b,ijk,lab.shape,cfg['map_resolution']);h=Human(b.prior(),cfg);a=h.analyze(lab,layer,ijk,False)
    validator=ReferenceValidator(lab,ijk,cfg['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
    rows,summary=evaluate(a,truth,validator,b.prior(),0,0);csv_write(out/'targets.csv',rows);write_json(out/'summary.json',summary)
    np.savez_compressed(out/'complete_map.npz',labels=lab,ijk=ijk,**layer)
    controls=[]
    for r in truth:
        if r['reachable']:
            goal=next(g for g in b.goals if g['id']==r['id']);v=validator.validate(motion_path(r['path']),b.entry,goal);controls.append(dict(goal=r['id'],**v))
    write_json(out/'positive_controls.json',controls)
    assert all(r['path_status']=='valid' if r['reference_reachable'] else r['verdict']=='blocked' for r in rows),[(r['goal'],r['verdict'],r['path_status'],r['first_failure']) for r in rows]
    assert all(c['status']=='valid' for c in controls),controls
    return summary


def run(cfg,scene_seed,run_seed,condition,policy,out,n_drones=None,duration=None,ideal=False,execution=None):
    out=Path(out)
    if (out/'summary.json').exists():raise FileExistsError(f'refusing to overwrite {out}')
    out.mkdir(parents=True,exist_ok=True);(out/'maps').mkdir(exist_ok=True)
    write_json(out/'config.json',dict(configuration=cfg,scene_seed=scene_seed,run_seed=run_seed,condition=condition,policy=policy,ideal_depth=ideal))
    wall=perf_counter();b=make_world(cfg,scene_seed);truth,_,_=b.reference(cfg['map_resolution']);b.render_scene(textured=backend_name(cfg)=='legacy_sgbm');gt,ijk=b.labels(cfg['map_resolution']);prior=b.prior()
    write_json(out/'prior.json',prior);write_json(out/'reference_targets.json',truth)
    sensor=Sensor(b,cfg,scene_seed*1000+run_seed,condition);m=SharedMap(prior,cfg);human=Human(prior,cfg)
    validator=ReferenceValidator(gt,ijk,m.res,b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
    n=n_drones or cfg['n_drones'];T=duration if duration is not None else cfg['duration'];dt=cfg['dt']
    guarded=cfg['motion'].get('controller')=='guarded_v1';travel_camera=cfg['motion'].get('travel_facing_camera',False)
    guard_period=cfg['motion'].get('safety_period_s',1.)
    if guarded:
        if guard_period<dt or abs(guard_period/dt-round(guard_period/dt))>1e-9:raise ValueError('guard cadence must be a positive integer multiple of dt')
        from .flight_control import make_flight_map,guarded_stop,observed_free_until,split_rest_points
    if travel_camera:
        from .flight_control import follow_heading
    if cfg['sensor_period']<dt-1e-9 or abs(cfg['sensor_period']/dt-round(cfg['sensor_period']/dt))>1e-9:raise ValueError('sensor period must be an integer multiple of dt; reduce dt for faster sampling')
    rng=np.random.default_rng(scene_seed*1000+run_seed+cfg.get('pose_error',{}).get('seed_offset',777))
    biases=np.zeros((n,3))
    if cfg['pose_mode']=='controlled_translation_bias':
        directions=rng.normal(size=(n,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
        biases=directions*rng.uniform(0,1,(n,1))**(1/3)*cfg['pose_error']['max_norm_m']
    elif cfg['pose_mode']!='perfect_odometry':raise ValueError('unsupported pose mode')
    write_json(out/'pose_error.json',dict(mode=cfg['pose_mode'],biases=biases.tolist(),reference_alignment='fixed common world frame; no posthoc realignment'))
    states=[dict(id=i,p=p.copy()+biases[i],bias=biases[i],yaw=np.pi/2,pitch=cfg['sensor']['gimbal_scan']['lower'],traj=None,task_until=0.,failed=False,active=False,distance=0.,captures=0,wait=0.,last_assignment=-100.) for i,p in enumerate(b.starts[:n])]
    trajectory=[];frames=[];events=[];timing=[];targets=[];scores=[];assignment=[];series=[]
    if execution is not None:
        execution.prepare(cfg,scene_seed,run_seed,condition,policy,n,T,dt);sensor=execution.wrap_sensor(sensor)
        restored=execution.restore_map(m)
    else:restored=None
    mode='a' if restored is not None else 'w'
    log=(out/'decisions.jsonl').open(mode);candidates_log=(out/'candidates.jsonl').open(mode);assign_log=(out/'assignments.jsonl').open(mode)
    analysis=None;last_lab=None;last_layer=None;next_sensor=0.;next_assess=0.;next_assign=0.;next_snapshot=0.;fallback_seconds=0.;planned_failures=0
    next_guard=0.;guard_flight=None;guard_version=-1
    start_tick=0
    if restored is not None:
        state,last_lab,last_layer=restored
        states=state['states'];trajectory=state['trajectory'];frames=state['frames'];events=state['events'];timing=state['timing'];targets=state['targets'];scores=state['scores'];assignment=state['assignment'];series=state['series']
        next_sensor=state['next_sensor'];next_assess=state['next_assess'];next_assign=state['next_assign'];next_snapshot=state['next_snapshot'];fallback_seconds=state['fallback_seconds'];planned_failures=state['planned_failures']
        next_guard=state.get('next_guard',0.)
        start_tick=execution.record['tick']+1;wall-=execution.saved_elapsed
    for tick in range(start_tick,int(round(T/dt))+1):
        t=tick*dt
        # Execute old trajectories first. Sample true collisions at <=5 cm time
        # intervals, without letting truth repair a planner's chosen trajectory.
        old=[s['p'].copy() for s in states]
        for s in states:
            s['active']=t>=s['id']*cfg['entry_interval'] and not s['failed']
            if not s['active']:
                if not s['failed']:s['wait']+=dt if tick else 0
                continue
            tr=s['traj']
            if tr is not None:
                new=tr.sample(t);yaw,pitch=tr.orientation(t)
                collision=b.flight_collision(s['p']-s['bias'],new-s['bias'],cfg['motion']['radius']) if hasattr(b,'flight_collision') else truth_collision(s['p']-s['bias'],new-s['bias'],b.lo,b.hi,cfg['motion']['radius'])
                if collision:
                    s['failed']=True;events.append(dict(time=t,drone=s['id'],event='geometry_collision_frozen',position=s['p'].tolist()));s['traj']=None
                else:
                    s['distance']+=float(np.linalg.norm(new-s['p']));s['p']=new
                    if not travel_camera:s['yaw']=yaw
                if t>=tr.end:
                    s['traj']=None
                    if tr.stop_at is not None:s['task_until']=t;events.append(dict(time=t,drone=s['id'],event='stopped_at_rest_for_replan'))
        # Common physical gimbal program, observed during travel as well as dwell.
        scan=cfg['sensor']['gimbal_scan'];phase=t%scan['period'];low=scan['lower'];high=scan['upper'];hold=scan['hold'];slew=scan['slew']
        if phase<=hold:actual_pitch=low
        elif phase<=hold+slew:actual_pitch=low+(high-low)*(phase-hold)/slew
        elif phase<=2*hold+slew:actual_pitch=high
        elif phase<=2*hold+2*slew:actual_pitch=high-(high-low)*(phase-2*hold-slew)/slew
        else:actual_pitch=low
        for s in states:
            if not s['failed']:
                if travel_camera:
                    s['pitch']+=float(np.clip(actual_pitch-s['pitch'],-cfg['motion']['pitch_rate']*dt,cfg['motion']['pitch_rate']*dt))
                    if s['active']:s['yaw']=follow_heading(s,t,dt,cfg['motion'])
                else:s['pitch']=actual_pitch
        # Between-tick pair sweep detects swaps even if endpoints are separate.
        for i in range(n):
            for j in range(i+1,n):
                a=(old[i]-states[i]['bias'])-(old[j]-states[j]['bias']);d=((states[i]['p']-states[i]['bias'])-(states[j]['p']-states[j]['bias']))-a;f=np.clip(-a@d/max(d@d,1e-12),0,1)
                if np.linalg.norm(a+f*d)<2*cfg['motion']['radius']:
                    for k in (i,j):
                        if not states[k]['failed']:events.append(dict(time=t,drone=k,event='teammate_collision_frozen',position=states[k]['p'].tolist()))
                        states[k]['failed']=True;states[k]['traj']=None;states[k]['p']=old[k]
        for s in states:
            trajectory.append(dict(time=t,drone=s['id'],x=s['p'][0]-s['bias'][0],y=s['p'][1]-s['bias'][1],z=s['p'][2]-s['bias'][2],estimated_x=s['p'][0],estimated_y=s['p'][1],estimated_z=s['p'][2],yaw=s['yaw'],gimbal_pitch=s['pitch'],active=s['active'],failed=s['failed']))
        if t+1e-8>=next_sensor:
            for s in states:
                if not s['active'] or s['failed']:continue
                fid=f'd{s["id"]}_t{tick:05d}';mates=[q['p']-q['bias'] for q in states if q['id']!=s['id']];estimated_mates=[q['p'] for q in states if q['id']!=s['id']]
                start=perf_counter();cam,z,sz,valid,meta=sensor.capture(s['p']-s['bias'],s['yaw'],s['pitch'],t,mates,out/'frames',fid,ideal,drone_id=s['id']);sensor_wall=perf_counter()-start
                estimated_cam=cam.copy();estimated_cam[:3,3]+=s['bias']
                start=perf_counter();support=m.integrate(sensor.sd,estimated_cam,z,sz,valid,estimated_mates);map_wall=perf_counter()-start
                s['captures']+=1;frames.append(dict(time=t,drone=s['id'],frame_id=fid,map_version=m.version,camera_pose=estimated_cam.tolist(),camera_pose_truth=cam.tolist(),sensor_wall=sensor_wall,mapping_wall=map_wall,support_points=support,**meta))
            start=perf_counter();last_lab,last_layer=m.dense();timing.append(dict(time=t,component='dense_map',seconds=perf_counter()-start));next_sensor+=cfg['sensor_period']
        if guarded and t+1e-8>=next_guard:
            start=perf_counter()
            if guard_version!=m.version:
                guard_flight=make_flight_map(last_lab,m.ijk,m.res,cfg['motion']);guard_version=m.version
            for moving in states:
                tr=moving['traj']
                if tr is None or moving['failed'] or observed_free_until(tr,t,tr.end,guard_flight):continue
                others=[q['traj'] if q['traj'] is not None else Hover(q['p']) for q in states if q['id']!=moving['id']]
                stopped,reason=guarded_stop(tr,moving,t,others,guard_flight,cfg['motion'])
                if stopped is not None:
                    moving['traj']=stopped;moving.pop('item',None)
                    events.append(dict(time=t,drone=moving['id'],event='observed_map_guard_stop',kind=reason,stop_time=stopped.end,trajectory=stopped.record(),initial_position=tr.sample(t).tolist(),initial_velocity=tr.sample(t,1).tolist(),initial_acceleration=tr.sample(t,2).tolist(),map_version=m.version))
                else:events.append(dict(time=t,drone=moving['id'],event='observed_map_guard_no_safe_brake',reason=reason,map_version=m.version))
            timing.append(dict(time=t,component='observed_map_flight_guard',seconds=perf_counter()-start));next_guard+=guard_period
        if t+1e-8>=next_assess or tick==int(round(T/dt)):
            start=perf_counter();analysis=human.analyze(last_lab,last_layer,m.ijk,True);analysis['map_version']=m.version;timing.append(dict(time=t,component='human',seconds=perf_counter()-start))
            start=perf_counter();rows,summary=evaluate(analysis,truth,validator,prior,t,m.version);timing.append(dict(time=t,component='reference_path_evaluation',seconds=perf_counter()-start));targets.extend(rows);series.append(summary)
            log.write(json.dumps(dict(time=t,map_version=m.version,decisions=analysis['decisions'],impact=analysis['impact'],cut=analysis['cut']),default=lambda x:x.item())+'\n');log.flush();next_assess+=cfg['assessment_period']
        if t+1e-8>=next_snapshot:
            m.save(out/'maps'/f't{tick:05d}.npz',last_lab,last_layer);next_snapshot+=cfg['snapshot_period']
        if t+1e-8>=next_assign and t<T:
            start=perf_counter();analysis,refreshed=ensure_current_analysis(human,analysis,last_lab,last_layer,m.ijk,m.version)
            if refreshed:
                log.write(json.dumps(dict(time=t,map_version=m.version,role='planning_refresh',decisions=analysis['decisions'],impact=analysis['impact'],cut=analysis['cut']))+'\n');log.flush()
            assert analysis['map_version']==m.version
            if cfg['utility'].get('entry_objective','legacy')!='legacy':
                # Actual past estimated camera poses only, at1Hz per drone.
                # No depth/error/reference fields enter utility history.
                history=[]
                for observation in frames:
                    if abs(observation['time']-round(observation['time']))>1e-8:continue
                    camera=np.asarray(observation['camera_pose']);direction=camera[:3,2]
                    history.append((camera[:3,3].tolist(),float(np.arctan2(direction[1],direction[0]))))
                analysis['_observed_views']=history
            if refreshed:timing.append(dict(time=t,component='human_planning_refresh',seconds=perf_counter()-start))
            start=perf_counter()
            if guarded:
                if guard_version!=m.version:guard_flight=make_flight_map(last_lab,m.ijk,m.res,cfg['motion']);guard_version=m.version
                flight=guard_flight
            else:flight=FlightMap(last_lab,m.ijk,m.res,cfg['motion'])
            epoch=int(round(t/cfg['assignment_period']))
            # Re-evaluate current map at a common bounded cadence. Both utilities
            # receive precisely the same inference/candidate computation budget.
            for moving in ([] if guarded else states):
                tr=moving['traj']
                if tr is None or tr.stop_at is not None:continue
                future=np.arange(t,tr.end+.05,.25)
                if not all(flight.point_clear(tr.sample(ti)) for ti in future):
                    other_reservations=[q['traj'] if q['traj'] is not None else Hover(q['p']) for q in states if q['id']!=moving['id']]
                    stopped=safe_rest_stop(tr,t,other_reservations,cfg['motion'])
                    if stopped is not None:
                        moving['traj']=stopped
                        events.append(dict(time=t,drone=moving['id'],event='map_change_joint_safe_stop',stop_time=stopped.stop_at))
                    else:
                        events.append(dict(time=t,drone=moving['id'],event='no_joint_safe_rest_stop',reason='retain published trajectory; no instantaneous velocity reset'))
            pool=candidates(last_lab,m.ijk,m.res,last_layer,flight,states,cfg,epoch,run_seed,prior)
            if cfg['candidates'].get('body_reobserve_views'):
                from .candidate_views import refine_body_views
                pool=refine_body_views(pool,analysis,last_lab,last_layer,m.ijk,m.res,sensor.sd,cfg,prior,flight=flight,states=states)
            items=score_pool(pool,analysis,last_lab,last_layer,m.ijk,m.res,sensor.sd,cfg,prior)
            for item in items:
                candidates_log.write(json.dumps(dict(time=t,map_version=m.version,entry_graph_version=analysis['map_version'],**item['candidate'],unknown_prediction=len(item['unknown']),entry_target_scores=item['targets'],contributions=item['contributions'],objective_diagnostics=item.get('objective_diagnostics'),planning_inputs='observed_map_and_current_poses_only'))+'\n')
            entry_active=any(any(v>0 for v in it['targets'].values()) for it in items)
            if policy=='entry' and not entry_active:fallback_seconds+=min(cfg['assignment_period'],T-t)
            # Existing reservations enter marginal union by their current endpoint
            # prediction, excluding teammate observations that are not yet seen.
            allocated=[]
            pending=[s for s in states if s['traj'] is not None and 'item' in s]
            pending_items=score_pool([s['item']['candidate'].copy() for s in pending],analysis,last_lab,last_layer,m.ijk,m.res,sensor.sd,cfg,prior)
            for state,item in zip(pending,pending_items):
                state['item']=item;allocated.append(item)
                events.append(dict(time=t,drone=state['id'],event='pending_utility_refreshed',map_version=m.version,entry_target_scores=item['targets']))
            order=[(epoch+i)%n for i in range(n)]
            for i in order:
                s=states[i]
                if not s['active'] or s['failed'] or t<s['task_until'] or s['traj'] is not None:continue
                # No utility-dependent route prefilter: every candidate receives
                # an identical observed-map feasibility/trajectory check.
                options=sorted(items,key=lambda it:it['candidate']['id'])
                if len(options)>cfg['candidates']['max_route_trials_per_drone']:raise ValueError('route budget must cover the complete common pool')
                events.append(dict(time=t,drone=i,event='route_pool_all_candidates',candidate_ids=[it['candidate']['id'] for it in options],n_candidates=len(options)))
                offers=[]
                stationary=[q['p'] if q['traj'] is None else q['traj'].sample(q['traj'].end) for q in states if q['id']!=i]
                routed_flight=flight.avoid(stationary)
                for it in options:
                    c=it['candidate'];path=routed_flight.route(s['p'],c['position'])
                    if path is None:
                        planned_failures+=1;events.append(dict(time=t,drone=i,event='path_plan_failed',candidate=c['id']));continue
                    if guarded:path=split_rest_points(path,cfg['motion'].get('rest_spacing_m'))
                    others=[q['traj'] if q['traj'] is not None else Hover(q['p']) for q in states if q['id']!=i]
                    tr=reserve(path,s,c,t,others,cfg['motion'])
                    if tr is None:
                        planned_failures+=1;events.append(dict(time=t,drone=i,event='reservation_failed',candidate=c['id']));continue
                    cost=tr.total+cfg['capture_dwell']+cfg['motion'].get('camera_setup_seconds',0.)
                    gain,fallback=marginal(it,policy,allocated,cfg,entry_active,remaining=T-t,cost=cost)
                    value=gain-cfg['utility']['time_penalty']*cost
                    if policy=='entry' and cfg['utility'].get('offer_objective')=='gain_rate':
                        value=gain/max(cost,cfg['sensor_period'])
                    assignment.append(dict(time=t,drone=i,candidate=c['id'],policy=policy,gain=gain,cost_seconds=cost,score=value,fallback=fallback,path_length=float(np.linalg.norm(np.diff(np.asarray(path),axis=0),axis=1).sum()),reservation_wait=tr.wait))
                    offers.append((value,-cost,c['id'],it,tr,fallback))
                if offers and policy=='entry' and cfg['utility'].get('entry_objective')=='completion_opportunity_v1':
                    # Do not activate a task mode because an UNREACHABLE view
                    # had a score. Test only feasible, on-time offers for this
                    # actual drone after reservations and pending assignments.
                    gains=[marginal(o[3],'entry',allocated,cfg,True)[0] if -o[1]<=T-t else 0. for o in offers]
                    intervene=any(g>0 for g in gains);updated=[]
                    for row,o,gain in zip(assignment[-len(offers):],offers,gains):
                        _,negative_cost,cid,item,tr,_=o;cost=-negative_cost
                        if intervene:value=gain/max(cost,cfg['sensor_period']) if gain>0 else -1e9;fallback=False
                        else:
                            gain,_=marginal(item,'unknown',allocated,cfg,False);value=gain-cfg['utility']['time_penalty']*cost;fallback=True
                        row.update(gain=gain,score=value,fallback=fallback)
                        updated.append((value,negative_cost,cid,item,tr,fallback))
                    offers=updated
                    events.append(dict(time=t,drone=i,event='completion_opportunity_gate',feasible_completion_offers=sum(g>0 for g in gains),exploration_fallback=not intervene))
                if offers:
                    value,_,cid,it,tr,fallback=sorted(offers,key=lambda x:(-x[0],-x[1],x[2]))[0]
                    s['traj']=tr;s['task_until']=max(t+cfg['minimum_task_time'],tr.end+cfg['capture_dwell']+cfg['motion'].get('camera_setup_seconds',0.));s['last_assignment']=t;s['item']=it;allocated.append(it)
                    record=dict(time=t,drone=i,candidate=it['candidate'],policy=policy,score=value,fallback=fallback,map_version=m.version,trajectory=tr.record(),task_until=s['task_until'],selection_before_sensor=True)
                    assign_log.write(json.dumps(record)+'\n');assign_log.flush();events.append(dict(time=t,drone=i,event='assigned',candidate=cid,wait=tr.wait))
                else:events.append(dict(time=t,drone=i,event='safe_hover_no_feasible_offer',position=s['p'].tolist()))
            candidates_log.flush();timing.append(dict(time=t,component='candidates_assignment_planning',seconds=perf_counter()-start));next_assign+=cfg['assignment_period']
        if tick and tick%int(8/dt)==0:print(f'{out.name} t={t:.0f} frames={m.version} distances={[round(s["distance"],1) for s in states]} V={series[-1]["valid_confirmed"]} B={series[-1]["correct_blocked"]}',flush=True)
        if execution is not None:
            runtime=dict(states=states,trajectory=trajectory,frames=frames,events=events,timing=timing,targets=targets,scores=scores,assignment=assignment,series=series,next_sensor=next_sensor,next_assess=next_assess,next_assign=next_assign,next_snapshot=next_snapshot,next_guard=next_guard,fallback_seconds=fallback_seconds,planned_failures=planned_failures,T=T)
            execution.maybe_save(tick,m,runtime,[log,candidates_log,assign_log],perf_counter()-wall,last_lab,last_layer)
    log.close();candidates_log.close();assign_log.close()
    initial={r['goal']:r for r in targets if r['time']==0}
    transition=[]
    for r in [q for q in targets if q['time']==T]:
        old=initial[r['goal']]
        transition.append(dict(goal=r['goal'],before=old['verdict'],after=r['verdict'],unknown_to_valid=old['verdict']=='unverified' and r['path_status']=='valid',unknown_to_correct_blocked=old['verdict']=='unverified' and r['verdict']=='blocked' and not r['reference_reachable'],new_invalid=r['path_status']=='invalid' and old['path_status']!='invalid',revoked_passable=old['verdict']=='passable' and r['verdict']!='passable',revoked_invalid=old['path_status']=='invalid' and r['verdict']!='passable',lost_valid=old['path_status']=='valid' and r['path_status']!='valid'))
    csv_write(out/'transitions.csv',transition)
    final=dict(series[-1],scene_seed=scene_seed,run_seed=run_seed,condition=condition,policy=policy,n_drones=n,duration=T,ideal_depth=ideal,pose_mode=cfg['pose_mode'],travel_distance=sum(s['distance'] for s in states),captures=sum(s['captures'] for s in states),entry_wait=sum(min(T,i*cfg['entry_interval']) for i in range(n)),reservation_wait=sum(e.get('wait',0) for e in events),failed_drones=sum(s['failed'] for s in states),collision_events=sum('collision' in e['event'] for e in events),planning_failures=planned_failures,safe_hover_events=sum(e['event']=='safe_hover_no_feasible_offer' for e in events),fallback_seconds=fallback_seconds,wall_seconds=perf_counter()-wall)
    csv_write(out/'trajectories.csv',trajectory);csv_write(out/'frames.csv',frames);csv_write(out/'events.csv',events);csv_write(out/'timing.csv',timing);csv_write(out/'target_paths.csv',targets);csv_write(out/'assignment_scores.csv',assignment);csv_write(out/'metrics_time.csv',series);m.save(out/'maps'/'final.npz',last_lab,last_layer);write_json(out/'summary.json',final)
    print('COMPLETED',out,final,flush=True);return final
