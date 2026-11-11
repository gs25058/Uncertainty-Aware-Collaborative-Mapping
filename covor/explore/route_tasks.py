"""Observed-map repair tasks on potential paths, never confirmed routes.

Search is rooted in the actual confirmed entrance component. Requests contain
the path's own unconfirmed nodes and diagonal corner dependencies, rather than
the whole connected uncertainty segment. No scene, sensor, or truth API exists
here. The human graph and its evidence are never modified.
"""
import heapq
import numpy as np


def build_route_tasks(context, analysis):
    g=analysis['graph']; opt=context.opt
    starts=sorted(analysis['confirmed_search']['best'])
    targets={goal['id']:set(g.mission_nodes(goal)) for goal in context.prior['goals']
             if context.decisions[goal['id']]['verdict']=='unverified'}
    targets={key:value for key,value in targets.items() if value}; lookup={}
    for goal,nodes in targets.items():
        for node in nodes:lookup.setdefault(node,[]).append(goal)
    distance={u:0. for u in starts}; parent={u:None for u in starts}
    heap=[(0.,u) for u in starts]; heapq.heapify(heap)
    found={}; expanded=0; budget=opt.get('soft_path_max_expansions',50000)
    while heap and expanded<budget and len(found)<len(targets):
        d,u=heapq.heappop(heap)
        if d>distance[u]:continue
        expanded+=1
        for goal in lookup.get(u,[]):found.setdefault(goal,u)
        for v,length,deps in g.edges(u,False):
            # Cost is remaining repair effort plus a small geometric tie cost.
            # Unknown connection budget is relaxed ONLY inside this value search.
            repair=opt.get('unknown_connection_penalty',4.) if deps else 0.
            nd=d+length*(.01+repair)+g.body.c_rise*abs(g.node_height[v]-g.node_height[u])
            if nd<distance.get(v,np.inf):
                distance[v]=nd; parent[v]=u; heapq.heappush(heap,(nd,int(v)))
    requests={}; path_sections={}; route_records={};context.value_paths={}
    extent=np.array([opt.get('repair_section_extent_m',.6)]*2+[.3])
    for goal,tip in sorted(found.items()):
        path=[]; node=tip
        while node is not None:path.append(node); node=parent[node]
        path.reverse(); needed={n for n in path if not g.node_known[n]}
        context.value_paths[goal]=path
        for u,v in zip(path,path[1:]):
            for destination,_,corners in g._raw_neighbors(u):
                if destination!=v:continue
                for choices in corners:
                    known=[n for n in choices if g.node_known[n]]
                    if not known and choices:needed.add(choices[0])
                break
        grouped={}
        for n in sorted(needed):
            point=np.r_[g.node_xy[n],g.node_height[n]-g.foot_contact_offset+.002]
            key=tuple(np.floor((point+[0,0,.15])/extent).astype(int))
            grouped.setdefault(key,[]).append(point)
        path_sections[goal]=len(grouped)
        route_records[goal]=dict(potential_nodes=len(path),unconfirmed_nodes=len(needed),
                                remaining_repair_cost=float(distance[tip]),sections=len(grouped))
        for key,points in grouped.items():
            request=requests.setdefault(key,dict(points=[],goals={}))
            request['points'].extend(points); request['goals'][goal]=1/max(1,len(grouped))
    ordered=sorted(requests.items(),key=lambda kv:(-sum(kv[1]['goals'].values()),kv[0]))
    for key,request in ordered[:opt.get('max_anchored_queries',8)]:
        context.query('repair_'+str(key),np.unique(request['points'],axis=0),request['goals'],'anchored')
    context.serial_sections=path_sections
    context.soft_route_stats=dict(expanded_nodes=expanded,expansion_budget=budget,
        goals_with_observed_anchors=len(targets),goals_with_value_only_path=len(found),
        uncertain_sections_by_goal=path_sections,queried_sections=min(len(ordered),opt.get('max_anchored_queries',8)),
        classification_unchanged=True,root='actual_confirmed_entrance_component',
        query_geometry='path_nodes_and_required_diagonal_corners_only',routes=route_records)


def serial_values(predictions, section_counts, partial_weight=.15):
    """Diminishing remaining-work potential; all unqueried sections stay missing.

    This is planning utility, not a probability of a valid route. A last repair
    is worth more than the same fractional repair on a long unfinished route.
    Complementary foot/body views are combined before the minimum.
    """
    progress={}; partial={}
    for pred in predictions.values():
        if pred['kind']!='anchored':continue
        f=float(np.mean(pred['foot'])); b=float(np.mean(pred['body']))
        f0=pred['foot_before']; b0=pred['body_before']
        improvement=max(0.,min(f,b)-min(f0,b0))
        extra=max(0.,(f+b-f0-b0)/2)
        for goal in pred['goals']:
            progress[goal]=progress.get(goal,0.)+min(1.,improvement)
            partial[goal]=partial.get(goal,0.)+min(1.,extra)
    result={}
    for goal,n in section_counts.items():
        if n<=0:continue
        repaired=min(float(n),progress.get(goal,0.))
        base=1/(1+n)
        result[goal]=(1/(1+n-repaired)-base)/(1-base)
        result[goal]+=partial_weight*min(1.,partial.get(goal,0.)/n)
    return result
