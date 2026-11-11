#!/usr/bin/env python3
"""Fixed-map route-ranking ablation. All routes are generated before truth opens."""
import argparse,copy,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.human import Human, relaxed_certificate
from covor.explore.route_quality import rerank_confirmed_analysis
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.runner import evaluate,write_json,csv_write
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--complete-goal-cut',action='store_true');p.add_argument('--skip-quality',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    summaries=[];details=[]
    for source in a.runs:
        out=a.out/source.name
        if out.exists():raise FileExistsError(out)
        out.mkdir()
        metadata=json.loads((source/'config.json').read_text());cfg=metadata['configuration']
        prior=json.loads((source/'prior.json').read_text())
        with np.load(source/'maps/final.npz') as q:
            lab=q['labels'];ijk=q['ijk'];version=int(q['version']);layer={k:q[k] for k in q.files if k not in ('labels','ijk','res','version')}
        analyses={'shortest':Human(prior,cfg).analyze(lab,layer,ijk,False)};configs={}
        if not a.skip_quality:analyses['observed_quality']=rerank_confirmed_analysis(analyses['shortest'],prior)
        for name,weight in [('shortest',0.),('observed_quality',1.)]:
            c=copy.deepcopy(cfg);c['human']['observed_route_quality_weight']=weight;configs[name]=c
        if not a.skip_quality:assert [d['verdict'] for d in analyses['shortest']['decisions']]==[d['verdict'] for d in analyses['observed_quality']['decisions']]
        if a.complete_goal_cut:
            graph=analyses['shortest']['graph'];context={};decisions=[]
            for d,goal in zip(analyses['shortest']['decisions'],prior['goals']):
                proof=None
                if d['verdict']!='passable' and analyses['shortest']['n_entry'] and not d['possible_path']:
                    proof=relaxed_certificate(lab,layer,ijk,cfg['map_resolution'],prior['entry'],goal,graph.body,graph,context,complete_goal=True)
                verdict='passable' if d['verdict']=='passable' else 'blocked' if proof else 'unverified'
                reason=proof['reason'] if proof else ('whole_target_cut_not_certified' if d['verdict']=='blocked' else d['reason'])
                decisions.append(dict(d,verdict=verdict,certificate=proof,reason=reason))
            analyses['complete_goal_cut']=dict(analyses['shortest'],decisions=decisions)
            if not a.skip_quality:analyses['quality_and_cut']=dict(analyses['observed_quality'],decisions=[dict(d,path=q['path']) for d,q in zip(decisions,analyses['observed_quality']['decisions'])])
            for name,weight in [('complete_goal_cut',0.),('quality_and_cut',1.)]:
                c=copy.deepcopy(cfg);c['human'].update(full_goal_cut_certificate=True,observed_route_quality_weight=weight);configs[name]=c
        # Validate baseline replay against the path that was ACTUALLY exported.
        csv.field_size_limit(sys.maxsize)
        archived=list(csv.DictReader((source/'target_paths.csv').open()));last=max(float(r['time']) for r in archived)
        archived={r['goal']:json.loads(r['path']) for r in archived if float(r['time'])==last}
        exact={d['goal']:d['path']==archived[d['goal']] for d in analyses['shortest']['decisions']}
        assert all(exact.values()),exact
        write_json(out/'selection_before_truth.json',dict(source=str(source),all_paths_generated_before_reference=True,graph_feasibility_unchanged=True,route_quality_verdicts_unchanged=True,goal_cut_can_change_blocked_unverified=a.complete_goal_cut,baseline_paths_exact=exact,
            candidates={key:[dict(goal=d['goal'],path=d['path']) for d in analysis['decisions']] for key,analysis in analyses.items()}))
        # Only now is independent reference geometry introduced.
        b=ReconFireWorld(metadata['scene_seed'],cfg);gt,gtijk=b.labels(cfg['map_resolution']);b.render_scene(False)
        truth=json.loads((source/'reference_targets.json').read_text())
        validator=ReferenceValidator(gt,gtijk,cfg['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
        for name,analysis in analyses.items():
            rows,summary=evaluate(analysis,truth,validator,prior,last,version)
            summary.update(source=str(source),method=name,scene=metadata['scene_seed'],condition=metadata['condition'],extra_captures=0,extra_movement=0.)
            csv_write(out/(name+'_paths.csv'),rows);summaries.append(summary)
            details.extend(dict(source=str(source),method=name,**r) for r in rows)
            print(source.name,name,summary['valid_confirmed'],summary['invalid_paths'],flush=True)
        write_json(out/'configurations.json',{key:value for key,value in configs.items() if key in analyses})
    csv_write(a.out/'summary.csv',summaries);csv_write(a.out/'target_paths.csv',details)
    write_json(a.out/'scope.json',dict(mode='fixed_observed_map_route_and_goal_certificate_ablation',development_diagnostic=True,no_reobservation=True,human_geometry_and_graph_feasibility_unchanged=True,goal_certificate_terminal_coverage=a.complete_goal_cut,reference_not_used_to_select_route=True))


if __name__=='__main__':main()
