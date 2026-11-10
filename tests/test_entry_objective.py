"""Map-only utility controls; no changes to human/reference route decisions."""
import inspect,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from types import SimpleNamespace
from covor.explore.entry_objective import TaskContext,Query
from covor.explore.policy import marginal


class EntryObjectiveControls(unittest.TestCase):
    def context(self):
        c=TaskContext.__new__(TaskContext);c.lab=np.zeros((20,20,40),np.uint8)
        c.ijk=np.zeros(3,int);c.res=.1;c.opt={'unknown_occlusion_discount':.5}
        c.sd=SimpleNamespace(fx=30.,fy=30.,cx=50.,cy=50.,R1=np.eye(3),calib=SimpleNamespace(size=(100,100)),cfg=SimpleNamespace(z_min=.1,z_max=6.),range_metric='optical_z')
        return c
    def test_observed_wall_occludes_even_important_enquiry(self):
        c=self.context();foot=np.array([[.5,.5,2.]]);body=foot+[0,0,.3]
        q=Query('important',foot,body,np.zeros(1),np.zeros(1),np.ones(1),{'g':3.},'boundary')
        before,_=c.observation_value(q,[np.eye(4)],c.sd);self.assertGreater(before,0)
        c.lab[:,:,10]=2;after,_=c.observation_value(q,[np.eye(4)],c.sd);self.assertEqual(after,0)
    def test_full_body_query_finds_unknown_side_cell_missed_by_centreline(self):
        from covor.entry.traverse import TravCfg
        c=self.context();c.lab[:]=1;c.lab[7,5,3]=0;c.human_body=TravCfg(res=.1)
        shape=c.lab.shape;c.layer=dict(height=np.full(shape,np.nan),support=np.zeros(shape,bool),votes=np.zeros(shape,np.uint16))
        c.layer['height'][:,:,0]=.1;c.layer['support'][:,:,0]=True;c.layer['votes'][:,:,0]=4;c.queries=[]
        c.query('centreline',[[.55,.55,.102]],{'g':1.},'boundary');self.assertTrue(c.queries[-1].body_known.all())
        c.opt['body_query_mode']='missing_footprint';c.query('full_body',[[.55,.55,.102]],{'g':1.},'boundary')
        self.assertEqual(len(c.queries[-1].body),1)
        np.testing.assert_allclose(c.queries[-1].body[0],[.75,.55,.35]);self.assertEqual(c.queries[-1].body_known[0],0.)
    def test_foot_quality_uses_public_sigma_and_sampling_not_true_error(self):
        c=self.context();c.opt.update(foot_precision_scale_m=.006,mapping_sample_stride=4,effective_scan_views=4)
        c.sensor_cfg={'error_model':{'sigma_constant_m':.002,'sigma_quadratic_per_m':.001}}
        scores=[]
        for height in [1.,2.,4.,6.]:
            T=np.eye(4);T[:3,:3]=np.diag([1,-1,-1]);T[2,3]=height
            scores.append(float(c.foot_quality(np.array([[0,0,0.]]),T,c.sd)[0]))
        self.assertTrue(all(0<a<=1 for a in scores));self.assertTrue(all(a>b for a,b in zip(scores,scores[1:])))
    def test_contact_readiness_matches_the_actual_height_bound_not_votes_alone(self):
        c=self.context();c.lab[:]=1;shape=c.lab.shape;c.queries=[]
        c.layer=dict(height=np.full(shape,np.nan),support=np.zeros(shape,bool),votes=np.zeros(shape,np.uint16),height_std=np.zeros(shape),hit_z_max=np.full(shape,np.nan))
        c.layer['height'][:,:,0]=.1;c.layer['support'][:,:,0]=True;c.layer['votes'][:,:,0]=20;c.layer['height_std'][:,:,0]=.02;c.layer['hit_z_max'][:,:,0]=.101
        c.opt['contact_aware_foot_evidence']=True;c.query('uncertain_height',[[.55,.55,.102]],{'g':1.},'boundary')
        self.assertEqual(c.queries[-1].foot_known[0],0.)
        c.layer['height_std'][:,:,0]=.001;c.query('precise_height',[[.55,.55,.102]],{'g':1.},'boundary')
        self.assertEqual(c.queries[-1].foot_known[0],1.)
    def test_known_support_does_not_require_observing_it_again(self):
        c=self.context();foot=np.array([[.5,.5,2.]]);body=foot+[0,0,.3]
        q=Query('body_missing',foot,body,np.ones(1),np.zeros(1),np.zeros(1),{'g':1.},'goal')
        v,proof=c.observation_value(q,[np.eye(4)],c.sd);self.assertGreater(v,0);self.assertEqual(proof['foot_visible'],0.)
    def test_only_foot_view_cannot_complete_both_missing_requirements(self):
        c=self.context();q=Query('only_floor',np.array([[.5,.5,2.]]),np.array([[9.,9.,2.]]),np.zeros(1),np.zeros(1),np.ones(1),{'g':1.},'goal')
        self.assertEqual(c.observation_value(q,[np.eye(4)],c.sd)[0],0.)
    def test_new_task_scores_and_unknown_union_do_not_duplicate(self):
        cfg={'utility':{'entry_objective':'task_progress_v1','unknown_normalizer':10,'task_progress':{'exploration_weight':.1}}}
        item={'unknown':{1,2},'targets':{'g':.7},'evidence':{}}
        self.assertGreater(marginal(item,'entry',[],cfg,True)[0],0)
        self.assertEqual(marginal(item,'entry',[item],cfg,True)[0],0)
        self.assertEqual(marginal(item,'unknown',[],cfg,True)[0],.2)
    def test_deadline_discount_applies_to_entry_value_only(self):
        cfg={'capture_dwell':6.,'utility':{'entry_objective':'task_progress_v2','unknown_normalizer':10,'task_progress':{'exploration_weight':.1,'deadline_discount':True}}}
        item={'unknown':{1,2},'targets':{'g':.7},'evidence':{}}
        self.assertEqual(marginal(item,'entry',[],cfg,True,remaining=4.,cost=20.)[0],0.)
        self.assertEqual(marginal(item,'unknown',[],cfg,True,remaining=4.,cost=20.)[0],.2)
    def test_complementary_foot_body_views_and_duplicate_queries(self):
        from covor.explore.entry_objective import joint_marginal,pooled_target_values
        cfg={'utility':{'task_progress':{'boundary_weight':1.,'partial_evidence_weight':.15}}}
        def item(f,b,key='bridge'):
            return dict(task_channels={},task_query_predictions={key:dict(kind='boundary',goals={'g':1.},foot=[f],body=[b],foot_before=0.,body_before=0.)})
        a=item(1.,0.);b=item(0.,1.)
        self.assertGreater(joint_marginal(b,[a],cfg),joint_marginal(b,[],cfg))
        self.assertEqual(joint_marginal(a,[a],cfg),0.)
        one=pooled_target_values([item(1.,1.)],cfg)['g']
        duplicate=pooled_target_values([item(1.,1.),item(1.,1.,'other_patch_same_goal')],cfg)['g']
        self.assertEqual(one,duplicate)
    def test_anchored_planning_relaxes_unknown_length_without_confirming_serial_gaps(self):
        class Graph:
            body=SimpleNamespace(c_rise=0.)
            node_region=np.array([0,1,2,0]);node_factor=np.ones(4);node_height=np.full(4,.025)
            node_xy=np.array([[0.,0.],[1.5,0.],[3.,0.],[4.,0.]])
            wall=False
            def mission_nodes(self,region,confirmed=False):return [3] if region.get('id')=='far' else [0]
            def edges(self,u,confirmed):
                for v in [u-1,u+1]:
                    if v<0 or v>3 or (self.wall and {u,v}=={1,2}):continue
                    yield v,1.,tuple(sorted(set(self.node_region[[u,v]])-{0}))
        g=Graph();c=TaskContext.__new__(TaskContext);c.opt={};c.prior={'entry':{},'goals':[{'id':'far'}]};c.decisions={'far':{'verdict':'unverified'}};queries=[]
        c.query=lambda key,feet,goals,kind:queries.append((key,goals,kind))
        c.anchored_route_queries({'graph':g})
        self.assertEqual(c.soft_route_stats['goals_with_value_only_path'],1)
        self.assertEqual(len(queries),2);self.assertTrue(all(q[1]['far']==.5 for q in queries));self.assertEqual(c.decisions['far']['verdict'],'unverified')
        g.wall=True;queries.clear();c.anchored_route_queries({'graph':g})
        self.assertEqual(c.soft_route_stats['goals_with_value_only_path'],0);self.assertFalse(queries)
    def test_serial_sections_have_a_single_goal_progress_budget(self):
        from covor.explore.entry_objective import pooled_target_values,joint_marginal
        cfg={'utility':{'task_progress':{'anchored_weight':1.,'partial_evidence_weight':.15}}}
        def item(key):return dict(task_query_predictions={key:dict(kind='anchored',goals={'g':.5},foot=[1.],body=[1.],foot_before=0.,body_before=0.)})
        a=item('door');b=item('landing')
        self.assertEqual(pooled_target_values([a],cfg)['g'],.5)
        self.assertEqual(pooled_target_values([a,b],cfg)['g'],1.)
        self.assertEqual(joint_marginal(a,[a],cfg),0.)
        self.assertEqual(joint_marginal(b,[a],cfg),.5)
    def test_past_repeated_views_reduce_entry_value_without_changing_unknown(self):
        from covor.explore.entry_objective import augment
        cfg={'utility':{'entry_objective':'task_progress_v2','task_progress':{'completion_weight':1.,'channel_aggregation':'max','view_novelty_radius_m':.45,'view_novelty_scale':6.}},'sensor':{'baseline':.095},'candidates':{'pitches':[-.65,.17]}}
        analysis={'_task_context':SimpleNamespace(queries=[]),'_observed_views':[([0,0,1],0.)]*12}
        def item(p):return dict(candidate=dict(position=p,yaw=0),unknown={1,2},targets={'g':1.},evidence={})
        out=augment([item([0,0,1]),item([1,0,1])],analysis,None,None,None,None,None,cfg,None)
        self.assertAlmostEqual(out[0]['targets']['g'],1/3)
        self.assertEqual(out[1]['targets']['g'],1.)
        self.assertEqual(out[0]['unknown'],out[1]['unknown'])
    def test_same_observation_repeated_is_deterministic_and_no_truth_api(self):
        c=self.context();q=Query('repeat',np.array([[.5,.5,2.]]),np.array([[.5,.5,2.4]]),np.zeros(1),np.zeros(1),np.ones(1),{'g':1.},'goal')
        self.assertEqual(c.observation_value(q,[np.eye(4)],c.sd),c.observation_value(q,[np.eye(4)],c.sd))
        self.assertNotIn('truth',str(inspect.signature(TaskContext)))
        self.assertNotIn('building',str(inspect.signature(TaskContext)))
    def test_useful_progress_exists_without_a_complete_possible_goal_route(self):
        from covor.explore.human import Human
        from covor.explore.policy import score_pool
        from covor.explore.depth_observation import camera_for
        cfg=json.loads(Path('configs/fire_recon_20261107.json').read_text())
        cfg['utility']['entry_objective']='task_progress_v1'
        cfg['utility']['task_progress']=dict(completion_weight=.8,boundary_weight=1.,goal_weight=.35,exploration_weight=.35,stability_weight=0.)
        shape=(90,30,35);lab=np.zeros(shape,np.uint8);lab[:,:,0]=2;lab[:40,:,1:30]=1
        layer=dict(support=np.zeros(shape,bool),height=np.full(shape,np.nan,np.float32),votes=np.zeros(shape,np.uint16),evidence=np.zeros(shape,np.float32),height_std=np.zeros(shape,np.float32),surface_x=np.full(shape,np.nan,np.float32),surface_y=np.full(shape,np.nan,np.float32),hit_z_max=np.full(shape,np.nan,np.float32))
        layer['support'][:40,:,0]=True;layer['height'][:40,:,0]=.1;layer['votes'][:40,:,0]=4;layer['evidence'][:40,:,0]=4;layer['hit_z_max'][:,:,0]=.1
        prior=dict(entry=dict(xy_bounds=[.4,.7,1.,2.2],floor_z_bounds=[.05,.25]),goals=[dict(id='far',xy_bounds=[7.,.7,8.,2.2],floor_z_bounds=[.05,.25])],exploration_box=[[.1,.1,.2],[8.9,2.9,3.]])
        analysis=Human(prior,cfg).analyze(lab,layer,np.zeros(3,int),True)
        self.assertEqual(analysis['decisions'][0]['verdict'],'unverified');self.assertFalse(analysis['impact'])
        pool=[dict(id='frontier',position=[3.,1.5,1.2],yaw=0.,pitch=-.65,kind='frontier')]
        result=score_pool(pool,analysis,lab,layer,np.zeros(3,int),.1,camera_for(cfg),cfg,prior)
        self.assertGreater(result[0]['targets'].get('far',0.),0.)
        self.assertEqual(analysis['decisions'][0]['verdict'],'unverified')


if __name__=='__main__':unittest.main()
