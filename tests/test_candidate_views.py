import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from types import SimpleNamespace
from covor.explore.candidate_views import refine_body_views


class CommonViewControls(unittest.TestCase):
    def test_mission_relative_heights_use_public_floor_intervals_only(self):
        from covor.explore.policy import candidates
        from covor.explore.motion import FlightMap
        lab=np.zeros((50,50,80),np.uint8);lab[5:45,5:45,2:75]=1
        layer=dict(votes=np.zeros(lab.shape,np.uint16),support=np.zeros(lab.shape,bool),evidence=np.zeros(lab.shape))
        flight=FlightMap(lab,np.zeros(3,int),.1,{'radius':.14,'margin':.04})
        cfg={'candidates':dict(heights=[.85,1.25,1.85],height_mode='mission_floor_offsets',max_total=24,max_reobserve=0,pitches=[-.65,.17])}
        prior=dict(exploration_box=[[0,0,0],[5,5,8]],goals=[dict(floor_z_bounds=[-.05,.05]),dict(floor_z_bounds=[3.15,3.25])])
        pool=candidates(lab,np.zeros(3,int),.1,layer,flight,[dict(p=np.array([1.25,1.25,1.25]))],cfg,0,0,prior)
        self.assertTrue(pool)
        allowed=np.array([.85,1.25,1.85,4.05,4.45,5.05])
        self.assertTrue(all(np.min(abs(allowed-c['position'][2]))<.051 for c in pool))
        self.assertTrue(all(flight.point_clear(c['position']) for c in pool))
        cfg['candidates']['min_view_clearance_m']=0.
        zero=candidates(lab,np.zeros(3,int),.1,layer,flight,[dict(p=np.array([1.25,1.25,1.25]))],cfg,0,0,prior)
        self.assertEqual(pool,zero)
        cfg['candidates']['min_view_clearance_m']=.6
        stand=candidates(lab,np.zeros(3,int),.1,layer,flight,[dict(p=np.array([1.25,1.25,1.25]))],cfg,0,0,prior)
        self.assertTrue(stand)
        self.assertTrue(all(c['observed_view_clearance_m']>=.6 for c in stand))

    def test_disabled_is_exact_original_pool(self):
        pool=[dict(id='a',position=[0,0,1],yaw=2.)]
        out=refine_body_views(pool,None,None,None,None,None,None,{'candidates':{}},None)
        self.assertIs(out,pool)
    def test_positions_count_and_common_deterministic_selection(self):
        class Context:
            queries=[SimpleNamespace(kind='boundary',key='observed_patch',body=np.array([[2.,0.,.5]]))]
            def observation_value(self,q,views,sd):
                # Optical forward x rewards yaw0, while yawpi/2 sees sideways.
                value=max(float(v[0,2]) for v in views)
                return max(0.,value),{}
        pool=[dict(id='a',position=[0.,0.,1.],yaw=np.pi/2),dict(id='b',position=[0.,1.,1.],yaw=np.pi/2)]
        original=copy.deepcopy(pool);cfg={'candidates':{'body_reobserve_views':True,'max_reobserve':2,'pitches':[-.65,.17]},'sensor':{'baseline':.095}}
        analysis={'_candidate_task_context':Context()}
        a=refine_body_views(pool,analysis,None,None,None,None,None,cfg,None);b=refine_body_views(pool,analysis,None,None,None,None,None,cfg,None)
        self.assertEqual(a,b);self.assertEqual(len(a),len(pool));self.assertEqual(pool,original)
        self.assertEqual([x['position'] for x in a],[x['position'] for x in pool]);self.assertEqual(a[0]['yaw'],0.)
        self.assertGreater(a[0]['visibility_after'],a[0]['visibility_before'])
    def test_new_viewpoint_positions_are_in_connected_observed_free_only(self):
        from covor.explore.motion import FlightMap
        class Context:
            queries=[SimpleNamespace(kind='boundary',key='patch',feet=np.array([[2.,2.,.002]]),body=np.array([[2.,2.,.35]]))]
            def observation_value(self,q,views,sd):return 1.,{}
        lab=np.zeros((50,50,30),np.uint8);lab[:35,:35,1:25]=1
        motion={'radius':.14,'margin':.04};flight=FlightMap(lab,np.zeros(3,int),.1,motion)
        cfg={'candidates':{'body_reobserve_views':True,'evidence_viewpoint_positions':True,'evidence_proposal_budget':96,'max_reobserve':2,'pitches':[-.65,.17]},'sensor':{'baseline':.095}}
        pool=[dict(id=f'c{i:03d}',position=[1.,1.,1.15],yaw=0.) for i in range(4)]
        result=refine_body_views(pool,{'_candidate_task_context':Context()},lab,None,None,None,None,cfg,None,flight=flight,states=[{'p':np.array([1.,1.,1.15])}])
        self.assertEqual(len(result),len(pool));self.assertTrue(any(c.get('orientation_source')=='common_observed_evidence_viewpoint' for c in result))
        self.assertTrue(all(flight.point_clear(c['position']) for c in result))

    def test_path_repair_budget_is_shared_round_robin_and_deterministic(self):
        from covor.explore.motion import FlightMap
        class Context:
            queries=[SimpleNamespace(kind='anchored',key=key,feet=np.array([[x,2.,.002]]),
                body=np.array([[x,2.,.35]]),foot_known=np.zeros(1),body_known=np.zeros(1))
                for key,x in [('a',2.),('b',2.5)]]
            def point_probabilities(self,points,views,sd,foot=False):return np.ones(len(points))
        lab=np.ones((50,50,30),np.uint8);flight=FlightMap(lab,np.zeros(3,int),.1,{'radius':.14,'margin':.04})
        cfg={'candidates':{'body_reobserve_views':True,'path_repair_views':True,'evidence_proposal_budget':2,'max_reobserve':2,'pitches':[-.65,.17]},'sensor':{'baseline':.095}}
        pool=[dict(id=f'c{i:03d}',position=[1.,1.,1.15],yaw=0.) for i in range(4)]
        original=copy.deepcopy(pool);kwargs=dict(flight=flight,states=[{'p':np.array([1.,1.,1.15]),'failed':False}])
        a=refine_body_views(pool,{'_candidate_task_context':Context()},lab,None,None,None,None,cfg,None,**kwargs)
        b=refine_body_views(pool,{'_candidate_task_context':Context()},lab,None,None,None,None,cfg,None,**kwargs)
        self.assertEqual(a,b);self.assertEqual(pool,original);self.assertEqual(len(a),len(pool))
        self.assertEqual({c['target_query'] for c in a if c.get('kind')=='path_repair'},{'a','b'})
        self.assertTrue(all(flight.point_clear(c['position']) for c in a))


if __name__=='__main__':unittest.main()
