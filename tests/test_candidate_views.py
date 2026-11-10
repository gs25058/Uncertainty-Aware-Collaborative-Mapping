import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from types import SimpleNamespace
from covor.explore.candidate_views import refine_body_views


class CommonViewControls(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
