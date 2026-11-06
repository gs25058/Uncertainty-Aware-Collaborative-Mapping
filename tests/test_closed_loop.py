"""Controls for motion, truth separation and human blocked-route semantics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,unittest
import numpy as np
from covor.explore.scene import Building
from covor.explore.motion import Trajectory,Hover,reservation_conflict,FlightMap
from covor.explore.policy import marginal,predicted_unknown
from covor.explore.sensing import camera_pose

CFG=json.load(open('configs/closed_loop_20261105.json'))
class ClosedLoopControls(unittest.TestCase):
    def test_native_fuel_limits_and_endpoints(self):
        tr=Trajectory([[0,0,0],[2,1,1]],0,-.65,2.,.17,0,CFG['motion'])
        self.assertTrue(np.allclose(tr.sample(0),[0,0,0]));self.assertTrue(np.allclose(tr.sample(tr.end),[2,1,1]))
        self.assertLessEqual(max(np.linalg.norm(tr.sample(t,1)) for t in np.linspace(0,tr.end,501)),CFG['motion']['vmax']+1e-8)
        self.assertLessEqual(max(np.linalg.norm(tr.sample(t,2)) for t in np.linspace(0,tr.end,501)),CFG['motion']['amax']+1e-3)
        self.assertTrue(np.allclose(tr.sample(0,1),0));self.assertTrue(np.allclose(tr.sample(tr.end,1),0))
    def test_swap_between_samples_is_reserved(self):
        c=dict(CFG['motion'],reservation_dt=.1)
        a=Trajectory([[-1,0,0],[1,0,0]],0,0,0,0,0,c)
        b=Trajectory([[1,0,0],[-1,0,0]],0,0,0,0,0,c)
        self.assertTrue(reservation_conflict(a,[b],c))
        self.assertFalse(reservation_conflict(a,[Hover([0,2,0])],c))
    def test_unknown_not_flight_free_and_wall_no_shortcut(self):
        lab=np.ones((30,30,30),np.uint8);lab[15,:,:]=2
        f=FlightMap(lab,np.zeros(3,int),.1,CFG['motion'])
        self.assertIsNone(f.route([1,1,1],[2,1,1]));lab[:]=0
        self.assertFalse(FlightMap(lab,np.zeros(3,int),.1,CFG['motion']).point_clear([1,1,1]))
    def test_duplicate_goal_count_and_complementary_votes(self):
        item=dict(unknown={1,2},targets={'a':.7,'b':.5},evidence={1:(.7,2)})
        gain,_=marginal(item,'entry',[],CFG,True);again,_=marginal(item,'entry',[item],CFG,True)
        self.assertGreater(gain,0);self.assertEqual(again,0)
        self.assertAlmostEqual(marginal(item,'unknown',[item],CFG)[0],0)
    def test_independent_geometry_has_all_fixed_cases(self):
        for seed in CFG['scene_seeds']:
            b=Building(seed);r,_,_=b.reference();truth={x['id']:x['reachable'] for x in r}
            self.assertEqual(truth,dict(g0=True,g1=False,g2=True,g3=True,g4=True,g5=False,g6=False))
            # Standing fits in the goal but its sole door is unstepable.
            self.assertTrue(b.body_clear([9.6,1.55,.055],0,0))
            self.assertFalse(b.body_clear([7.5,b.door_y[0],.055],0,0))
            # A real bypass exists for g2. A proposed direct shortcut through
            # its blocked main door remains a collision, despite that bypass.
            self.assertFalse(b.body_clear([3.5,4.2,.055],2,1))
            self.assertFalse(b.body_clear([7.5,4.2,.055],0,0))
            self.assertTrue(b.body_clear([7.5,4.2,.055],1,0))
    def test_camera_visibility_respects_effective_range(self):
        from types import SimpleNamespace
        from covor.entry.assignment import visible_fraction
        lab=np.ones((20,20,60),np.uint8);sd=SimpleNamespace(fx=20.,fy=20.,cx=50.,cy=50.,R1=np.eye(3),calib=SimpleNamespace(size=(100,100)))
        cells=np.array([[0,0,48]])
        self.assertEqual(visible_fraction(cells,lab,np.zeros(3,int),.1,np.eye(4),sd)['fraction'],1.)
        self.assertEqual(visible_fraction(cells,lab,np.zeros(3,int),.1,np.eye(4),sd,max_range_m=4.5)['fraction'],0.)
    def test_stopping_midroute_cannot_break_an_existing_reservation(self):
        from covor.explore.motion import safe_rest_stop
        import copy
        c=CFG['motion'];a=Trajectory([[-1,0,1],[0,0,1],[1,0,1]],0,0,0,0,0,c)
        b=Trajectory([[0,-1,1],[0,1,1]],0,0,0,0,1.5,c)
        self.assertFalse(reservation_conflict(a,[b],c))
        unsafe=copy.copy(a);unsafe.stop_at=a.durations[0];unsafe.end=unsafe.stop_at
        self.assertTrue(reservation_conflict(unsafe,[b],c))
        safe=safe_rest_stop(a,1.,[b],c)
        self.assertIsNotNone(safe);self.assertEqual(safe.end,a.end)
        self.assertFalse(reservation_conflict(safe,[b],c))
        self.assertTrue(np.allclose(safe.sample(safe.end,1),0))
    def test_entry_graph_refreshes_on_map_version_before_assignment(self):
        from covor.explore.human import ensure_current_analysis
        class Counter:
            calls=0
            def analyze(self,lab,layer,ijk,influence):
                self.calls+=1;return {'graph':None}
        human=Counter();old={'map_version':1};a,changed=ensure_current_analysis(human,old,None,None,None,2)
        self.assertTrue(changed);self.assertEqual(a['map_version'],2);self.assertEqual(human.calls,1)
        b,changed=ensure_current_analysis(human,a,None,None,None,2)
        self.assertFalse(changed);self.assertIs(a,b);self.assertEqual(human.calls,1)
    def test_map_only_policy_interface(self):
        import inspect
        from covor.explore import policy
        for fn in (policy.candidates,policy.score_pool,policy.marginal):
            sig=str(inspect.signature(fn))
            self.assertNotIn('truth',sig);self.assertNotIn('renderer',sig);self.assertNotIn('building',sig)
        from covor.explore.human import Human
        b=Building(11);lab,ijk=b.labels();shape=lab.shape
        layer=dict(support=np.zeros(shape,bool),height=np.full(shape,np.nan),votes=np.zeros(shape,np.uint16),evidence=np.zeros(shape))
        a=Human(b.prior(),CFG).analyze(lab,layer,ijk,False)
        self.assertTrue(all(d['verdict']=='unverified' for d in a['decisions']))
if __name__=='__main__':unittest.main()
