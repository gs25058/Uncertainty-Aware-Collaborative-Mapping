"""Observed-map, continuous-state controls; no reference scene is used."""
import inspect,pickle,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from covor.explore.flight_control import (BrakingTrajectory,WholeVoxelFlightMap,
    follow_heading,guarded_stop,norm_extrema,observed_free_until,split_rest_points)
from covor.explore.motion import FlightMap,Hover,reservation_conflict

CFG=dict(radius=.14,margin=.04,vmax=1.,amax=1.5,yaw_rate=np.pi/2,pitch_rate=np.pi/3,reservation_dt=.1,max_wait=8.)


class Published:
    start=0.;wait=0.;end=5.;durations=[5.];stop_at=None
    def sample(self,t,derivative=0):
        return np.array([1.+.8*t,1.,1.]) if derivative==0 else np.array([.8,0,0]) if derivative==1 else np.zeros(3)


class FlightControlTests(unittest.TestCase):
    def test_brake_preserves_position_velocity_acceleration(self):
        p=np.array([1.,1.,1.]);v=np.array([.65,.1,.05]);a=np.array([.1,-.04,0.])
        tr=BrakingTrajectory(p,v,a,p+v*.75,1.5,4.,0.,-.65,CFG)
        for derivative,expected in enumerate([p,v,a]):np.testing.assert_allclose(tr.sample(4.,derivative),expected,atol=1e-12)
        np.testing.assert_allclose(tr.sample(tr.end),p+v*.75)
        np.testing.assert_allclose(tr.sample(tr.end,1),0.);np.testing.assert_allclose(tr.sample(tr.end,2),0.)
        self.assertTrue(tr.within_limits())
        restored=pickle.loads(pickle.dumps(tr));np.testing.assert_array_equal(restored.sample(4.37),tr.sample(4.37))

    def test_norm_extrema_includes_constant_axes_and_internal_peaks(self):
        c=np.array([[0.,.7,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
        self.assertAlmostEqual(norm_extrema(c,2.,1),.7)
        tr=BrakingTrajectory([0,0,0],[.8,0,0],[0,0,0],[.4,0,0],1.,0.,0.,0.,CFG)
        for d in (1,2):
            sampled=max(np.linalg.norm(tr.sample(t,d)) for t in np.linspace(0,tr.total,2001))
            self.assertGreaterEqual(norm_extrema(tr.coefficients,tr.total,d)+1e-10,sampled)
            self.assertLess(norm_extrema(tr.coefficients,tr.total,d)-sampled,1e-4)

    def test_new_obstacle_can_be_braked_before_distant_rest_node(self):
        lab=np.ones((60,30,30),np.uint8);lab[20,:,:]=2
        flight=WholeVoxelFlightMap(lab,np.zeros(3,int),.1,CFG);old=Published()
        state=dict(yaw=0.,pitch=-.65)
        stopped,kind=guarded_stop(old,state,0.,[],flight,CFG)
        self.assertIsNotNone(stopped);self.assertEqual(kind,'continuous_brake')
        self.assertFalse(observed_free_until(old,0.,old.end,flight))
        self.assertTrue(observed_free_until(stopped,0.,stopped.end,flight))
        np.testing.assert_allclose(stopped.sample(0.,1),old.sample(0.,1));self.assertTrue(stopped.within_limits())
        self.assertLess(stopped.sample(stopped.end)[0],1.8)

    def test_unknown_or_peer_conflict_is_not_repaired_with_reference_geometry(self):
        lab=np.zeros((60,30,30),np.uint8);flight=WholeVoxelFlightMap(lab,np.zeros(3,int),.1,CFG)
        tr,why=guarded_stop(Published(),dict(yaw=0.,pitch=0.),0.,[],flight,CFG)
        self.assertIsNone(tr);self.assertEqual(why,'no_known_free_joint_safe_brake')
        lab[:]=1;lab[20,:,:]=2;flight=WholeVoxelFlightMap(lab,np.zeros(3,int),.1,CFG)
        tr,_=guarded_stop(Published(),dict(yaw=0.,pitch=0.),0.,[Hover([1.1,1,1])],flight,CFG)
        self.assertIsNone(tr)

    def test_whole_voxel_clearance_protects_margin_near_corner(self):
        lab=np.ones((30,30,30),np.uint8);lab[15,15,15]=2
        old=FlightMap(lab,np.zeros(3,int),.1,CFG);new=WholeVoxelFlightMap(lab,np.zeros(3,int),.1,CFG)
        self.assertTrue(old.safe[13,13,15]);self.assertFalse(new.safe[13,13,15])
        self.assertTrue(np.all(~new.safe|old.safe));self.assertIsInstance(new.avoid([]),WholeVoxelFlightMap)

    def test_rate_limited_heading_and_stationary_task_orientation(self):
        state=dict(yaw=np.pi-.02,traj=None,item={'candidate':{'yaw':-np.pi+.02}})
        self.assertAlmostEqual(follow_heading(state,0,.25,CFG),np.pi+.02)
        state.update(yaw=np.pi/2,traj=Published())
        yaw=follow_heading(state,0,.25,CFG);self.assertAlmostEqual(yaw,np.pi/2-CFG['yaw_rate']*.25)

    def test_rest_nodes_do_not_change_spatial_path_or_teleport(self):
        points=[[0,0,0],[3,0,0],[3,0,2]];p=np.array(split_rest_points(points,1.25))
        np.testing.assert_allclose(p[0],points[0]);np.testing.assert_allclose(p[-1],points[-1])
        self.assertLessEqual(np.linalg.norm(np.diff(p,axis=0),axis=1).max(),1.25)
        self.assertAlmostEqual(np.linalg.norm(np.diff(p,axis=0),axis=1).sum(),5.)

    def test_control_interfaces_have_no_truth_or_renderer(self):
        for fn in (guarded_stop,follow_heading,observed_free_until):
            signature=str(inspect.signature(fn));self.assertNotIn('truth',signature);self.assertNotIn('scene',signature)


if __name__=='__main__':unittest.main()
