import sys,json,unittest,copy,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from unittest.mock import patch
from scripts.explore.validate_depth_model import Fixture
from covor.explore.depth_observation import RayDepthSensor,camera_for,correlated_field,ray_limit
from covor.explore.sensing import Sensor

CFG=json.load(open('configs/depth_model_20261106.json'))
class DepthObservationTests(unittest.TestCase):
 def sensor(self,kind='plane',d=2.,condition='clean',cfg=None):return RayDepthSensor(Fixture(kind,d),cfg or CFG,11000,condition)
 def test_no_stereo_and_no_truth_in_primary_cache(self):
  with tempfile.TemporaryDirectory() as out,patch('cv2.StereoSGBM_create',side_effect=AssertionError('unused stereo')):
   s=Sensor(Fixture('plane',2.),CFG,11000,'clean');s.capture([0,0,0],0,0,0,[],out,'frame',drone_id=0)
   q=np.load(Path(out)/'frame.npz');self.assertEqual(set(q.files),{'Z','sigma','valid','T','time','drone_id','teammates'});self.assertFalse(list(Path(out).glob('*.png')))
 def test_z_and_ray_length_and_anisotropic_intrinsics(self):
  sd=camera_for(CFG);self.assertNotEqual(sd.fx,sd.fy);z=np.full((180,320),2.);v=np.zeros(z.shape,bool);v[90,200]=True
  p,_=sd.backproject(z,z*.001,v,1);self.assertEqual(p[0,2],2.);self.assertGreater(np.linalg.norm(p[0]),2.);self.assertGreater(ray_limit(CFG),6.)
 def test_occlusion_and_range(self):
  for d in (.5,7.):self.assertFalse(self.sensor(d=d).measure(np.eye(4),0,0)[2].any())
  (_,_,_),a=self.sensor('wall').measure(np.eye(4),0,0,audit=True);self.assertTrue(np.allclose(a['true_z'][a['geometric']],2.))
  (_,_,_),a=self.sensor('occlusion').measure(np.eye(4),0,0,audit=True);self.assertLess(a['geometric'].sum(),a['left_geometric'].sum());self.assertTrue(a['geometric'][:,50].any())
 def test_teammate_is_first_hit(self):
  (_,_,_),a=self.sensor().measure(np.eye(4),0,0,[[0,0,1]],audit=True);self.assertAlmostEqual(float(a['true_z'][90,160]),.86,places=5)
 def test_sigma_does_not_scale_with_grid_resolution(self):
  for w,h in ((160,90),(320,180)):
   cfg=copy.deepcopy(CFG);cfg['sensor']['width']=w;cfg['sensor']['height']=h
   (_,_,_),a=self.sensor(cfg=cfg).measure(np.eye(4),0,0,audit=True)
   normal=~a['edge'].flat[a['ids']];self.assertTrue(np.allclose(a['sigma_true'][normal],.006))
 def test_stateless_replay_and_persistent_bias(self):
  s=self.sensor();a,priv=s.measure(np.eye(4),.25,0,audit=True);s.measure(np.eye(4),40,3);b,priv2=s.measure(np.eye(4),.25,0,audit=True)
  self.assertTrue(np.array_equal(a[0],b[0],equal_nan=True));self.assertEqual(priv['bias'],priv2['bias']);_,p=s.measure(np.eye(4),3,0,audit=True);self.assertEqual(p['bias'],priv['bias'])
 def test_correlations_and_unit_variance(self):
  rng=np.random.default_rng(72);x=rng.uniform(-5,5,(12000,3));f=lambda ch,t=0,xyz=x:correlated_field(xyz,t,11000,ch)
  common=f(11);a=.5*common+np.sqrt(.55)*f(1000)+np.sqrt(.2)*rng.normal(size=len(x));b=.5*common+np.sqrt(.55)*f(1031)+np.sqrt(.2)*rng.normal(size=len(x))
  self.assertTrue(.95<a.std()<1.05);self.assertTrue(.15<np.corrcoef(a,b)[0,1]<.35)
  self.assertGreater(np.corrcoef(common,f(11,.25))[0,1],.7);self.assertLess(abs(np.corrcoef(common,f(11,1.5))[0,1]),.05)
  self.assertGreater(np.corrcoef(common,f(11,xyz=x+[.03,0,0]))[0,1],np.corrcoef(common,f(11,xyz=x+[.3,0,0]))[0,1])
 def test_invalid_is_not_clamped(self):
  z,sz,v=self.sensor(d=6.,condition='strong').measure(np.eye(4),0,0);self.assertTrue(np.isnan(z[~v]).all());self.assertTrue(np.isnan(sz[~v]).all());self.assertTrue((z[v]<=6).all());self.assertFalse((z[v]==6).any())
 def test_degradation_order_and_shared_environment(self):
  records=[]
  for condition in ('clean','weak','medium','strong'):
   public,private=self.sensor(d=4.,condition=condition).measure(np.eye(4),.5,0,audit=True);records.append((public,private))
  clean=records[0][1]
  for i,condition in enumerate(('weak','medium','strong'),1):
   public,a=records[i];k=CFG['sensor']['error_model']['degradation'][condition]['k']
   self.assertTrue(np.allclose(a['sigma_true'],clean['sigma_true']*(1+k*(1-a['transmission']))))
   self.assertTrue(np.all(~clean['drop']|a['drop']))
   self.assertGreater(public[2].sum()/a['geometric'].sum(),.7)
  self.assertTrue(np.array_equal(records[1][1]['transmission'],records[3][1]['transmission']))
 def test_ideal_backend_has_zero_error_and_uncertainty(self):
  cfg=copy.deepcopy(CFG);cfg['sensor']['backend']='ideal_depth';s=self.sensor(cfg=cfg);(z,sz,v),a=s.measure(np.eye(4),0,0,audit=True)
  self.assertTrue(np.array_equal(v,a['geometric']));self.assertTrue(np.array_equal(z[v],a['true_z'][v]));self.assertTrue((sz[v]==0).all())
if __name__=='__main__':unittest.main()
