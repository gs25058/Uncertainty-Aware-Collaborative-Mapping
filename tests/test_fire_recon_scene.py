"""Regression checks for isolated, opt-in scene repair and asset preservation."""
import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from covor.explore.fire_world import FireWorld
from covor.explore.fire_recon_scene import ReconFireWorld
class RepairTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cfg=json.load(open('configs/fire_recon_20261107.preflight.json'));cls.original=FireWorld(1,cls.cfg);cls.repaired=ReconFireWorld(1,cls.cfg)
 def test_original_geometry_and_fire_sources_are_preserved(self):
  a=self.original;b=self.repaired;self.assertEqual(len(b.components),len(a.components)+2)
  for x,y in zip(a.components,b.components):
   self.assertEqual(x['name'],y['name']);np.testing.assert_array_equal(x['vertices'],y['vertices']);np.testing.assert_array_equal(x['faces'],y['faces']);np.testing.assert_array_equal(x['color'],y['color'])
  self.assertEqual(a.fires,b.fires);self.assertEqual(a.goals,b.goals);self.assertEqual(a.entry,b.entry)
 def test_repair_is_opt_in_and_reference_key_changes(self):
  c=copy.deepcopy(self.cfg);c['scene']['repair_bottom_landings']=False;plain=ReconFireWorld(1,c)
  self.assertEqual(len(plain.components),len(self.original.components));self.assertEqual(plain.geometry_repairs,[]);self.assertNotEqual(plain.geometry_hash(),self.repaired.geometry_hash())
 def test_landings_bridge_the_actual_hole_and_first_tread(self):
  import open3d as o3d
  b=self.repaired;b.render_scene(False)
  for s in [s for s in b.stairs if s['floor']==1]:
   ray=np.array([[s['x']+.625,s['y']-.15,3.38,0,0,-1]],np.float32);hit=b.scene.cast_rays(o3d.core.Tensor(ray))['t_hit'].numpy()[0]
   self.assertAlmostEqual(3.38-float(hit),3.3,places=5)
  self.assertEqual(len(b.geometry_repairs),2)
if __name__=='__main__':unittest.main()
