import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from covor.explore.contact_geometry import ContactGeometry
from covor.entry.decision import DecisionGraph


class LocalContactControls(unittest.TestCase):
    def layer(self):
        shape=(25,25,30);lab=np.ones(shape,np.uint8);lab[10,10,0]=2
        layer=dict(support=np.zeros(shape,bool),height=np.full(shape,np.nan),votes=np.zeros(shape,np.uint16),evidence=np.zeros(shape),height_std=np.zeros(shape),surface_x=np.full(shape,np.nan),surface_y=np.full(shape,np.nan),hit_z_max=np.full(shape,-np.inf))
        q=(10,10,0);layer['support'][q]=True;layer['height'][q]=0.;layer['votes'][q]=2;layer['evidence'][q]=2.
        layer['surface_x'][q]=layer['surface_y'][q]=1.05;layer['hit_z_max'][q]=.08
        return lab,layer,q

    def graph(self,lab,layer):
        return DecisionGraph(lab,np.zeros(3,int),.1,support_layer=layer,surface_geometry=True,resolve_low_surface_band=True,use_surface_xy=True)

    def test_off_centre_low_obstacle_does_not_occupy_centre_foot(self):
        lab,layer,q=self.layer();self.assertFalse(self.graph(lab,layer).confirmed[q])
        fine=ContactGeometry();fine.integrate_world(np.array([[1.05,1.05,0.],[1.091,1.091,.08]]))
        layer['contact_hit_z_max']=fine.dense(layer,np.zeros(3,int))
        self.assertEqual(layer['hit_z_max'][q],.08)
        self.assertEqual(layer['contact_hit_z_max'][q],0.)
        self.assertTrue(self.graph(lab,layer).confirmed[q])

    def test_obstacle_at_foot_and_unknown_contact_evidence_remain_rejected(self):
        lab,layer,q=self.layer();fine=ContactGeometry();fine.integrate_world(np.array([[1.05,1.05,0.],[1.051,1.051,.08]]))
        layer['contact_hit_z_max']=fine.dense(layer,np.zeros(3,int))
        self.assertFalse(self.graph(lab,layer).confirmed[q])
        layer['contact_hit_z_max']=ContactGeometry().dense(layer,np.zeros(3,int))
        self.assertFalse(self.graph(lab,layer).confirmed[q])

    def test_full_torso_upper_bound_is_still_enforced(self):
        lab,layer,q=self.layer();fine=ContactGeometry();fine.integrate_world(np.array([[1.05,1.05,0.],[1.091,1.091,.08]]))
        other=(10,11,2);lab[other]=2;layer['support'][other]=True;layer['height'][other]=.2;layer['votes'][other]=2;layer['evidence'][other]=2;layer['hit_z_max'][other]=.28
        layer['contact_hit_z_max']=fine.dense(layer,np.zeros(3,int))
        self.assertFalse(self.graph(lab,layer).confirmed[q])


if __name__=='__main__':unittest.main()
