import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import json,copy
import tempfile
from covor.entry.decision import DecisionGraph
from covor.explore.human import Human
from covor.explore.sensing import SharedMap,camera_pose
from covor.explore.depth_observation import camera_for
from covor.explore.scene import Building
from covor.entry.surfaces import SupportSurfaces


def stairs():
    lab=np.ones((20,35,40),np.uint8)
    layer=dict(support=np.zeros(lab.shape,bool),height=np.full(lab.shape,np.nan),height_std=np.zeros(lab.shape),votes=np.zeros(lab.shape,np.uint16),evidence=np.zeros(lab.shape))
    for i in range(20):
        for j in range(35):
            h=0 if j<8 else min(4,(j-8)//3+1)*.165;k=int(np.floor(h/.1));lab[i,j,:k+1]=2
            layer['support'][i,j,k]=True;layer['height'][i,j,k]=h;layer['votes'][i,j,k]=2;layer['evidence'][i,j,k]=2
    return lab,layer


def path(lab,layer,enabled=True):
    g=DecisionGraph(lab,[0,0,0],.1,support_layer=layer,surface_geometry=True,foot_contact_offset=.025,resolve_low_surface_band=enabled)
    a,b=int(g.ids[10,4,0]),int(g.ids[10,24,6]);p,_=g.path(g.search([a],True),[b]);return bool(p)


class StairEvidence(unittest.TestCase):
    def test_measured_165mm_steps_cross_two_voxel_rows(self):
        lab,layer=stairs();self.assertFalse(path(lab,layer,False));self.assertTrue(path(lab,layer))

    def test_unknown_band_still_blocks(self):
        lab,layer=stairs();lab[:,9:11,3]=0;self.assertFalse(path(lab,layer))

    def test_tall_observed_wall_is_not_resolved_as_a_floor(self):
        lab,layer=stairs();lab[:,12:14,:30]=2;self.assertFalse(path(lab,layer))

    def test_high_surface_scatter_requires_more_evidence(self):
        lab,layer=stairs();layer['height_std'][:]=.04;self.assertFalse(path(lab,layer))

    def test_unconfirmed_surfaces_never_clear_body(self):
        lab,layer=stairs();layer['support'][:,11:14,:]=False;self.assertFalse(path(lab,layer))

    def test_observed_xy_is_exported_and_intruding_hits_are_not_erased(self):
        lab=np.ones((25,25,35),np.uint8)
        layer=dict(support=np.zeros(lab.shape,bool),height=np.full(lab.shape,np.nan),height_std=np.zeros(lab.shape),surface_x=np.full(lab.shape,np.nan),surface_y=np.full(lab.shape,np.nan),hit_z_max=np.full(lab.shape,np.nan),votes=np.zeros(lab.shape,np.uint16),evidence=np.zeros(lab.shape))
        q=(10,10,0);layer['support'][q]=True;layer['height'][q]=.05;layer['surface_x'][q]=1.09;layer['surface_y'][q]=1.05;layer['hit_z_max'][q]=.05;layer['votes'][q]=2;layer['evidence'][q]=2;lab[q]=2
        g=DecisionGraph(lab,[0,0,0],.1,support_layer=layer,surface_geometry=True,foot_contact_offset=.025,resolve_low_surface_band=True,use_surface_xy=True)
        mission=dict(xy_bounds=[1.085,1.0,1.095,1.1],floor_z_bounds=[0,.2]);nodes=g.mission_nodes(mission,True);self.assertTrue(nodes)
        out,_=g.path(g.search(nodes,True),nodes);self.assertAlmostEqual(out[0]['xyz'][0],1.09)
        layer['hit_z_max'][q]=.09
        g=DecisionGraph(lab,[0,0,0],.1,support_layer=layer,surface_geometry=True,foot_contact_offset=.025,resolve_low_surface_band=True,use_surface_xy=True);self.assertFalse(g.confirmed[q])

    def test_observed_graph_crop_preserves_nodes_paths_and_impacts(self):
        shape=(60,60,45);lab=np.zeros(shape,np.uint8);lab[10:45,10:45,4:28]=1
        layer=dict(support=np.zeros(shape,bool),height=np.full(shape,np.nan),height_std=np.zeros(shape),surface_x=np.full(shape,np.nan),surface_y=np.full(shape,np.nan),hit_z_max=np.full(shape,np.nan),votes=np.zeros(shape,np.uint16),evidence=np.zeros(shape))
        for i in range(20,30):
            for j in range(20,30):
                q=(i,j,5);lab[q]=2;layer['support'][q]=True;layer['height'][q]=.05;layer['surface_x'][q]=(i-5+.5)*.1;layer['surface_y'][q]=(j-5+.5)*.1;layer['hit_z_max'][q]=.05;layer['votes'][q]=2;layer['evidence'][q]=2
        prior=dict(entry=dict(xy_bounds=[1.5,1.5,1.7,1.7],floor_z_bounds=[0,.2]),goals=[dict(id='room',xy_bounds=[2.2,2.2,2.4,2.4],floor_z_bounds=[0,.2])])
        cfg=json.loads(Path('configs/fire_closed_loop_20261107.json').read_text());cfg['human']['crop_observed_graph']=False
        a=Human(prior,cfg).analyze(lab,layer,np.array([-5,-5,-5]));cfg=copy.deepcopy(cfg);cfg['human']['crop_observed_graph']=True;b=Human(prior,cfg).analyze(lab,layer,np.array([-5,-5,-5]))
        self.assertEqual(a['decisions'],b['decisions']);self.assertEqual(a['impact'],b['impact']);self.assertEqual(a['n_possible'],b['n_possible'])
        np.testing.assert_array_equal(a['graph'].node_xy,b['graph'].node_xy)

    def test_depth_arrays_integrate_with_xy_and_hit_height_and_roundtrip(self):
        cfg=json.loads(Path('configs/fire_closed_loop_20261107.json').read_text());m=SharedMap(Building(11).prior(),cfg);sd=camera_for(cfg)
        z=np.full((180,320),1.2,np.float32);sigma=np.full_like(z,.003);valid=np.ones(z.shape,bool);pose=camera_pose(np.array([2.,2.,1.2]),0,-np.pi/2,.095)
        for _ in range(2):m.integrate(sd,pose,z,sigma,valid,[])
        _,layer=m.dense();self.assertEqual(m.version,2);self.assertTrue(layer['support'].any());self.assertTrue(np.isfinite(layer['surface_x']).any());self.assertTrue(np.isfinite(layer['hit_z_max']).any())
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'surfaces.npz';m.surfaces.save(p);s=SupportSurfaces.load(p);other=s.dense(m.ijk,m.shape)
            np.testing.assert_array_equal(layer['surface_x'],other['surface_x']);np.testing.assert_array_equal(layer['height_std'],other['height_std'])

if __name__=='__main__':unittest.main()
