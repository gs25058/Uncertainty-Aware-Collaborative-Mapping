import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from covor.explore.completion_opportunity import CompletionOpportunity,missing_body_cells
from covor.explore.policy import marginal
from covor.entry.traverse import TravCfg


class CompletionControls(unittest.TestCase):
    def test_serial_body_requirements_must_all_be_visible(self):
        class Graph:
            xyz=np.zeros((4,3),int);node_known=np.array([True,False,False,True])
            def _raw_neighbors(self,u):return [(v,1.,[]) for v in {0:[1],1:[0,2],2:[1,3],3:[2]}[u]]
        model=CompletionOpportunity.__new__(CompletionOpportunity);model.g=Graph();model.indices={1:np.array([0]),2:np.array([1])}
        model.components=np.array([0,-1,-1,1]);model.root={0};model.targets={'goal':[3]}
        self.assertEqual(model.target_values(np.array([1.,0.])),{})
        self.assertEqual(model.target_values(np.array([.5,.8])),{'goal':.5})

    def test_missing_body_promises_do_not_open_observed_obstacles_or_bad_contact(self):
        shape=(25,25,30);lab=np.ones(shape,np.uint8);q=(10,10,0);lab[q]=2
        s=dict(support=np.zeros(shape,bool),height=np.full(shape,np.nan),height_std=np.zeros(shape),hit_z_max=np.zeros(shape))
        s['support'][q]=True;s['height'][q]=0.
        g=SimpleNamespace(xyz=np.array([q]),node_height=np.array([.025]),node_xy=np.array([[1.05,1.05]]),support_layer=s,lab=lab,res=.1,ijk=np.zeros(3,int),body=TravCfg(res=.1))
        lab[11,10,4]=0
        cells=missing_body_cells(g,0);self.assertIn([11,10,4],cells.tolist())
        lab[11,10,4]=2;self.assertIsNone(missing_body_cells(g,0))
        lab[11,10,4]=0;s['hit_z_max'][q]=.03;self.assertIsNone(missing_body_cells(g,0))
        s['hit_z_max'][q]=np.nan;self.assertIsNone(missing_body_cells(g,0))

    def test_fallback_is_exact_unknown_and_assigned_goals_are_not_recounted(self):
        cfg={'utility':{'entry_objective':'completion_opportunity_v1','unknown_normalizer':10.}}
        item=dict(unknown={1,2,3},targets={});allocated=[dict(unknown={1},targets={})]
        self.assertEqual(marginal(item,'entry',allocated,cfg,False)[0],marginal(item,'unknown',allocated,cfg,False)[0])
        item['targets']={'g':.5};self.assertEqual(marginal(item,'entry',[],cfg,True)[0],.5)
        self.assertEqual(marginal(item,'entry',[item],cfg,True)[0],0.)

    def test_real_human_graph_has_a_visible_complete_repair_opportunity(self):
        from covor.explore.human import Human
        from covor.explore.sensing import camera_pose
        shape=(40,13,32);lab=np.ones(shape,np.uint8);lab[:,:,0]=2;lab[:,:,28:]=2
        lab[:,0,:]=lab[:,-1,:]=2;lab[0,:,:]=lab[-1,:,:]=2;lab[20,1:-1,2:12]=0
        layer=dict(support=np.zeros(shape,bool),height=np.full(shape,np.nan),votes=np.zeros(shape,np.uint16),evidence=np.zeros(shape),height_std=np.zeros(shape),surface_x=np.full(shape,np.nan),surface_y=np.full(shape,np.nan),hit_z_max=np.full(shape,-np.inf))
        layer['support'][:,:,0]=True;layer['height'][:,:,0]=.05;layer['votes'][:,:,0]=3;layer['evidence'][:,:,0]=3;layer['hit_z_max'][:,:,0]=.05
        layer['surface_x'][:,:,0]=(np.arange(40)[:,None]+.5)*.1;layer['surface_y'][:,:,0]=(np.arange(13)[None,:]+.5)*.1
        prior=dict(entry=dict(xy_bounds=[.5,.5,.6,.8],floor_z_bounds=[0,.2]),goals=[dict(id='room',xy_bounds=[3.4,.5,3.6,.8],floor_z_bounds=[0,.2])])
        cfg=dict(map_resolution=.1,human=dict(foot_contact_offset=.025,use_surface_xy=True,resolve_low_surface_band=True),candidates=dict(max_impact_segments=6,pitches=[-.65,.17]),sensor=dict(baseline=.095),utility=dict(task_progress={}))
        analysis=Human(prior,cfg).analyze(lab,layer,np.zeros(3,int),False)
        self.assertEqual(analysis['decisions'][0]['verdict'],'unverified')
        model=CompletionOpportunity(analysis,lab,layer,np.zeros(3,int),.1,cfg,prior)
        sd=SimpleNamespace(fx=100.,fy=100.,cx=80.,cy=45.,R1=np.eye(3),calib=SimpleNamespace(size=(160,90)),cfg=SimpleNamespace(z_min=.6,z_max=6.))
        values=[]
        for height in [1.,2.6]:
            views=[camera_pose(np.array([1.25,.65,height]),0,p,.095) for p in cfg['candidates']['pitches']]
            values.append(model.target_values(model.visibility.point_probabilities(model.points,views,sd)))
        self.assertGreater(values[0].get('room',0),0.)
        self.assertEqual(values[1],{})
        self.assertEqual(analysis['decisions'][0]['verdict'],'unverified')


if __name__=='__main__':unittest.main()
