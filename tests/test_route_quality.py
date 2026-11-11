import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from covor.entry.decision import DecisionGraph
from covor.entry.traverse import TravCfg
from covor.explore.human import relaxed_certificate


class RouteQualityControls(unittest.TestCase):
    def test_penalty_changes_route_without_changing_reachable_set(self):
        g=DecisionGraph.__new__(DecisionGraph);g.xyz=np.c_[np.arange(4),np.zeros((4,2),int)]
        g.confirmed=np.ones((4,1,1),bool);g.cfg=SimpleNamespace(max_unknown_length_m=.5)
        g.res=.1;g.node_factor=np.ones(4);g.node_height=np.zeros(4)
        g.surface_geometry=True;g.body=SimpleNamespace(c_rise=0.)
        network={0:[(1,1.),(2,1.2)],1:[(0,1.),(3,1.)],2:[(0,1.2),(3,1.2)],3:[(1,1.),(2,1.2)]}
        g.edges=lambda u,*args:((v,d,()) for v,d in network[u])
        a=g.search([0],True);b=g.search([0],True,node_penalty=np.array([0,10,0,0.]))
        self.assertEqual(set(a['best']),set(b['best']))
        self.assertEqual(a['parent'][a['best'][3]][0],1)
        self.assertEqual(b['parent'][b['best'][3]][0],2)
        self.assertEqual(a,g.search([0],True,node_penalty=None))


class CompleteGoalCertificateControls(unittest.TestCase):
    def scene(self):
        shape=(32,32,36);lab=np.zeros(shape,np.uint8)
        lab[10:23,10:23,0]=2;lab[10:23,10:23,28]=2
        lab[10,10:23,:29]=2;lab[22,10:23,:29]=2
        lab[10:23,10,:29]=2;lab[10:23,22,:29]=2
        layer=dict(height=np.full(shape,np.nan),support=np.zeros(shape,bool))
        layer['support'][3,3,0]=True;layer['height'][3,3,0]=0.
        entry=dict(xy_bounds=[.3,.3,.4,.4],floor_z_bounds=[-.01,.2])
        goal=dict(xy_bounds=[1.3,1.3,1.9,1.9],floor_z_bounds=[0.,.25])
        return lab,layer,entry,goal

    def certificate(self,lab,layer,entry,goal,complete=True):
        return relaxed_certificate(lab,layer,np.zeros(3,int),.1,entry,goal,TravCfg(res=.1),complete_goal=complete)

    def test_observed_enclosing_cut_does_not_need_interior_support_observation(self):
        lab,layer,e,g=self.scene()
        self.assertIsNone(self.certificate(lab,layer,e,g,False))
        proof=self.certificate(lab,layer,e,g)
        self.assertIsNotNone(proof);self.assertEqual(proof['observed_goal_support_nodes'],0)

    def test_unknown_doorway_keeps_goal_unverified(self):
        lab,layer,e,g=self.scene();lab[10,12:21,1:28]=0
        self.assertIsNone(self.certificate(lab,layer,e,g))

    def test_unknown_roof_does_not_form_closed_enclosure(self):
        lab,layer,e,g=self.scene();lab[10:23,10:23,28]=0
        self.assertIsNone(self.certificate(lab,layer,e,g))

    def test_missing_support_in_open_room_is_not_a_blocked_certificate(self):
        lab,layer,e,g=self.scene();lab[:]=0
        self.assertIsNone(self.certificate(lab,layer,e,g))

    def test_partly_out_of_bounds_goal_cannot_be_called_blocked(self):
        lab,layer,e,g=self.scene();g['xy_bounds']=[1.3,1.3,4.,1.9]
        self.assertIsNone(self.certificate(lab,layer,e,g))

    def test_enclosed_observed_patch_is_not_the_entire_target_region(self):
        lab,layer,e,g=self.scene()
        layer['support'][15,15,0]=True;layer['height'][15,15,0]=0.
        g['xy_bounds']=[.3,1.3,1.9,1.9]
        self.assertIsNotNone(self.certificate(lab,layer,e,g,False))
        self.assertIsNone(self.certificate(lab,layer,e,g))

    def test_human_adapter_enables_full_target_cut_without_changing_default(self):
        from covor.explore.human import Human
        lab,layer,e,g=self.scene();lab[:9,:,:]=1
        layer['votes']=layer['support'].astype(np.uint16)*2
        layer['evidence']=layer['support'].astype(float)*2
        cfg=dict(map_resolution=.1,human=dict(foot_contact_offset=.025),candidates=dict(max_impact_segments=6))
        prior=dict(entry=e,goals=[dict(g,id='sealed')])
        old=Human(prior,cfg).analyze(lab,layer,np.zeros(3,int),False)
        self.assertEqual(old['decisions'][0]['verdict'],'unverified')
        cfg['human']['full_goal_cut_certificate']=True
        new=Human(prior,cfg).analyze(lab,layer,np.zeros(3,int),False)
        self.assertEqual(new['decisions'][0]['verdict'],'blocked')
        np.testing.assert_array_equal(old['graph'].confirmed,new['graph'].confirmed)


if __name__=='__main__':unittest.main()
