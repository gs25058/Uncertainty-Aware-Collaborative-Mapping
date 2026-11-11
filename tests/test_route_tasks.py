"""Path-local repair requests and serial-task utility controls."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from covor.explore.route_tasks import build_route_tasks, serial_values
from covor.explore.entry_objective import pooled_target_values, joint_marginal


class RouteTaskControls(unittest.TestCase):
    def prediction(self,foot,body,key='door'):
        return {key:dict(kind='anchored',goals={'g':.5},foot=[foot],body=[body],foot_before=0.,body_before=0.)}

    def test_unqueried_serial_section_does_not_complete_goal(self):
        partial=serial_values(self.prediction(1,1),{'g':2},0)['g']
        self.assertGreater(partial,0);self.assertLess(partial,1)
        all_sections={**self.prediction(1,1),**self.prediction(1,1,'landing')}
        self.assertAlmostEqual(serial_values(all_sections,{'g':2},0)['g'],1.)

    def test_last_repair_is_more_valuable_than_first_on_long_route(self):
        p=self.prediction(1,1)
        self.assertGreater(serial_values(p,{'g':1},0)['g'],serial_values(p,{'g':5},0)['g'])
        a=serial_values(p,{'g':2},0)['g']
        b=serial_values({**p,**self.prediction(1,1,'landing')},{'g':2},0)['g']
        self.assertGreater(b-a,a)

    def test_complementary_views_union_before_serial_minimum(self):
        cfg={'utility':{'task_progress':{'serial_repair_potential':True,'anchored_weight':1.,'partial_evidence_weight':0.}}}
        def item(f,b):return dict(task_query_predictions=self.prediction(f,b),serial_sections={'g':2})
        a=item(1,0);b=item(0,1)
        self.assertEqual(pooled_target_values([a],cfg)['g'],0.)
        self.assertGreater(joint_marginal(b,[a],cfg),0.)
        self.assertEqual(joint_marginal(a,[a],cfg),0.)
        self.assertEqual(joint_marginal(a,[a,b],cfg),0.)

    def test_queries_exclude_off_path_nodes_in_same_uncertainty_segment(self):
        class Graph:
            body=SimpleNamespace(c_rise=0.)
            node_known=np.array([True,False,False,False])
            node_xy=np.array([[0.,0.],[1.,0.],[2.,0.],[1.,.4]])
            node_height=np.full(4,.025);foot_contact_offset=.025
            def mission_nodes(self,region):return [2]
            def edges(self,u,confirmed):
                for v in {0:[1],1:[0,2,3],2:[1],3:[1]}[u]:yield v,1.,(1,)
            def _raw_neighbors(self,u):
                return [(v,length,[]) for v,length,_ in self.edges(u,False)]
        g=Graph();queries=[]
        c=SimpleNamespace(opt={'max_anchored_queries':8},prior={'goals':[{'id':'g'}]},decisions={'g':{'verdict':'unverified'}})
        c.query=lambda key,feet,goals,kind:queries.append((key,feet,goals))
        analysis={'graph':g,'confirmed_search':{'best':{0:None}}}
        build_route_tasks(c,analysis)
        self.assertEqual(c.soft_route_stats['routes']['g']['unconfirmed_nodes'],2)
        self.assertTrue(queries)
        self.assertTrue(all(np.all(q[1][:,1]==0) for q in queries))
        before=[(q[0],q[1].tolist(),q[2]) for q in queries];queries.clear()
        build_route_tasks(c,analysis)
        self.assertEqual(before,[(q[0],q[1].tolist(),q[2]) for q in queries])
        self.assertEqual(c.decisions['g']['verdict'],'unverified')

    def test_no_confirmed_entrance_component_yields_no_invented_route(self):
        g=SimpleNamespace(mission_nodes=lambda goal:[1])
        c=SimpleNamespace(opt={},prior={'goals':[{'id':'g'}]},decisions={'g':{'verdict':'unverified'}},query=lambda *args:self.fail('invented query'))
        build_route_tasks(c,{'graph':g,'confirmed_search':{'best':{}}})
        self.assertEqual(c.serial_sections,{})
        self.assertEqual(c.soft_route_stats['goals_with_value_only_path'],0)


if __name__=='__main__':unittest.main()
