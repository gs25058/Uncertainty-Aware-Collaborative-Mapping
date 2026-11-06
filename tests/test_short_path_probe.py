"""Goal-directed search preserves route decisions; independent mesh control math."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'scripts/reobserve')]
import numpy as np
from covor.entry.decision import DecisionGraph
from short_path_probe import clip_triangle_box,route_query


def test_triangle_box_certificate_rejects_bounding_box_false_overlap():
    low=np.array([-.1,-.1,-.1]);high=-low
    triangle=np.array([[2.,0,0],[0,2.,0],[2.,2.,0]])
    assert not clip_triangle_box(triangle,low,high)
    assert clip_triangle_box(np.array([[-2.,-2.,0],[2.,-2.,0],[0,2.,0]]),low,high)


def test_target_search_matches_complete_search_with_unknown_threshold_and_wall():
    def patch(x):return dict(id='goal',xy_bounds=[x-.04,.51,x+.04,.59],floor_z_bounds=[.15,.25])
    for condition in ('open','unknown','wall'):
        lab=np.ones((40,11,30),np.uint8);lab[:,:,1]=2;lab[:,:,27]=2
        lab[:,0,:]=2;lab[:,-1,:]=2
        if condition=='unknown':lab[7:9,:,1]=0
        if condition=='wall':lab[8,:,:]=2
        g=DecisionGraph(lab,np.zeros(3,int));entry=patch(.55);goal=patch(1.25)
        for confirmed in (True,False):
            starts=g.mission_nodes(entry,confirmed);targets=g.mission_nodes(goal,confirmed)
            full=g.search(starts,confirmed);short=g.search(starts,confirmed,target_nodes=targets)
            a=g.path(full,targets);b=g.path(short,targets)
            assert a==b,(condition,confirmed)
            assert len(short['best'])<=len(full['best'])
        actual=route_query(g,entry,goal);original=g.decide(entry,[goal])['decisions'][0]
        assert actual['verdict']==original['verdict']
        assert actual['path']==original['path']


if __name__=='__main__':
    for name,test in list(globals().items()):
        if name.startswith('test_'):test();print('PASS',name,flush=True)
