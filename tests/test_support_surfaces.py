"""Support evidence must survive air carving but not invented/contradicted floors."""
from types import SimpleNamespace
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from covor.entry.surfaces import SupportSurfaces,SurfaceCfg
from covor.entry.decision import DecisionGraph


def down_camera():
    sd=SimpleNamespace(fx=60.,fy=60.,cx=40.,cy=40.,R1=np.eye(3))
    T=np.eye(4);T[:3,:3]=np.diag([1.,-1.,-1.]);T[:3,3]=[1.,1.,1.]
    return sd,T


def test_repeated_horizontal_surface_survives_free_volume_and_roundtrip():
    sd,T=down_camera();z=np.ones((80,80),np.float32)*.95
    s=SupportSurfaces();valid=np.ones(z.shape,bool);sigma=np.ones_like(z)*.01
    s.integrate(sd,z,sigma,valid,T)
    assert not s.dense(np.zeros(3,int),(30,30,30))['support'].any()
    s.integrate(sd,z,sigma,valid,T)
    layer=s.dense(np.zeros(3,int),(30,30,30))
    assert layer['support'].sum()>40
    assert np.allclose(layer['height'][layer['support']],.05)
    lab=np.ones((30,30,30),np.uint8)
    assert not DecisionGraph(lab,np.zeros(3,int)).confirmed.any()
    assert DecisionGraph(lab,np.zeros(3,int),support_layer=layer).confirmed.any()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'surface.npz'
        s.save(path);q=SupportSurfaces.load(path)
    assert np.array_equal(layer['support'],q.dense(np.zeros(3,int),lab.shape)['support'])


def test_observed_ray_below_floor_contradicts_support():
    sd,T=down_camera();z=np.ones((80,80),np.float32)*.95
    s=SupportSurfaces();valid=np.ones(z.shape,bool);sigma=np.ones_like(z)*.01
    for _ in range(3):s.integrate(sd,z,sigma,valid,T)
    # A view of a lower surface provides measured contradiction to the old one.
    for _ in range(4):s.integrate(sd,z+.3,sigma,valid,T)
    layer=s.dense(np.array([0,0,-5]),(30,30,35))
    assert not layer['support'][:,:,5].any()


def test_wall_depth_patch_does_not_become_a_floor():
    sd,T=down_camera();T[:3,:3]=np.eye(3)
    z=np.ones((80,80),np.float32)
    s=SupportSurfaces()
    for _ in range(3):s.integrate(sd,z,z*.01,np.ones(z.shape,bool),T)
    assert not s.cells


def test_independent_support_does_not_remove_body_wall_or_fill_unknown_floor():
    a=np.ones((35,15,30),np.uint8);support=np.zeros(a.shape,bool);support[:,:,0]=True
    support[17:20,:,0]=False;a[17:20,:,0]=0
    layer=dict(support=support,height=np.full(a.shape,.05))
    def r(x):return dict(id=str(x),xy_bounds=[x-.04,.71,x+.04,.79],floor_z_bounds=[0,.2])
    d=DecisionGraph(a,np.zeros(3,int),support_layer=layer).decide(r(.55),[r(2.85)])['decisions'][0]
    assert d['verdict']=='unverified'
    a[17:20,:,:]=2
    d=DecisionGraph(a,np.zeros(3,int),support_layer=layer).decide(r(.55),[r(2.85)])['decisions'][0]
    assert not d['path'] and not d['possible_path']


def test_measured_feet_and_head_margin_choose_lower_posture():
    a=np.ones((30,15,30),np.uint8);a[:,:,0]=2;a[:,:,20]=2
    support=np.zeros(a.shape,bool);support[:,:,0]=True
    layer=dict(support=support,height=np.full(a.shape,.095),votes=support.astype(int)*2,evidence=support.astype(float)*2)
    old=DecisionGraph(a,np.zeros(3,int),support_layer=layer)
    new=DecisionGraph(a,np.zeros(3,int),support_layer=layer,surface_geometry=True)
    assert old.cp[10,7,0]==0 and new.cp[10,7,0]==1
    assert np.isclose(new.foot_z[10,7,0],.1)


def test_diagonal_voxel_corner_is_in_the_body_disc():
    a=np.ones((30,15,30),np.uint8);a[:,:,0]=2;a[12,9,3:25]=2
    support=np.zeros(a.shape,bool);support[:,:,0]=True
    layer=dict(support=support,height=np.full(a.shape,.05),votes=support.astype(int)*2,evidence=support.astype(float)*2)
    old=DecisionGraph(a,np.zeros(3,int),support_layer=layer)
    new=DecisionGraph(a,np.zeros(3,int),support_layer=layer,surface_geometry=True)
    assert old.cp[10,7,0]>=0 and new.cp[10,7,0]<0


def test_relative_heights_use_overlap_and_reject_disconnected_maps():
    from covor.entry.registration import relative_heights
    surfaces={r:SupportSurfaces() for r in ('a','b','c')}
    for r,z in zip(surfaces,(.05,.55,1.45)):
        for x in range(10):
            for y in range(10):surfaces[r].cells[(x,y,int(z/.1))]=np.array([2.,2*z,2*z*z,2.,0.])
    offsets,_,status=relative_heights(surfaces)
    assert status['accepted'] and np.allclose(list(offsets.values()),[0.,-.5,-1.4])
    surfaces['c'].cells={}
    offsets,_,status=relative_heights(surfaces)
    assert not status['accepted'] and not any(offsets.values())


def test_one_view_score_does_not_promise_completion_across_unseen_serial_section():
    from covor.entry.assignment import choose
    a=np.ones((80,15,30),np.uint8);a[:,:,2]=2
    sd=SimpleNamespace(calib=SimpleNamespace(size=(100,100),baseline=.05),R1=np.eye(3),fx=35.,fy=35.,cx=50.,cy=50.)
    T=np.eye(4);T[:3,:3]=np.array([[0,0,1],[1,0,0],[0,1,0]]);T[:3,3]=[.55,.65,1.2]
    def cells(sid,_):
        x=15 if sid==1 else 65
        return ((np.array([[x,6,2]]),1),(np.array([[x,6,8]]),1))
    layer=dict(votes=np.ones(a.shape,int),evidence=np.ones(a.shape,float))
    g=SimpleNamespace(lab=a,ijk=np.zeros(3,int),res=.1,support=np.zeros(a.shape,bool),support_layer=layer,region_observation_cells=cells)
    analysis=dict(impact={1:['g'],2:['g']},decisions=[dict(goal='g',uncertain_segments=[1,2])])
    class Forbidden:
        def __array__(self,*args):raise AssertionError('future observation read')
    candidates=[dict(id=0,robot='a',T=T,move_cost=.1,depth=Forbidden(),truth=Forbidden())]
    _,old,reason=choose(g,analysis,candidates,{'a':sd})
    _,new,new_reason=choose(g,analysis,candidates,{'a':sd},single_view=True,surface_cfg=SurfaceCfg())
    assert old[0]['priority']>0 and not reason['fallback']
    assert new[0]['priority']==0 and new_reason['fallback']


def test_cut_reachability_dominance_matches_full_distance_state_search():
    rng=np.random.default_rng(38)
    for _ in range(4):
        a=np.ones((25,13,26),np.uint8);a[:,:,1]=2;a[:,:,24]=2
        missing=rng.random((25,13))<.2;a[:,:,1][missing]=0
        a[12,1:5,2:24]=2
        g=DecisionGraph(a,np.zeros(3,int))
        region=dict(xy_bounds=[.31,.61,.39,.69],floor_z_bounds=[.15,.25])
        starts=g.mission_nodes(region)
        for closed in (None,*[s['id'] for s in g.segments[:3]]):
            full=g.search(starts,closed=closed);fast=g.reachability_search(starts,closed=closed)
            assert set(full['best'])==set(fast['best'])


def test_repair_depth_uses_rectified_matcher_and_rejects_legacy_cache():
    import tempfile,json
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/reobserve'))
    import repair
    from covor.occupancy import DepthCfg
    class Matcher:
        def compute(self,left,right):return np.full(left.shape,80,np.int16)
    def forbidden(*args):raise AssertionError('already rectified pair remapped again')
    sd=SimpleNamespace(cfg=DepthCfg(),fx=100.,fy=100.,cx=4.,cy=4.,R1=np.eye(3),
        calib=SimpleNamespace(baseline=.1),sgbm=Matcher(),depth=forbidden)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);source=root/'source';(source/'input_cache').mkdir(parents=True)
        (source/'design.json').write_text(json.dumps({'conditions':{'smoke':{'beta':.15,'field':'uniform','seed':0}}}))
        image=np.ones((8,8),np.uint8)*50;z=np.ones((8,8),np.float32)*2
        np.savez_compressed(source/'input_cache'/'r_00000.npz',left=image,right=image,z=z,zr=z,sz=z*.01,valid=np.ones(z.shape,bool))
        out=root/'out';out.mkdir();poses={'r':dict(T_truth=np.eye(4)[None],t=np.array([0.]))}
        previous=repair.SOURCE;repair.SOURCE=source
        try:
            actual=repair.depth(out,'smoke','r',0,poses,sd)
            assert np.array_equal(actual[0],z) and actual[2].all()
            cached=repair.depth(out,'smoke','r',0,poses,sd)
            assert all(np.array_equal(a,b) for a,b in zip(actual,cached))
            np.savez_compressed(out/'depth_cache'/'smoke_r_00000.npz',z=z,sz=z*.01,valid=np.ones(z.shape,bool))
            try:repair.depth(out,'smoke','r',0,poses,sd)
            except ValueError:pass
            else:raise AssertionError('unverified legacy cache reused')
        finally:repair.SOURCE=previous


if __name__=='__main__':
    for name,test in list(globals().items()):
        if name.startswith('test_'):test();print('PASS',name,flush=True)
