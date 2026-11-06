"""Value tests for bounded hypotheses, segment cuts and proposed-path checks."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gtsam  # before open3d
from covor.entry.decision import DecisionGraph
from covor.entry.assignment import visible_fraction,choose,unknown_visibility
from covor.entry.path_eval import ReferenceValidator,motion_path


def room(nx=50,ny=13,nz=32):
    a=np.zeros((nx,ny,nz),np.uint8);a[:,:,2]=2;a[:,:,28]=2;a[:,:,3:28]=1
    a[0,:,2:29]=a[-1,:,2:29]=2;a[:,0,2:29]=a[:,-1,2:29]=2
    return a


def region(x,y,ident="r"):
    return dict(id=ident,xy_bounds=[x-.04,y-.04,x+.04,y+.04],floor_z_bounds=[.25,.35])


def test_common_unknown_threshold_changes_three_goals():
    a=room();a[18:21,1:-1,2]=0
    g=DecisionGraph(a,np.zeros(3,int));entry=region(.55,.65)
    goals=[region(x,.65,str(i)) for i,x in enumerate((2.55,3.55,4.25))]
    result=g.decide(entry,goals,True)
    important=[v for v in result["impact"].values() if v]
    assert important==[["0","1","2"]],important
    assert all(d["verdict"]=="unverified" and d["possible_path"] for d in result["decisions"])


def test_alternate_path_removes_cut_importance_and_no_goal_double_count():
    a=room(ny=25);a[18:21,1:12,2]=0
    g=DecisionGraph(a,np.zeros(3,int))
    r=g.decide(region(.55,.65),[region(4.25,.65,"g")],True)
    assert r["decisions"][0]["verdict"]=="passable"
    assert not any(r["impact"].values())


def test_serial_unknown_thresholds_do_not_confirm_a_final_goal():
    a=room();a[12:14,1:-1,2]=0;a[24:26,1:-1,2]=0
    g=DecisionGraph(a,np.zeros(3,int));goal=region(4.25,.65,"g")
    r=g.decide(region(.55,.65),[goal],True)
    assert len([v for v in r["impact"].values() if v])==2
    assert r["decisions"][0]["verdict"]=="unverified"
    a[12:14,1:-1,2]=2
    r=DecisionGraph(a,np.zeros(3,int)).decide(region(.55,.65),[goal])
    assert r["decisions"][0]["verdict"]=="unverified"


def test_observed_wall_and_observed_support_void_never_open():
    for wall in (True,False):
        a=room()
        if wall:a[20,:,2:29]=2
        else:a[18:23,1:-1,:5]=1
        g=DecisionGraph(a,np.zeros(3,int))
        result=g.decide(region(.55,.65),[region(4.25,.65,"g")])
        assert not result["decisions"][0]["path"]
        assert not result["decisions"][0]["possible_path"]
        if wall:assert result["decisions"][0]["verdict"]=="blocked"


def camera():
    sd=SimpleNamespace(calib=SimpleNamespace(size=(100,100)),R1=np.eye(3),fx=35.,fy=35.,cx=50.,cy=50.)
    T=np.eye(4);T[:3,:3]=np.array([[0,0,1],[1,0,0],[0,1,0]])
    T[:3,3]=[.55,.65,1.2]
    return sd,T


def test_important_threshold_behind_a_wall_is_occluded():
    a=room();sd,T=camera();cells=np.array([[24,6,2],[25,6,2]])
    clear=visible_fraction(cells,a,np.zeros(3,int),.1,T,sd,support_surface=True)
    a[15,:,2:29]=2
    blocked=visible_fraction(cells,a,np.zeros(3,int),.1,T,sd,support_surface=True)
    assert clear["fraction"]>0 and blocked["fraction"]==0


def test_selection_never_reads_future_depth_or_truth_and_is_deterministic():
    class Forbidden:
        def __array__(self,*args):raise AssertionError("future data accessed")
    a=room();a[18:21,1:-1,2]=0
    g=DecisionGraph(a,np.zeros(3,int));r=g.decide(region(.55,.65),[region(4.25,.65,"g")],True)
    sd,T=camera()
    candidates=[dict(id=i,robot="drone",T=T,move_cost=.1,depth=Forbidden(),truth=Forbidden()) for i in (2,1)]
    first=choose(g,r,candidates,{"drone":sd});second=choose(g,r,candidates,{"drone":sd})
    assert first==second
    assert first[0]==dict(nearest=1,unknown=1,decision=1)
    assert all(s["priority"]<=1 for s in first[1])


def test_unknown_visibility_counts_cells_and_stops_at_observed_wall():
    a=room();a[24:28,3:10,5:19]=0
    sd,T=camera()
    before=unknown_visibility(a,np.zeros(3,int),.1,T,sd)
    a[15,:,2:29]=2
    after=unknown_visibility(a,np.zeros(3,int),.1,T,sd)
    assert before>0 and after<before,(before,after)


def test_optional_legacy_predecessors_preserve_default_arrays():
    from covor.entry import traverse as TV
    a=room();a[18:21,1:-1,2]=0
    plain=TV.traverse(a,TV.TravCfg(),(.55,.65),np.zeros(3,int),snap_m=0)
    saved=TV.traverse(a,TV.TravCfg(),(.55,.65),np.zeros(3,int),snap_m=0,return_paths=True)
    for k,v in plain.items():
        if isinstance(v,np.ndarray):assert np.array_equal(v,saved[k])
        else:assert v==saved[k]
    assert saved["path_parent"] and saved["path_best"]


def mesh_scene(boxes):
    import open3d as o3d
    import trimesh
    scene=o3d.t.geometry.RaycastingScene()
    for low,high in boxes:
        mesh=trimesh.creation.box(extents=np.array(high)-low)
        mesh.apply_translation((np.array(high)+low)/2)
        tm=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tm))
    return scene


def test_empty_destination_but_single_closed_door_is_not_reachable():
    a=room();a[20,:,2:29]=2
    d=DecisionGraph(a,np.zeros(3,int)).decide(region(.55,.65),[region(4.25,.65,"g")])["decisions"][0]
    assert a[42,6,10]==1 and d["verdict"]=="blocked"


def test_goal_has_detour_but_proposed_path_through_closed_door_is_invalid():
    a=room(ny=30)
    # Partial divider leaves a genuine detour around y=2.2.
    a[20,1:15,2:29]=2
    g=DecisionGraph(a,np.zeros(3,int));entry=region(.55,.65);goal=region(4.25,.65,"g")
    assert g.decide(entry,[goal])["decisions"][0]["verdict"]=="passable"
    scene=mesh_scene([([0,0,.2],[5,3,.3]),([2.,0,.3],[2.1,1.5,2.9])])
    validator=ReferenceValidator(a,np.zeros(3,int),.1,scene)
    wrong=[dict(xyz=[x,.65,.3],posture=0,width=0,observed_support=True) for x in (.55,1.55,2.55,4.25)]
    result=validator.validate(wrong,entry,goal)
    assert result["status"]=="invalid" and result["reason"]=="mesh_body_collision",result


def test_missing_floor_observation_stays_unverified_even_with_visible_air():
    a=room();a[18:21,1:-1,2]=0
    d=DecisionGraph(a,np.zeros(3,int)).decide(region(.55,.65),[region(4.25,.65,"g")])["decisions"][0]
    assert d["verdict"]=="unverified" and not d["path"]


def test_one_actual_frame_can_cancel_a_false_pass_or_confirm_a_block():
    from covor.occupancy import OccupancyBuilder,OccCfg
    initial=room(nx=30,ny=9)
    entry=region(.55,.45);goal=region(2.55,.45,"g")
    assert DecisionGraph(initial,np.zeros(3,int)).decide(entry,[goal])["decisions"][0]["verdict"]=="passable"
    # One observation, many identical endpoint rays accumulate within that
    # frame. This is the original mapper's additivity, not label replacement.
    b=OccupancyBuilder(OccCfg(weighted=False,dyn_radius=0))
    endpoints=np.array([[1.55,y,z] for y in np.arange(.15,.8,.1) for z in np.arange(.35,2.9,.1)])
    endpoints=np.repeat(endpoints,3,axis=0)
    T=np.eye(4);T[:3,3]=[.55,.45,1.2]
    b.integrate_frame(T,0,endpoints-T[:3,3],np.zeros(len(endpoints)))
    occ,_=b.classify_points()
    after=initial.copy();ind=np.floor(occ/.1+1e-5).astype(int)
    ind=ind[((ind>=0)&(ind<np.array(after.shape))).all(1)];after[tuple(ind.T)]=2
    d=DecisionGraph(after,np.zeros(3,int)).decide(entry,[goal])["decisions"][0]
    assert d["verdict"]!="passable",d


def test_body_width_and_posture_are_checked_on_reference_mesh():
    a=room()
    scene=mesh_scene([([0,0,.2],[5,1.3,.3]),
                      ([2.,0,.3],[2.1,.4,2.9]),([2.,.9,.3],[2.1,1.3,2.9]),
                      ([3.,0,1.75],[3.1,1.3,1.85])])
    validator=ReferenceValidator(a,np.zeros(3,int),.1,scene)
    assert validator.body_clear([2.05,.65,.3],0,0)[0]=="invalid"
    assert validator.body_clear([2.05,.65,.3],0,1)[0]=="valid"
    assert validator.body_clear([3.05,.65,.3],0,0)[0]=="invalid"
    assert validator.body_clear([3.05,.65,.3],1,0)[0]=="valid"


def test_wrong_floor_and_unknown_reference_are_not_valid_paths():
    a=room();scene=mesh_scene([([0,0,.2],[5,1.3,.3])])
    validator=ReferenceValidator(a,np.zeros(3,int),.1,scene)
    entry=region(.55,.65);goal=region(2.55,.65)
    path=[dict(xyz=[x,.65,.3],posture=0,width=0,observed_support=True) for x in (.55,1.55,2.55)]
    bad=[dict(p) for p in path];bad[-1]=dict(bad[-1],xyz=[2.55,.65,1.3])
    assert validator.validate(bad,entry,goal)["status"]=="invalid"
    a[12:18,2:11,6:20]=0
    assert ReferenceValidator(a,np.zeros(3,int),.1,scene).validate(path,entry,goal)["status"]=="unverifiable"


def test_metric_zero_checked_denominator_is_na_and_coverage_is_reported():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/reobserve"))
    from evaluation import metrics
    rows=[dict(case_id=str(i),reference_state="reachable",path_status="not_proposed",verdict="unverified") for i in range(3)]
    m=metrics(rows)
    assert m["wrong_path_rate"]=="N/A" and m["valid_path_confirm_rate"]==0
    assert m["n_reference_reachable"]==3 and m["n_decided"]==0 and m["n_unverified"]==3


if __name__=="__main__":
    tests=[v for k,v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test();print("PASS",test.__name__,flush=True)
    print(len(tests),"passed")
