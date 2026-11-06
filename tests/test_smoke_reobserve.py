"""Smoke physics approximations and GT-blind selectors, checked by value."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from covor.smoke import (SmokeCfg, attenuate, complete_depth, density,
                         line_mean_density, rectified_camera_poses)
from covor.reobserve import rank_candidates, goal_verdicts


def test_zero_smoke_is_bit_identical_even_when_depth_missing():
    image=np.arange(120,dtype=np.uint8).reshape(10,12)
    result,t,_=attenuate(image,np.full(image.shape,np.nan),
                         np.ones((*image.shape,3)),np.eye(4),4.,SmokeCfg())
    assert np.array_equal(result,image)
    assert np.array_equal(t,np.ones(image.shape))


def test_uniform_transmission_uses_euclidean_range_and_linear_blend():
    image=np.full((2,2),50,np.uint8)
    dirs=np.tile([0.,0.,1.],(2,2,1)); dirs[0,0]=[1.,0.,1.]
    z=np.full((2,2),2.)
    fog,t,_=attenuate(image,z,dirs,np.eye(4),0,SmokeCfg(beta=.4))
    expected=np.exp(-.4*2*np.linalg.norm(dirs,axis=-1))
    np.testing.assert_allclose(t,expected,rtol=1e-6)
    assert np.array_equal(fog,np.rint(expected*50+(1-expected)*204).astype(np.uint8))
    _,strong,_=attenuate(image,z,dirs,np.eye(4),0,SmokeCfg(beta=.8))
    assert (strong<t).all()


def test_completion_cannot_modify_original_or_claim_new_measurements():
    z=np.array([[2.,np.nan],[2.,np.nan]],np.float32)
    old=z.copy()
    filled,fraction=complete_depth(z)
    assert np.array_equal(z,old,equal_nan=True)
    assert np.array_equal(filled,np.full((2,2),2.)) and fraction==.5
    filled,fraction=complete_depth(np.full((2,2),np.nan))
    assert (filled==5).all() and fraction==1.


def test_exact_line_integral_matches_numerical_quadrature():
    origin=np.array([.2,-.1,1.])
    end=np.array([[2.,.5,1.5],[.2,-.1,1.],[1.,-4.,.5]])
    a=(np.arange(20000)+.5)/20000
    p=origin+(end[:,None,:]-origin)*a[None,:,None]
    numeric=density(p,3.7,4).mean(axis=1)
    np.testing.assert_allclose(line_mean_density(origin,end,3.7,4),numeric,atol=1e-8)


def test_shared_field_is_temporally_continuous_and_stereo_world_consistent():
    sd=SimpleNamespace(R1=np.eye(3),calib=SimpleNamespace(baseline=.05))
    left,right=rectified_camera_poses(sd,np.eye(4))
    np.testing.assert_array_equal(right[:3,3]-left[:3,3],[.05,0,0])
    p=np.array([[1.,2.,3.]])
    assert abs(float(density(p,5.00001,0)[0]-density(p,5.,0)[0]))<1e-5
    assert np.array_equal(density(p,5.,0),density(p,5.,0))


def test_candidate_ranking_has_no_access_to_future_images_or_truth():
    class Forbidden:
        def __array__(self,*args):
            raise AssertionError("future observation accessed")
    labels=np.zeros((30,30,25),np.uint8)
    labels[:,:,0]=2
    sd=SimpleNamespace(calib=SimpleNamespace(size=(40,30)),R1=np.eye(3),
                       cx=20.,cy=15.,fx=30.,fy=30.)
    T=np.eye(4); T[:3,3]=[1.5,1.5,.5]
    candidates=[dict(id=i,robot="r",timestamp=float(i),T=T.copy(),image=Forbidden(),gt=Forbidden())
                for i in range(3)]
    args=(labels,np.array([0,0,0]),.1,candidates,{"r":sd},
          {"r":T[:3,3]},(1.5,1.5),[dict(id="goal",x=1.5,y=2.)])
    picks,scores,importance=rank_candidates(*args)
    assert picks==dict(nearest=0,unknown=0,route=0)
    again=rank_candidates(*args)
    assert picks==again[0] and scores==again[1]
    assert np.array_equal(importance,again[2])


def test_gap_assisted_reach_is_still_unverified_without_supported_route():
    tm=dict(reach=np.array([[True,True]]))
    strict=dict(reach=np.array([[True,False]]))
    goals=[dict(id="supported",x=.05,y=.05),dict(id="gap",x=.05,y=.15)]
    rows=goal_verdicts(tm,strict,np.zeros((1,2,5)),np.zeros(3,int),.1,goals)
    assert rows[0]["verdict"]=="confirmed"
    assert rows[1]["gap_permitted_reach"] and rows[1]["verdict"]=="unverified"


def test_right_depth_uses_its_own_stereo_camera_coordinates():
    from covor.occupancy import StereoCalib,StereoDepth,DepthCfg
    from covor.synth.render import _depth_rectified
    from covor.smoke import clean_right_depth
    K=np.array([[100.,0,100.],[0,100.,40.],[0,0,1.]])
    calib=StereoCalib(K,np.zeros(4),K,np.zeros(4),np.eye(3),np.array([-.1,0,0]),(200,80))
    sd=StereoDepth(calib,DepthCfg(num_disp=32))
    rng=np.random.default_rng(123)
    left=rng.integers(0,256,(80,200),dtype=np.uint8)
    right=np.zeros_like(left);right[:,:-5]=left[:,5:]
    z,_,v=_depth_rectified(sd,left,right)
    zr=clean_right_depth(sd,left,right)
    assert v.sum()>8000 and np.isfinite(zr).sum()>8000
    assert abs(float(np.nanmedian(z))-2.)<.02
    assert abs(float(np.nanmedian(zr))-2.)<.02


def test_full_ot_clone_preserves_exact_evidence_before_reobservation():
    import tempfile
    import octomap
    tree=octomap.OcTree(.1)
    for p,l in (([.05,.05,.05],.321),([.15,.05,.05],-.432),([.25,.05,.05],1.234)):
        tree.updateNode(p,l,True)
    def values(t):
        return {tuple(n.getCoordinate()):n.getValue() for n in t.begin_leafs()}
    with tempfile.TemporaryDirectory() as d:
        path=str(Path(d)/"tree.ot").encode()
        tree.write(path)
        clone=octomap.OcTree.read(path)
        assert values(tree)==values(clone)


def test_nearby_support_does_not_confirm_an_unobserved_exact_entry():
    from covor.entry.traverse import entry_node
    posture=np.full((1,5,4),-1,np.int8)
    posture[0,2,1]=0
    assert entry_node(posture,(.05,.05),np.zeros(3,int),.1,snap_m=1.5)[1]==.2
    assert entry_node(posture,(.05,.05),np.zeros(3,int),.1,snap_m=0.) is None


if __name__=="__main__":
    tests=[v for k,v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS",test.__name__)
    print(len(tests),"passed")
