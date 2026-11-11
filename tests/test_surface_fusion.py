import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from covor.explore.surface_fusion import SurfaceFusion,FusionCfg


class SurfaceFusionControls(unittest.TestCase):
    def camera(self):return SimpleNamespace(fx=100.,fy=100.,cx=40.,cy=30.,R1=np.eye(3),cfg=SimpleNamespace(z_max=6.))

    def test_invalid_depth_creates_no_evidence_or_free_cells(self):
        fusion=SurfaceFusion();z=np.full((60,80),np.nan,np.float32)
        fusion.integrate(self.camera(),z,z,np.zeros_like(z,bool),np.eye(4))
        self.assertEqual(fusion.vbg.hashmap().size(),0)
        lab=np.zeros((8,8,8),np.uint8);layer={'support':np.zeros(lab.shape,bool)}
        new,_,_=fusion.reconstruct(lab,layer,np.zeros(3,int),.1)
        np.testing.assert_array_equal(new,lab)

    def test_one_frame_cannot_meet_repeated_evidence_threshold(self):
        fusion=SurfaceFusion();z=np.ones((60,80),np.float32)
        fusion.integrate(self.camera(),z,z*.003,np.ones_like(z,bool),np.eye(4))
        keys,_=fusion.arrays();self.assertEqual(len(keys),0)

    def test_sparse_valid_pixel_is_not_lost_by_native_touch_subsampling(self):
        f=SurfaceFusion();z=np.zeros((60,80),np.float32);z[1,1]=1.
        sigma=np.full_like(z,.003);valid=z>0
        f.integrate(self.camera(),z,sigma,valid,np.eye(4))
        self.assertGreater(f.vbg.hashmap().size(),0)

    def test_same_time_error_remains_correlated_but_separated_times_add_information(self):
        sd=self.camera();z=np.ones((60,80),np.float32);sigma=z*.003;extra=z*.00000875
        values=[]
        for times in [[0.,0.,0.,0.],[0.,1.5,3.,4.5]]:
            f=SurfaceFusion(FusionCfg(finite_time_correlation=True))
            for t in times:f.integrate(sd,z,sigma,np.ones_like(z,bool),np.eye(4),independent_variance=extra,time=t)
            keys,v=f.arrays();i=np.flatnonzero((keys==[0,0,40]).all(1))[0]
            values.append(float(v['time_variance'][i]/v['weight'][i]**2))
        self.assertAlmostEqual(values[0],.003**2+.00000875,places=9)
        self.assertAlmostEqual(values[0]/values[1],4.,places=4)

    def test_sparse_error_cannot_be_assumed_independent_in_actual_model(self):
        f=SurfaceFusion();z=np.ones((60,80),np.float32)
        with self.assertRaises(ValueError):f.integrate(self.camera(),z,z*.003,z>0,np.eye(4),independent_variance=z*.00001)

    def test_observed_quality_uses_inlier_and_blunder_terms_separately(self):
        import json
        from covor.explore.observed_depth_quality import uncertainty_components
        cfg=json.load(open('configs/entry_current_20261109/pure/precision_v4.json'))
        z=np.ones((60,80),np.float32);sigma,extra=uncertainty_components(self.camera(),z,np.ones_like(z,bool),np.eye(4),cfg)
        self.assertAlmostEqual(float(sigma[30,40]),.003,places=6)
        self.assertAlmostEqual(float(extra[30,40]),.0005*(.05**2+.05*.2+.2**2)/3,places=10)

    def test_quality_weighted_update_matches_explicit_scalar_fusion(self):
        fusion=SurfaceFusion();z=np.ones((60,80),np.float32);valid=np.ones_like(z,bool)
        fusion.integrate(self.camera(),z,z*.002,valid,np.eye(4))
        fusion.integrate(self.camera(),z+.04,z*.02,valid,np.eye(4))
        xyz,flat=fusion.vbg.voxel_coordinates_and_flattened_indices();xyz=xyz.numpy();ids=flat.numpy().reshape(-1)
        selected=np.flatnonzero(np.linalg.norm(xyz-[0,0,1.],axis=1)<1e-6);self.assertEqual(len(selected),1)
        i=ids[selected[0]];sdf=float(fusion.vbg.attribute('tsdf').numpy().reshape(-1)[i])
        weight=float(fusion.vbg.attribute('weight').numpy().reshape(-1)[i])
        a=.006**2/(.006**2+.002**2);b=.006**2/(.006**2+.02**2)
        self.assertAlmostEqual(weight,a+b,places=5)
        self.assertAlmostEqual(sdf*.1,b*.04/(a+b),places=5)

    def test_repeated_wall_is_preserved_and_space_behind_it_is_not_freed(self):
        f=SurfaceFusion();sd=self.camera();z=np.ones((60,80),np.float32);T=np.eye(4)
        T[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];T[:3,3]=[0,1,1]
        for _ in range(12):f.integrate(sd,z,z*.003,np.ones_like(z,bool),T)
        shape=(25,25,25);lab=np.zeros(shape,np.uint8);lab[10,:,:]=2
        lab[8,10,10]=2
        layer=dict(support=np.zeros(shape,bool),height=np.full(shape,np.nan),votes=np.zeros(shape,np.uint16),evidence=np.zeros(shape),height_std=np.full(shape,np.nan),hit_z_max=np.full(shape,np.nan))
        new,surface,_=f.reconstruct(lab,layer,np.zeros(3,int),.1)
        self.assertEqual(new[10,10,10],2)
        self.assertEqual(new[12,10,10],0)
        self.assertEqual(new[8,10,10],1)
        self.assertFalse(surface['support'].any())

    def test_noisy_floor_uses_measured_foot_cell_and_nonzero_uncertainty(self):
        from covor.entry.surfaces import SupportSurfaces,SurfaceCfg
        f=SurfaceFusion();sd=self.camera();s=SupportSurfaces(SurfaceCfg(return_height_std=True,return_surface_xy=True));T=np.eye(4)
        T[:3,:3]=np.diag([1.,-1.,-1.]);T[:3,3]=[1,1,1]
        rng=np.random.default_rng(7);shape=(25,25,25);ijk=np.array([0,0,-5]);hit=np.full(shape,-np.inf)
        for _ in range(16):
            z=(1+rng.normal(0,.003,(60,80))).astype(np.float32);sigma=np.full_like(z,.003);valid=np.ones_like(z,bool)
            f.integrate(sd,z,sigma,valid,T);s.integrate(sd,z,sigma,valid,T)
            vv,uu=np.indices(z.shape);world=np.stack(((uu-sd.cx)*z/sd.fx,-(vv-sd.cy)*z/sd.fy,1-z),-1)+[1,1,0]
            q=np.floor(world.reshape(-1,3)/.1).astype(int)-ijk
            np.maximum.at(hit,tuple(q.T),world.reshape(-1,3)[:,2])
        lab=np.ones(shape,np.uint8);lab[:,:,4:6]=2;layer=s.dense(ijk,shape);layer['hit_z_max']=hit
        new,layer,_=f.reconstruct(lab,layer,ijk,.1)
        mask=layer['support'];self.assertGreater(mask.sum(),5)
        self.assertLess(np.max(abs(layer['height'][mask])),.01)
        self.assertTrue((layer['height_std'][mask]>=f.cfg.bias_bound).all())
        nodes=np.argwhere(mask);foot=np.floor((layer['height'][mask]+.025)/.1).astype(int)-ijk[2]
        np.testing.assert_array_equal(nodes[:,2],foot)
        self.assertTrue((layer['votes'][mask]>=2).all())


if __name__=='__main__':unittest.main()
