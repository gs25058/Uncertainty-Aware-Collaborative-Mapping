"""Experimental observed-depth TSDF geometry, with public-quality weighting.

No scene/reference API is used. A coarse FREE correction requires every fine
grid corner to have repeated positive-distance evidence. Support is still gated
by actual plane observations; no unobserved floor is created.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FusionCfg:
    voxel_size: float = .025
    truncation: float = .10
    reference_sigma: float = .006
    min_evidence: float = 1.2
    correlation_time: float = .75
    capture_period: float = .25
    bias_bound: float = .0015
    free_margin: float = .005
    variance_weights: tuple = (.25,.55,.20)
    finite_time_correlation: bool = False
    local_contact_refinement: bool = False
    contact_radius: float = .01


class SurfaceFusion:
    def __init__(self,cfg=None):
        import open3d as o3d
        self.cfg=cfg or FusionCfg();self.frames=0;self.last_time=-np.inf
        names=('tsdf','weight','sigma_sum','variance_sum','range_sigma_sum','range_variance_sum','outlier_height_sum','outlier_range_sum')
        if self.cfg.finite_time_correlation:names+=('time_epoch','time_left','time_right','time_closed','range_time_left','range_time_right','range_time_closed')
        self.vbg=o3d.t.geometry.VoxelBlockGrid(
            attr_names=names,
            attr_dtypes=(o3d.core.float32,)*len(names),attr_channels=((1,),)*len(names),
            voxel_size=self.cfg.voxel_size,block_resolution=16,block_count=10000,
            device=o3d.core.Device('CPU:0'))

    def integrate(self,sd,Z,sigma,valid,T,teammates=None,dyn_radius=.35,independent_variance=None,time=None):
        import open3d as o3d
        c=self.cfg;Trect=np.asarray(T).copy();Trect[:3,:3]=T[:3,:3]@sd.R1.T
        if c.finite_time_correlation and time is None:raise ValueError('actual measurement time is required')
        if independent_variance is not None and not c.finite_time_correlation:raise ValueError('sparse-error variance requires the finite correlation model')
        if c.finite_time_correlation:
            if not np.isfinite(time) or time<self.last_time:raise ValueError('observation times must be finite and monotonic')
            self.last_time=float(time)
        depth=np.where(valid&np.isfinite(Z)&np.isfinite(sigma),Z,0).astype(np.float32)
        if teammates is not None and len(teammates):
            vv,uu=np.indices(depth.shape)
            rect=np.stack(((uu-sd.cx)*depth/sd.fx,(vv-sd.cy)*depth/sd.fy,depth),-1)
            world=rect@Trect[:3,:3].T+Trect[:3,3]
            for mate in teammates:depth[np.linalg.norm(world-np.asarray(mate),axis=-1)<dyn_radius]=0
        if not (depth>0).any():return
        image=o3d.t.geometry.Image(o3d.core.Tensor(depth))
        K=o3d.core.Tensor([[sd.fx,0.,sd.cx],[0.,sd.fy,sd.cy],[0.,0.,1.]],o3d.core.float64)
        E=o3d.core.Tensor(np.linalg.inv(Trect),o3d.core.float64)
        mult=c.truncation/c.voxel_size
        try:
            blocks=self.vbg.compute_unique_block_coordinates(image,K,E,1.,sd.cfg.z_max,mult)
        except RuntimeError as error:
            if 'No block is touched' not in str(error):raise
            # Open3D's depth-touch prepass subsamples pixels and can miss a
            # narrow valid aperture. Use ALL remaining measured endpoints.
            vv,uu=np.nonzero(depth>0);zz=depth[vv,uu]
            points=np.c_[(uu-sd.cx)*zz/sd.fx,(vv-sd.cy)*zz/sd.fy,zz]@Trect[:3,:3].T+Trect[:3,3]
            cloud=o3d.t.geometry.PointCloud(o3d.core.Tensor(points.astype(np.float32)))
            blocks=self.vbg.compute_unique_block_coordinates(cloud,mult)
        self.vbg.hashmap().activate(blocks)
        buffers,present=self.vbg.hashmap().find(blocks)
        if not present.numpy().all():raise RuntimeError('active TSDF block missing')
        coords,flat=self.vbg.voxel_coordinates_and_flattened_indices(buffers)
        ids=flat.numpy().reshape(-1);xyz=coords.numpy()
        w=self.vbg.attribute('weight').numpy().reshape(-1)
        sdf=self.vbg.attribute('tsdf').numpy().reshape(-1)
        oldw=w[ids].copy();olds=sdf[ids].copy()
        self.vbg.integrate(blocks,image,K,E,1.,sd.cfg.z_max,mult)
        changed=w[ids]>oldw+.5
        ix=ids[changed];before=oldw[changed]
        sample=sdf[ix]*(before+1)-olds[changed]*before
        optical=(xyz[changed]-Trect[:3,3])@Trect[:3,:3]
        u=np.rint(sd.fx*optical[:,0]/optical[:,2]+sd.cx).astype(int)
        v=np.rint(sd.fy*optical[:,1]/optical[:,2]+sd.cy).astype(int)
        h,width=depth.shape;inside=(u>=0)&(u<width)&(v>=0)&(v<h)
        sz=np.full(len(ix),np.inf);sz[inside]=sigma[v[inside],u[inside]]
        gain=np.abs((np.c_[(u-sd.cx)/sd.fx,(v-sd.cy)/sd.fy,np.ones(len(ix))]@Trect[:3,:3].T)[:,2])
        extra=np.zeros(len(ix))
        if independent_variance is not None:extra[inside]=independent_variance[v[inside],u[inside]]
        q=np.where(np.isfinite(sz),c.reference_sigma**2/(c.reference_sigma**2+sz**2+extra),0.)
        total=before+q
        sdf[ix]=np.divide(before*olds[changed]+q*sample,total,out=np.zeros_like(total),where=total>0)
        w[ix]=total
        height_sigma=np.where(np.isfinite(sz),sz*gain,0.)
        self.vbg.attribute('sigma_sum').numpy().reshape(-1)[ix]+=q*height_sigma
        self.vbg.attribute('variance_sum').numpy().reshape(-1)[ix]+=(q*height_sigma)**2
        ray_gain=np.linalg.norm(np.c_[(u-sd.cx)/sd.fx,(v-sd.cy)/sd.fy,np.ones(len(ix))],axis=1)
        range_sigma=np.where(np.isfinite(sz),sz*ray_gain,0.)
        self.vbg.attribute('range_sigma_sum').numpy().reshape(-1)[ix]+=q*range_sigma
        self.vbg.attribute('range_variance_sum').numpy().reshape(-1)[ix]+=(q*range_sigma)**2
        if independent_variance is not None:
            self.vbg.attribute('outlier_height_sum').numpy().reshape(-1)[ix]+=q*q*extra*gain*gain
            self.vbg.attribute('outlier_range_sum').numpy().reshape(-1)[ix]+=q*q*extra*ray_gain*ray_gain
        if c.finite_time_correlation:
            # All cameras may share the same latent field. Assume perfect
            # spatial/cross-drone correlation; only use the KNOWN finite time
            # interpolation kernel. No true noise field values are consulted.
            epoch=int(np.floor(time/c.correlation_time));fraction=time/c.correlation_time-epoch
            normalizer=np.hypot(1-fraction,fraction);left=(1-fraction)/normalizer;right=fraction/normalizer
            e=self.vbg.attribute('time_epoch').numpy().reshape(-1);old_epoch=e[ix].copy()
            fresh=before==0;advance=(old_epoch<epoch)&~fresh;adjacent=advance&(old_epoch==epoch-1);jump=advance&~adjacent
            if np.any((old_epoch>epoch)&~fresh):raise ValueError('observation times must be monotonic')
            total_sigma=np.sqrt(np.where(np.isfinite(sz),sz*sz,0.)+extra)
            for prefix,factor in [('',gain),('range_',ray_gain)]:
                a=self.vbg.attribute(prefix+'time_left').numpy().reshape(-1);b=self.vbg.attribute(prefix+'time_right').numpy().reshape(-1);v=self.vbg.attribute(prefix+'time_closed').numpy().reshape(-1)
                va=a[ix].copy();vb=b[ix].copy()
                v[ix[advance]]+=va[advance]**2
                v[ix[jump]]+=vb[jump]**2
                a[ix[advance]]=np.where(adjacent[advance],vb[advance],0.)
                b[ix[advance]]=0.;a[ix[fresh]]=0.;b[ix[fresh]]=0.;v[ix[fresh]]=0.
                increment=q*total_sigma*factor
                a[ix]+=increment*left;b[ix]+=increment*right
            e[ix]=epoch
        self.frames+=1

    def arrays(self):
        coords,flat=self.vbg.voxel_coordinates_and_flattened_indices()
        ids=flat.numpy().reshape(-1);xyz=coords.numpy()
        values={key:self.vbg.attribute(key).numpy().reshape(-1)[ids] for key in ('tsdf','weight','sigma_sum','variance_sum','range_sigma_sum','range_variance_sum')}
        for name in ('outlier_height_sum','outlier_range_sum'):
            try:values[name]=self.vbg.attribute(name).numpy().reshape(-1)[ids]
            except RuntimeError:values[name]=np.zeros(len(ids),np.float32)
        if self.cfg.finite_time_correlation:
            for prefix in ('','range_'):
                a=self.vbg.attribute(prefix+'time_left').numpy().reshape(-1)[ids]
                b=self.vbg.attribute(prefix+'time_right').numpy().reshape(-1)[ids]
                v=self.vbg.attribute(prefix+'time_closed').numpy().reshape(-1)[ids]
                values[prefix+'time_variance']=v+a*a+b*b
        good=values['weight']>=self.cfg.min_evidence
        keys=np.rint(xyz[good]/self.cfg.voxel_size).astype(np.int64)
        return keys,{k:v[good] for k,v in values.items()}

    def reconstruct(self,labels,layer,ijk,res,foot_offset=.025):
        """Observed map adapter. Human rules and reference remain unchanged."""
        c=self.cfg;ratio=int(round(res/c.voxel_size))
        if abs(ratio*c.voxel_size-res)>1e-8:raise ValueError('coarse resolution must be a multiple of TSDF spacing')
        keys,values=self.arrays()
        if not len(keys):return labels.copy(),{k:v.copy() for k,v in layer.items()},dict(repeated_fine_samples=0,coarse_free_corrections=0,actual_input_frames=self.frames)
        lo=keys.min(0);hi=keys.max(0)+1;shape=hi-lo
        def codes(q):return ((q[:,0]-lo[0])*shape[1]+q[:,1]-lo[1])*shape[2]+q[:,2]-lo[2]
        ids=codes(keys);order=np.argsort(ids);ids=ids[order]
        def lookup(q):
            inside=((q>=lo)&(q<hi)).all(1);code=codes(q);pos=np.searchsorted(ids,code)
            found=inside&(pos<len(ids));found[found]&=ids[pos[found]]==code[found]
            return found,np.minimum(pos,max(0,len(ids)-1))
        # A sample is a grid corner, not a cell centre. Require ALL (r+1)^3
        # corners, including the upper faces, before correcting a coarse voxel.
        result=labels.copy();candidates=np.unique(np.floor_divide(keys,ratio)-ijk,axis=0)
        inside=((candidates>=0)&(candidates<labels.shape)).all(1);candidates=candidates[inside]
        candidates=candidates[labels[tuple(candidates.T)]!=1]
        offsets=np.indices((ratio+1,)*3).reshape(3,-1).T
        rho=np.exp(-c.capture_period/c.correlation_time);correlation=(1+rho)/(1-rho)
        environment,temporal,independent=c.variance_weights
        w=values['weight'][order]
        def uncertainty(s,v):
            return environment*(s/np.maximum(w,1e-12))**2+(temporal*correlation+independent)*v/np.maximum(w,1e-12)**2+c.bias_bound**2
        range_variance=uncertainty(values['range_sigma_sum'][order],values['range_variance_sum'][order])+values['outlier_range_sum'][order]/np.maximum(w,1e-12)**2
        if c.finite_time_correlation:range_variance=values['range_time_variance'][order]/np.maximum(w,1e-12)**2+c.bias_bound**2
        sorted_sdf=values['tsdf'][order]
        positive=sorted_sdf*c.truncation>c.free_margin+3*np.sqrt(range_variance)
        corrected=0
        for chunk in np.array_split(candidates,max(1,int(np.ceil(len(candidates)/4096)))):
            if not len(chunk):continue
            query=((chunk+ijk)[:,None,:]*ratio+offsets).reshape(-1,3)
            found,pos=lookup(query);ok=(found&positive[pos]).reshape(len(chunk),-1).all(1)
            result[tuple(chunk[ok].T)]=1;corrected+=int(ok.sum())
        mesh=self.vbg.extract_triangle_mesh(weight_threshold=c.min_evidence).to_legacy()
        mesh.compute_vertex_normals();points=np.asarray(mesh.vertices);normal=np.asarray(mesh.vertex_normals)
        if not len(points):raise ValueError('no repeated observed surface')
        # Preserve all reconstructed obstacle heights, including non-floor faces.
        hit=np.full(labels.shape,-np.inf,np.float32)
        raw=np.floor(points/res).astype(int)-ijk;inside=((raw>=0)&(raw<labels.shape)).all(1)
        result[tuple(raw[inside].T)]=2
        np.maximum.at(hit,tuple(raw[inside].T),points[inside,2])
        # Keep original upper hits unless repeated fine evidence resolves the
        # full partial foot cell. A fused floor must not erase an unresolved rail.
        fused_hit=hit;hit=np.array(layer['hit_z_max'],copy=True)
        local_mesh=None
        if c.local_contact_refinement:
            from .contact_geometry import ContactGeometry
            local_mesh=ContactGeometry(res,c.voxel_size,c.contact_radius);local_mesh.integrate_world(points)
        floor=np.abs(normal[:,2])>=np.cos(np.deg2rad(30.))
        points=points[floor]
        # Store the support at the cell containing the ACTUAL foot coordinate,
        # instead of letting noise split one floor into two support rows.
        target=np.floor((points+[0,0,foot_offset])/res).astype(int)-ijk
        valid=((target>=0)&(target<labels.shape)).all(1);points=points[valid];target=target[valid]
        unique,inverse=np.unique(target,axis=0,return_inverse=True)
        counts=np.bincount(inverse);height=np.bincount(inverse,weights=points[:,2])/counts
        point_order=np.argsort(inverse);group_starts=np.r_[0,np.cumsum(counts)]
        variance=np.maximum(0,np.bincount(inverse,weights=points[:,2]**2)/counts-height**2)
        nearest=np.rint(points/c.voxel_size).astype(int);found,pos=lookup(nearest)
        variance_height=uncertainty(values['sigma_sum'][order],values['variance_sum'][order])+values['outlier_height_sum'][order]/np.maximum(w,1e-12)**2
        if c.finite_time_correlation:variance_height=values['time_variance'][order]/np.maximum(w,1e-12)**2+c.bias_bound**2
        point_var=np.full(len(points),np.inf);point_var[found]=variance_height[pos[found]]
        posterior=np.zeros(len(unique));np.maximum.at(posterior,inverse,point_var)
        # Absence of a confident fused patch is NOT evidence that a previously
        # measured/tentative support disappeared. Retain those observations;
        # otherwise the possibility graph acquires artificial closed cuts.
        output={key:value.copy() for key,value in layer.items()}
        # The fusion model has a persistent calibration/bias floor.  Keeping
        # an already accepted raw support must not silently remove that floor:
        # downstream contact checks consume this single uncertainty field for
        # both retained and newly fused supports.
        retained=output['support']&np.isfinite(output['height_std'])
        output['height_std'][retained]=np.maximum(output['height_std'][retained],c.bias_bound)
        output['hit_z_max']=hit
        if c.local_contact_refinement:output['contact_hit_z_max']=layer.get('contact_hit_z_max',layer['hit_z_max']).copy()
        def locally_free(samples):
            grid=samples/c.voxel_size;lower=np.floor(grid).astype(int);fraction=grid-lower
            signed=np.zeros(len(samples));sigma=np.zeros(len(samples));all_seen=np.ones(len(samples),bool)
            for delta in np.ndindex(2,2,2):
                delta=np.array(delta);weight=np.prod(np.where(delta,fraction,1-fraction),axis=1)
                found,index=lookup(lower+delta);all_seen&=found|(weight<1e-10)
                signed+=weight*sorted_sdf[index]*c.truncation
                # Worst-case cross-corner correlation; do not count spatial
                # interpolation as independent measurements.
                sigma+=weight*np.sqrt(range_variance[index])
            return bool(np.all(all_seen&(signed>3*sigma+c.free_margin)))
        for i,key in enumerate(unique):
            x,y,k=key;support=False;votes=0;evidence=0.
            # Gate by ACTUAL accepted planar frame evidence. Max, never sum
            # votes from multiple height bins belonging to the same frame.
            for z in range(max(0,k-2),min(labels.shape[2],k+2)):
                h=float(layer['height'][x,y,z])
                if np.isfinite(h) and abs(h-height[i])<.05:
                    support|=bool(layer['support'][x,y,z]);votes=max(votes,int(layer['votes'][x,y,z]));evidence=max(evidence,float(layer['evidence'][x,y,z]))
            if not support:continue
            std=np.sqrt(variance[i]+posterior[i])
            if not np.isfinite(std) or std>.05:continue
            corners=(key+ijk)[None,:]*ratio+offsets
            bottom=np.floor((height[i]+foot_offset)/c.voxel_size)*c.voxel_size
            above=corners[:,2]*c.voxel_size>=bottom-1e-9
            full,pos=lookup(corners[above])
            if not c.local_contact_refinement and len(full) and np.all(full&positive[pos]):hit[tuple(key)]=fused_hit[tuple(key)]
            # Replace only the retained raw support that represents this same
            # physical floor.  Leaving its adjacent height-bin support active
            # creates two foot rows for one noisy plane; do not touch a truly
            # separate floor at the same horizontal cell.
            old_h=output['height'][x,y]
            same_floor=output['support'][x,y]&np.isfinite(old_h)&(np.abs(old_h-height[i])<.05)
            output['support'][x,y,same_floor]=False
            q=tuple(key);output['support'][q]=True;output['height'][q]=height[i]
            output['height_std'][q]=std;output['votes'][q]=votes;output['evidence'][q]=evidence
            cloud=points[point_order[group_starts[i]:group_starts[i+1]]];centre=(key[:2]+ijk[:2]+.5)*res
            representative=cloud[np.argmin(((cloud[:,:2]-centre)**2).sum(1))]
            output['surface_x'][q],output['surface_y'][q]=representative[:2]
            if c.local_contact_refinement and std*3+.005<=foot_offset:
                xy=representative[:2]+np.array([[0,0],[c.contact_radius,0],[-c.contact_radius,0],[0,c.contact_radius],[0,-c.contact_radius]])
                bottom=height[i]+foot_offset;top=(k+ijk[2]+1)*res
                zs=np.linspace(bottom,max(bottom,top),max(2,int(np.ceil((top-bottom)/c.voxel_size))+1))
                samples=np.array([[x,y,z] for x,y in xy for z in zs])
                if locally_free(samples):
                    probe=dict(height=np.full((1,1,1),height[i]),surface_x=np.full((1,1,1),representative[0]),surface_y=np.full((1,1,1),representative[1]),hit_z_max=np.full((1,1,1),-np.inf))
                    local=local_mesh.dense(probe,np.array([0,0,k+ijk[2]]))[0,0,0]
                    output['contact_hit_z_max'][q]=local
        return result,output,dict(repeated_fine_samples=len(keys),coarse_free_corrections=corrected,measured_supports=int(output['support'].sum()),mesh_vertices=len(mesh.vertices),actual_input_frames=self.frames)
