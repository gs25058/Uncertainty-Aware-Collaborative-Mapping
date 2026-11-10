"""Adapter from the frozen multistorey library to the shared experiment runner."""
import pickle
import json,hashlib
from pathlib import Path
import numpy as np
from .fire_buildings import FireBuilding
from .fire_reference import MeshHumanReference


class FireWorld(FireBuilding):
    def __init__(self,index,cfg):
        super().__init__(int(index),fire_scale=cfg['scene']['fire_scale'])
        self.experiment_cfg=cfg;self._references={}
        # Use the physical entrance opening, which can contain an interior
        # partition, and a room arrival REGION rather than a furniture centre.
        # All coordinates are fixed before policy comparisons; geometry stays.
        self.entry=dict(xy_bounds=[self.width/2-.60,.35,self.width/2+.60,.95],floor_z_bounds=[-.05,.20])
        self.goals=[]
        for room in self.rooms:
            x0,y0,x1,y1=room['bounds'];z=room['floor']*3.3
            self.goals.append(dict(id=room['id'],xy_bounds=[x0+.05,y0+.05,x1-.05,y1-.05],floor_z_bounds=[z-.10,z+.25]))

    def reference(self,res=.1):
        if res in self._references:return self._references[res]
        digest=hashlib.sha256(json.dumps([self.entry,self.goals],sort_keys=True).encode()+Path(__file__).with_name('fire_reference.py').read_bytes()).hexdigest()[:12]
        cache=Path(self.experiment_cfg['scene']['reference_cache'])/self.id/f'reference_{res:.3f}_{digest}.pkl'
        if cache.exists():
            with cache.open('rb') as f:result=pickle.load(f)
        else:
            result=MeshHumanReference(self,res).solve();cache.parent.mkdir(parents=True,exist_ok=True)
            with cache.open('wb') as f:pickle.dump(result,f)
        certified=self.experiment_cfg['scene'].get('certified_reference_dir')
        if certified:
            path=Path(certified)/f'scene{self.index}.json'
            if path.exists():result=(json.loads(path.read_text()),result[1],result[2])
        self._references[res]=result;return result

    def save(self,out):self.export(out,self.experiment_cfg['map_resolution']);self.render_scene(False)

    def labels(self,res=.1):
        """Numerically stable positive-volume raster of the same solid mesh.

        A face at 10.6m must not acquire an extra cell because 10.6/.1 is
        105.999999999. This mirrors the legacy Building's boundary convention.
        Archived gallery voxels are preserved; experiment voxels are regenerated.
        """
        ijk=np.floor(self.bounds[0]/res).astype(int);shape=np.ceil(self.bounds[1]/res).astype(int)-ijk;lab=np.ones(shape,np.uint8);bound=np.full(shape,np.nan,np.float32)
        for c in self.components:
            if not c['solid']:continue
            v=c['vertices'];a=np.maximum(np.floor(v.min(0)/res+1e-8).astype(int)-ijk,0);b=np.minimum(np.ceil(v.max(0)/res-1e-8).astype(int)-ijk,shape)
            if (b<=a).any():continue
            q=np.indices(b-a).reshape(3,-1).T+a;points=(q+ijk+.5)*res;inside=np.ones(len(q),bool)
            for plane in c['planes']:inside&=points@plane[:3]+plane[3]<=res*.5*abs(plane[:3]).sum()+1e-7
            ids=tuple(q[inside].T);lab[ids]=2
            high=np.minimum((q[inside,2]+ijk[2]+1)*res,v[:,2].max());bound[ids]=np.fmax(bound[ids],high)
        self._gt_bound=(res,ijk,bound)
        return lab,ijk

    def complete_support(self,ijk,shape,res):
        """Validation only: all externally exposed upward faces, not just human
        standing nodes. A stair surface near a rail remains observed support
        even when a person cannot stand centred on that particular grid cell.
        """
        import inspect
        signature=hashlib.sha256((inspect.getsource(type(self).complete_support)+inspect.getsource(type(self).labels)+inspect.getsource(MeshHumanReference._supports)+str(res)).encode())
        for c in self.components:
            if c['solid']:signature.update(c['vertices'].tobytes());signature.update(c['faces'].tobytes())
        signature=signature.hexdigest();cache=Path(self.experiment_cfg['scene']['reference_cache'])/self.id/f'complete_surface_{res:.3f}.npz'
        origin=np.floor(self.bounds[0]/res).astype(int);fullshape=np.ceil(self.bounds[1]/res).astype(int)-origin
        if cache.exists():
            q=np.load(cache)
            if str(q['_signature'])==signature:
                off=ijk-origin;sl=tuple(slice(int(a),int(a+n)) for a,n in zip(off,shape));return {k:q[k][sl] for k in q.files if not k.startswith('_')}
        probe=MeshHumanReference.__new__(MeshHumanReference);probe.b=self;probe.res=res
        candidates=[[(i+.5)*res,(j+.5)*res,h] for i,j,h in probe._supports() if ijk[0]<=i<ijk[0]+shape[0] and ijk[1]<=j<ijk[1]+shape[1] and ijk[2]*res-.001<=h<(ijk[2]+shape[2])*res]
        # Sample every triangle-edge interval cut by the voxel grid. This
        # records partially covered stair cells at REAL surface XY positions.
        # Cell centres alone omit the front 1cm of a 29cm stair tread.
        for c in self.components:
            if not c['solid']:continue
            for face in c['faces']:
                tri=c['vertices'][face];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=max(np.linalg.norm(normal),1e-20)
                if normal[2]<np.cos(np.deg2rad(30)):continue
                if (tri.max(0)<ijk*res).any() or (tri.min(0)>(ijk+shape)*res).any():continue
                for i,j in ((0,1),(1,2),(2,0)):
                    start,end=tri[i],tri[j];delta=end-start;ts=[0.,1.]
                    for axis in (0,1):
                        if abs(delta[axis])<1e-12:continue
                        low,high=sorted((start[axis],end[axis]));grid=np.arange(np.floor(low/res)+1,np.ceil(high/res))*res
                        ts.extend(((grid-start[axis])/delta[axis]).tolist())
                    ts=np.unique(np.clip(ts,0,1))
                    for t in (ts[:-1]+ts[1:])/2:
                        point=(start+t*delta)*(1-1e-5)+tri.mean(0)*1e-5
                        q=np.floor(point/res).astype(int)-ijk
                        if ((q>=0)&(q<shape)).all():candidates.append(point.tolist())
        surface=np.asarray(candidates);points=surface+[0,0,.001];covered=np.zeros(len(points),bool)
        for c in self.components:
            if not c['solid']:continue
            lo=c['vertices'].min(0);hi=c['vertices'].max(0)
            ids=np.flatnonzero(~covered&(points>=lo).all(1)&(points<=hi).all(1))
            if len(ids):covered[ids]=((points[ids]@c['planes'][:,:3].T+c['planes'][:,3])<=1e-8).all(1)
        layer=dict(support=np.zeros(shape,bool),height=np.full(shape,np.nan,np.float32),height_std=np.zeros(shape,np.float32),surface_x=np.full(shape,np.nan,np.float32),surface_y=np.full(shape,np.nan,np.float32),votes=np.zeros(shape,np.uint16),evidence=np.zeros(shape,np.float32))
        surface=surface[~covered];q=np.floor(surface/res).astype(int)-ijk;good=((q>=0)&(q<shape)).all(1);surface=surface[good];q=q[good]
        flat=np.ravel_multi_index(q.T,shape);dist=((surface[:,:2]-(q[:,:2]+ijk[:2]+.5)*res)**2).sum(1)
        order=np.lexsort((dist,-np.round(surface[:,2],4),flat));pick=order[np.r_[True,flat[order][1:]!=flat[order][:-1]]]
        ids=tuple(q[pick].T);layer['support'][ids]=True;layer['height'][ids]=surface[pick,2];layer['surface_x'][ids]=surface[pick,0];layer['surface_y'][ids]=surface[pick,1];layer['votes'][ids]=2;layer['evidence'][ids]=2

        if not hasattr(self,'_gt_bound') or self._gt_bound[0]!=res:self.labels(res)
        _,origin,bound=self._gt_bound;off=ijk-origin;layer['hit_z_max']=bound[tuple(slice(int(a),int(a+n)) for a,n in zip(off,shape))].copy()
        if np.array_equal(ijk,origin) and np.array_equal(shape,fullshape):
            cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,_signature=signature,**layer)

        return layer

    def flight_collision(self,a,b,radius):
        import open3d as o3d
        scene,_=self.render_scene(False);n=max(1,int(np.ceil(np.linalg.norm(np.asarray(b)-a)/.04)))
        points=np.linspace(a,b,n+1).astype(np.float32)
        return bool((scene.compute_distance(o3d.core.Tensor(points)).numpy()<radius-1e-6).any())


def make_world(cfg,seed):
    if cfg.get('scene',{}).get('type')=='multistorey_fire':return FireWorld(seed,cfg)
    from .scene import Building
    return Building(seed)
