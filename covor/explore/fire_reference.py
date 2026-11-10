"""Independent mesh-convex geometry roadmap for multistorey scene evaluation.

No observed-map graph is used. Supports are sampled from upward mesh triangles;
body collision uses vertical clipping of each convex solid and exact XY polygon
distance for the same cylindrical human sizes. Fire visuals are excluded.
"""
from collections import defaultdict, deque
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from covor.entry.traverse import TravCfg


class PolygonSet:
    def __init__(self,polygons):
        self.polygons=polygons;self.bins=defaultdict(set)
        for i,p in enumerate(polygons):
            for x in range(int(np.floor(p['lo'][0]-.36)),int(np.floor(p['hi'][0]+.36))+1):
                for y in range(int(np.floor(p['lo'][1]-.36)),int(np.floor(p['hi'][1]+.36))+1):self.bins[x,y].add(i)

    def collision(self,xy,radius):
        hit=np.zeros(len(xy),bool)
        if len(xy)<16:
            ids=set()
            for q in xy:ids.update(self.bins.get(tuple(np.floor(q).astype(int)),()))
        else:ids=range(len(self.polygons))
        for i in ids:
            p=self.polygons[i];near=np.flatnonzero(~hit&((xy>=p['lo']-radius-1e-7)&(xy<=p['hi']+radius+1e-7)).all(1))
            if not len(near):continue
            points=xy[near];inside=np.ones(len(near),bool);distance=np.full(len(near),np.inf)
            for a,d,equation in zip(p['p'],p['d'],p['eq']):
                inside&=(points@equation[:2]+equation[2]<=1e-7) if radius==0 else (points@equation[:2]+equation[2]<-1e-7)
                t=np.clip((points-a)@d/max(d@d,1e-20),0,1)
                distance=np.minimum(distance,((points-a-t[:,None]*d)**2).sum(1))
            hit[near]=inside|(distance<max(0,radius-1e-5)**2)
        return hit


class MeshHumanReference:
    def __init__(self,building,res=.1):
        self.b=building;self.res=res;self.body=TravCfg(res=res);self.cache={}
        self.solids=[];self.solid_bins=defaultdict(set)
        for c in building.components:
            if not c['solid']:continue
            v=c['vertices'];edges=np.unique(np.sort(np.concatenate([c['faces'][:,[0,1]],c['faces'][:,[1,2]],c['faces'][:,[2,0]]]),axis=1),axis=0)
            self.solids.append(dict(v=v,edges=edges,lo=v.min(0),hi=v.max(0)))
            for x in range(int(np.floor((v[:,0].min()-.36)/2)),int(np.floor((v[:,0].max()+.36)/2))+1):
                for y in range(int(np.floor((v[:,1].min()-.36)/2)),int(np.floor((v[:,1].max()+.36)/2))+1):self.solid_bins[x,y].add(len(self.solids)-1)
        self.supports=self._supports();self.xyz=np.array([[ (i+.5)*res,(j+.5)*res,h+.025] for i,j,h in self.supports])
        self.posture=np.full(len(self.xyz),-1,np.int8);self.width=self.posture.copy()
        print('REFERENCE support candidates',building.id,len(self.xyz),flush=True)
        for posture in range(3):
            for width in range(2):
                ids=np.flatnonzero(self.posture<0)
                if not len(ids):break
                ok=self.clear(self.xyz[ids],posture,width);self.posture[ids[ok]]=posture;self.width[ids[ok]]=width
                print('REFERENCE body',posture,width,'remaining',int((self.posture<0).sum()),flush=True)
        good=self.posture>=0;self.xyz=self.xyz[good];self.posture=self.posture[good];self.width=self.width[good]
        self.keys=[k for k,keep in zip(self.supports,good) if keep];self.cols=defaultdict(list)
        for i,k in enumerate(self.keys):self.cols[k[:2]].append(i)
        print('REFERENCE valid nodes',building.id,len(self.xyz),flush=True)

    def _supports(self):
        nodes=set();r=self.res
        for c in self.b.components:
            if not c['solid']:continue
            v=c['vertices']
            for face in c['faces']:
                tri=v[face];n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=max(np.linalg.norm(n),1e-20)
                if n[2]<np.cos(np.deg2rad(30)):continue
                lo=np.ceil(tri[:,:2].min(0)/r-.5-1e-8).astype(int);hi=np.floor(tri[:,:2].max(0)/r-.5+1e-8).astype(int)
                if (hi<lo).any():continue
                ij=np.indices(hi-lo+1).reshape(2,-1).T+lo;xy=(ij+.5)*r
                a,b=tri[1,:2]-tri[0,:2],tri[2,:2]-tri[0,:2];den=a[0]*b[1]-a[1]*b[0]
                if abs(den)<1e-12:continue
                q=xy-tri[0,:2];u=(q[:,0]*b[1]-q[:,1]*b[0])/den;w=(a[0]*q[:,1]-a[1]*q[:,0])/den
                keep=(u>=-1e-8)&(w>=-1e-8)&(u+w<=1+1e-8)
                z=tri[0,2]-(q@n[:2])/n[2]
                keep&=(z<self.b.levels*3.3-.21)
                for (i,j),h in zip(ij[keep],z[keep]):nodes.add((int(i),int(j),round(float(h),5)))
        return sorted(nodes)

    def polygons(self,foot,posture,leg=False,tile=None):
        key=(round(float(foot),5),-1 if leg else posture,leg,tile)
        if key in self.cache:return self.cache[key]
        low=foot+1e-4 if leg else foot+self.body.step+1e-4
        high=foot+self.body.step if leg else foot+(self.body.h_stand,self.body.h_stoop,self.body.h_crawl)[posture]-1e-4
        polygons=[];unique=set()
        for ci in (range(len(self.solids)) if tile is None else sorted(self.solid_bins.get(tile,()))):
            c=self.solids[ci]
            if c['hi'][2]<=low or c['lo'][2]>=high:continue
            v=c['v'];points=[v[(v[:,2]>=low)&(v[:,2]<=high)]]
            for z in (low,high):
                a,b=v[c['edges'][:,0]],v[c['edges'][:,1]];mask=(a[:,2]-z)*(b[:,2]-z)<0
                if mask.any():
                    t=(z-a[mask,2])/(b[mask,2]-a[mask,2]);points.append(a[mask]+t[:,None]*(b[mask]-a[mask]))
            xy=np.concatenate(points)[:,:2]
            if len(xy)<3:continue
            try:hull=ConvexHull(xy)
            except QhullError:continue
            p=xy[hull.vertices];code=tuple(np.round(p,5).ravel())
            if code in unique:continue
            unique.add(code);polygons.append(dict(p=p,d=np.roll(p,-1,axis=0)-p,eq=hull.equations,lo=p.min(0),hi=p.max(0)))
        result=PolygonSet(polygons);self.cache[key]=result;return result

    def clear(self,points,posture,width):
        points=np.asarray(points);out=np.ones(len(points),bool);radius=(self.body.w,self.body.w_side)[width]/2-.001
        for z in np.unique(np.round(points[:,2],5)):
            ids=np.flatnonzero(abs(points[:,2]-z)<1e-5);xy=points[ids,:2]
            if len(ids)<32:
                tiles=np.floor(xy/2).astype(int)
                for tile in np.unique(tiles,axis=0):
                    mask=(tiles==tile).all(1);q=xy[mask];tile=tuple(tile)
                    out[ids[mask]]=~self.polygons(z,posture,True,tile).collision(q,0)&~self.polygons(z,posture,False,tile).collision(q,radius)
            else:out[ids]=~self.polygons(z,posture,True).collision(xy,0)&~self.polygons(z,posture).collision(xy,radius)
        return out

    def region(self,region):
        x0,y0,x1,y1=region['xy_bounds'];z0,z1=region['floor_z_bounds'];p=self.xyz
        return np.flatnonzero((p[:,0]>=x0)&(p[:,0]<=x1)&(p[:,1]>=y0)&(p[:,1]<=y1)&(p[:,2]>=z0)&(p[:,2]<=z1)).tolist()

    def solve(self):
        starts=self.region(self.b.entry)
        if not starts:raise ValueError('Independent geometry reference: unsupported/blocked entry')
        parent={u:None for u in starts};queue=deque(starts)
        while queue:
            u=queue.popleft();i,j,_=self.keys[u];a=self.xyz[u]
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di==dj==0:continue
                    for v in self.cols.get((i+di,j+dj),()):
                        b=self.xyz[v]
                        if v in parent or abs(a[2]-b[2])>.18+1e-8:continue
                        if di and dj and not all(any(abs(self.xyz[n,2]-a[2])<=.18+1e-8 and abs(self.xyz[n,2]-b[2])<=.18+1e-8 for n in self.cols.get(xy,())) for xy in ((i+di,j),(i,j+dj))):continue
                        p=max(self.posture[u],self.posture[v]);w=max(self.width[u],self.width[v])
                        steps=max(2,int(np.ceil(np.linalg.norm(b-a)/.025)))
                        mid=np.array([(1-t)*a+t*b for t in np.linspace(0,1,steps+1)[1:-1]]);mid[:,2]=max(a[2],b[2])
                        if not self.clear(mid,p,w).all():continue
                        parent[v]=u;queue.append(v)
        print('REFERENCE reached nodes',self.b.id,len(parent),'floors',sorted(set(int(self.xyz[u,2]//3.3) for u in parent)),flush=True)
        rows=[]
        for g in self.b.goals:
            hits=[u for u in self.region(g) if u in parent];path=[]
            if hits:
                u=hits[0]
                while u is not None:
                    path.append(dict(xyz=self.xyz[u].tolist(),posture=int(self.posture[u]),width=int(self.width[u]),observed_support=True));u=parent[u]
                path.reverse()
            rows.append(dict(id=g['id'],reachable=bool(hits),kind='multistorey_room_region',path=path,reference_method='mesh support triangles, clipped-convex cylindrical body, stepped roadmap'))
        nodes={k:np.r_[self.xyz[u,:2],k[2]+.005] for u,k in enumerate(self.keys)}
        modes={k:(int(self.posture[u]),int(self.width[u])) for u,k in enumerate(self.keys)}
        return rows,nodes,modes
