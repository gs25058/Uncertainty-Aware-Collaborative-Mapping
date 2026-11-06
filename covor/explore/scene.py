"""Complete, fixed collapsed-building boxes, mesh sensor geometry and independent truth.

Nothing in this module is passed to the candidate generator or motion planner.
All floor, ceiling and wall geometry is explicit. No scan holes or unknown GT.
"""
import heapq
from pathlib import Path
import json
import numpy as np
from covor.entry.traverse import TravCfg


class Building:
    def __init__(self,seed):
        self.seed=int(seed);self.boxes=[];rng=np.random.default_rng(seed)
        self.floor=.05;self.roof=2.65;self.bounds=np.array([[-.6,-2.6,-1.0],[12.6,9.6,3.0]])
        self.entry=dict(xy_bounds=[2.01,.61,2.09,.69],floor_z_bounds=[-.05,.25])
        self.starts=np.array([[x,-1.25,1.25] for x in (1.25,2.05,2.85,3.65)])
        self.pit=(7.35+float(rng.uniform(-.08,.08)),8.10+float(rng.uniform(-.06,.06)))
        def box(name,lo,hi):self.boxes.append(dict(name=name,lo=list(lo),hi=list(hi)))
        # A floor with an actual sunken pit; surrounding sides remain opaque.
        p0,p1=self.pit
        box('floor_south',(0,0,-.15),(12,p0,.05));box('floor_north',(0,p1,-.15),(12,9,.05))
        box('floor_east_of_pit',(3.4,p0,-.15),(12,p1,.05))
        box('pit_bottom',(0,p0,-.85),(3.4,p1,-.75))
        box('pit_south_face',(0,p0-.05,-.75),(3.4,p0,.05));box('pit_north_face',(0,p1,-.75),(3.4,p1+.05,.05))
        box('launch_floor',(.4,-2.1,-.15),(4.4,0,.05))
        box('roof',(0,0,2.65),(12,9,2.85))
        box('west_wall',(0,0,.05),(.18,9,2.65));box('east_wall',(11.82,0,.05),(12,9,2.65))
        box('north_wall',(0,8.82,.05),(12,9,2.65))
        box('entry_left',(0,0,.05),(1.4,.18,2.65));box('entry_right',(2.6,0,.05),(12,.18,2.65))
        # Vertical partitions with three openings.
        door_y=[1.55+float(rng.uniform(-.05,.05)),4.20,6.85]
        self.door_y=door_y
        def partition(x,widths):
            end=.18
            for n,(y,w) in enumerate(zip(door_y,widths)):
                a,b=y-w/2,y+w/2
                box(f'wall_{x}_{n}',(x-.1,end,.05),(x+.1,a,2.65));end=b
            box(f'wall_{x}_end',(x-.1,end,.05),(x+.1,8.82,2.65))
        partition(3.5,[1.1,1.1,1.1]);partition(7.5,[1.1,1.1,.40])
        # Horizontal partitions. The west A->B doorway is the real bypass.
        box('AB_west_left',(3.6,2.8,.05),(4.8,3.0,2.65))
        box('AB_west_right',(6.0,2.8,.05),(7.4,3.0,2.65))
        box('AB_east',(7.6,2.8,.05),(11.82,3.0,2.65))
        box('BC_partition',(3.6,5.4,.05),(11.82,5.6,2.65))
        # Human cannot step onto these barriers; drone can fly over them.
        h=float(rng.uniform(.78,.88))
        box('blocked_east_A',(7.28,door_y[0]-.55,.05),(7.72,door_y[0]+.55,h))
        box('blocked_main_B',(3.28,3.65,.05),(3.72,4.75,h))
        beam=float(rng.uniform(1.57,1.64))
        box('low_beam_east_B',(7.25,3.65,beam),(7.75,4.75,2.65))
        # Fixed non-blocking debris varies between scenes.
        box('side_debris',(9.4,float(rng.uniform(.4,.7)),.05),(10.4,1.1,.4))
        targets=[(5.5,1.55,'open'),(9.6,1.55,'unique_blocked'),(5.5,4.2,'bypass'),
                 (9.6,4.2,'low_beam'),(5.5,6.85,'open'),(9.6,6.85,'narrow'),(2.,8.45,'floor_hole')]
        self.goals=[dict(id=f'g{i}',xy_bounds=[x-.18,y-.18,x+.18,y+.18],floor_z_bounds=[-.05,.25]) for i,(x,y,_) in enumerate(targets)]
        self.annotations={f'g{i}':kind for i,(_,_,kind) in enumerate(targets)}
        self.lo=np.array([b['lo'] for b in self.boxes],float);self.hi=np.array([b['hi'] for b in self.boxes],float)
        self.scene=None;self.textures=None

    def prior(self):
        return dict(bounds=self.bounds.tolist(),exploration_box=[[.18,.18,.05],[11.82,8.82,2.65]],entry=self.entry,goals=self.goals,
            launch_free_box=[[.55,-1.95,.85],[4.25,-.15,1.65]],
            assumption='mission target regions supplied identically; internal walls/rooms/debris unknown')

    def labels(self,res=.1):
        ijk=np.floor(self.bounds[0]/res).astype(int);shape=np.ceil(self.bounds[1]/res).astype(int)-ijk
        lab=np.ones(shape,np.uint8)
        for lo,hi in zip(self.lo,self.hi):
            a=np.floor(lo/res+1e-8).astype(int)-ijk;b=np.ceil(hi/res-1e-8).astype(int)-ijk
            a=np.maximum(a,0);b=np.minimum(b,shape)
            lab[a[0]:b[0],a[1]:b[1],a[2]:b[2]]=2
        return lab,ijk

    def body_clear(self,p,posture,width,cfg=None):
        c=cfg or TravCfg();p=np.asarray(p,float);h=(c.h_stand,c.h_stoop,c.h_crawl)[posture]
        r=(c.w,c.w_side)[width]/2
        dx=np.maximum(np.maximum(self.lo[:,0]-p[0],p[0]-self.hi[:,0]),0)
        dy=np.maximum(np.maximum(self.lo[:,1]-p[1],p[1]-self.hi[:,1]),0)
        torso=(self.hi[:,2]>p[2]+c.step+1e-5)&(self.lo[:,2]<p[2]+h-1e-5)&(dx*dx+dy*dy<r*r-1e-9)
        leg=(self.hi[:,2]>p[2]+1e-4)&(self.lo[:,2]<p[2]+c.step)&(dx<1e-8)&(dy<1e-8)
        return not (torso|leg).any()

    def support_height(self,x,y,z,tol=.075):
        inside=(self.lo[:,0]<=x)&(self.hi[:,0]>=x)&(self.lo[:,1]<=y)&(self.hi[:,1]>=y)
        h=self.hi[inside,2];h=h[abs(h-z)<=tol]
        return float(h.max()) if len(h) else None

    def reference(self,res=.1):
        """Independent exact box geometry graph, not the observed-map classifier."""
        c=TravCfg(res=res);nodes={};modes={};counter=0
        for b in self.boxes:
            lo,hi=np.array(b['lo']),np.array(b['hi']);h=hi[2]
            if not -.9<=h<=1.5:continue
            for i in range(int(np.ceil(lo[0]/res-.5)),int(np.floor(hi[0]/res-.5))+1):
                for j in range(int(np.ceil(lo[1]/res-.5)),int(np.floor(hi[1]/res-.5))+1):
                    p=np.array([(i+.5)*res,(j+.5)*res,h+.005]);key=(i,j,round(float(h),5))
                    if key in nodes:continue
                    for posture in range(3):
                        for width in range(2):
                            if self.body_clear(p,posture,width,c):modes[key]=(posture,width);nodes[key]=p;break
                        if key in nodes:break
        cols={}
        for key in nodes:cols.setdefault(key[:2],[]).append(key)
        def region_keys(region):
            x0,y0,x1,y1=region['xy_bounds'];z0,z1=region['floor_z_bounds']
            return [k for k,p in nodes.items() if x0<=p[0]<=x1 and y0<=p[1]<=y1 and z0<=p[2]<=z1]
        starts=region_keys(self.entry);assert starts,'reference entry unsupported'
        parent={k:None for k in starts};queue=list(starts)
        for u in queue:
            i,j,_=u
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di==dj==0:continue
                    for v in cols.get((i+di,j+dj),[]):
                        if v in parent or abs(nodes[u][2]-nodes[v][2])>c.step+1e-8:continue
                        if di and dj and not all(any(abs(nodes[n][2]-nodes[u][2])<=c.step and abs(nodes[n][2]-nodes[v][2])<=c.step for n in cols.get(xy,[])) for xy in ((i,j+dj),(i+di,j))):continue
                        # Exact shape samples between nodes; small walls are >=.16 m.
                        posture=max(modes[u][0],modes[v][0]);width=max(modes[u][1],modes[v][1])
                        if not all(self.body_clear((1-t)*nodes[u]+t*nodes[v],posture,width,c) for t in (.25,.5,.75)):continue
                        parent[v]=u;queue.append(v)
        result=[]
        for goal in self.goals:
            hits=[k for k in region_keys(goal) if k in parent];path=[]
            if hits:
                k=sorted(hits)[0]
                while k is not None:
                    path.append(dict(xyz=nodes[k].tolist(),posture=modes[k][0],width=modes[k][1],observed_support=True));k=parent[k]
                path.reverse()
            result.append(dict(id=goal['id'],reachable=bool(hits),kind=self.annotations[goal['id']],path=path))
        assert any(r['reachable'] for r in result) and any(not r['reachable'] for r in result),'degenerate generated scene'
        return result,nodes,modes

    def render_scene(self):
        if self.scene is not None:return self.scene,self.textures
        import open3d as o3d
        import trimesh
        self.scene=o3d.t.geometry.RaycastingScene();textures=[];meshes=[]
        for n,(lo,hi) in enumerate(zip(self.lo,self.hi)):
            verts=[];faces=[];uv=[]
            for axis in range(3):
                others=[a for a in range(3) if a!=axis]
                for side in (0,1):
                    base=len(verts)
                    for a,b in ((0,0),(1,0),(1,1),(0,1)):
                        p=lo.copy();p[axis]=hi[axis] if side else lo[axis]
                        p[others[0]]=hi[others[0]] if a else lo[others[0]];p[others[1]]=hi[others[1]] if b else lo[others[1]]
                        verts.append(p);uv.append([a,b])
                    faces.extend(((base,base+1,base+2),(base,base+2,base+3)))
            verts=np.asarray(verts);faces=np.asarray(faces);uv=np.asarray(uv)
            mesh=trimesh.Trimesh(verts,faces,process=False);meshes.append(mesh)
            legacy=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),o3d.utility.Vector3iVector(faces))
            self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
            rng=np.random.default_rng(self.seed*1000+n)
            # Fixed fine texture, shared by both optical views, no pixel noise field.
            image=rng.integers(25,230,(1024,1024),dtype=np.uint8)
            textures.append((uv[faces],image))
        self.textures=textures;self.mesh=trimesh.util.concatenate(meshes)
        return self.scene,self.textures

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);self.render_scene()
        self.mesh.export(directory/'scene.ply')
        (directory/'geometry.json').write_text(json.dumps(dict(seed=self.seed,boxes=self.boxes,prior=self.prior(),annotations=self.annotations),indent=2)+'\n')
