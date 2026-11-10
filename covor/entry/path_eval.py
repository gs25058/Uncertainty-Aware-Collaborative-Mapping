"""Independent reference-geometry validation of the path actually proposed.

No replanning is done here. Mesh rays test support and swept body chords;
reference voxels establish which otherwise collision-free volume was observed.
Finite spatial sampling is explicit in PathEvalCfg, not a fire-safety guarantee.
"""
from dataclasses import dataclass
import numpy as np

from . import traverse as TV


@dataclass(frozen=True)
class PathEvalCfg:
    path_spacing_m: float = .025
    body_spacing_m: float = .05
    support_tolerance_m: float = .075
    step_limit_m: float = None  # None preserves archived voxel-rounded rules.
    contact_epsilon_m: float = 0.0  # Numerical tangency tolerance; archived default.


def region_contains(p,region):
    x0,y0,x1,y1=region["xy_bounds"];z0,z1=region["floor_z_bounds"]
    return x0-1e-9<=p[0]<=x1+1e-9 and y0-1e-9<=p[1]<=y1+1e-9 and z0-1e-9<=p[2]<=z1+1e-9


def motion_path(path):
    """Explicitly shrink at departure, expand at arrival; never change route."""
    if not path:return []
    out=[dict(path[0])]
    for a,b in zip(path,path[1:]):
        # codes increase as height/width shrink (stand=0, crawl=2; side=1).
        posture=max(a["posture"],b["posture"]);width=max(a["width"],b["width"])
        if out[-1]["posture"]!=posture or out[-1]["width"]!=width:
            out.append(dict(a,posture=posture,width=width,mode_transition=True))
        out.append(dict(b,posture=posture,width=width))
        if posture!=b["posture"] or width!=b["width"]:out.append(dict(b,mode_transition=True))
    return out


class ReferenceValidator:
    def __init__(self,labels,ijk,res,scene,body=None,cfg=None):
        self.lab=labels;self.ijk=np.asarray(ijk);self.res=res;self.scene=scene
        self.body=body or TV.TravCfg(res=res);self.cfg=cfg or PathEvalCfg()
        self.cache={};self.support_cache={};self.templates={}

    @property
    def step_height(self):
        return self.cfg.step_limit_m if self.cfg.step_limit_m is not None else self.body.step_rows*self.res

    def _cast(self,origins,directions):
        import open3d as o3d
        rays=np.c_[origins,directions].astype(np.float32)
        return self.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()

    def support(self,p):
        key=tuple(np.round(p,6))
        if key in self.support_cache:return self.support_cache[key]
        tol=self.cfg.support_tolerance_m
        start=np.asarray(p)+[0,0,tol]
        hit=self._cast(start[None,:],np.array([[0.,0.,-1.]]))[0]
        # A float32 ray exactly on a stair triangle edge can miss both faces.
        # Resolve only within the already declared numerical contact tolerance;
        # this does not snap a foot across a geometric gap or to another stair.
        eps=self.cfg.contact_epsilon_m
        if eps>0 and (not np.isfinite(hit) or hit>2*tol+1e-6):
            origins=start+np.array([[eps,0,0],[-eps,0,0],[0,eps,0],[0,-eps,0]])
            alternatives=self._cast(origins,np.tile([0.,0.,-1.],(4,1)))
            valid=alternatives[np.isfinite(alternatives)&(alternatives<=2*tol+1e-6)]
            if len(valid):hit=valid[np.argmin(abs(valid-tol))]
        ind=np.floor((np.asarray(p)-[0,0,self.res*.5])/self.res).astype(int)-self.ijk
        inside=((ind>=0)&(ind<np.array(self.lab.shape))).all()
        value=int(self.lab[tuple(ind)]) if inside else 0
        if np.isfinite(hit) and hit<=2*tol+1e-6:
            result=("valid",float(start[2]-hit))
        elif value==1:result=("invalid",None)
        else:result=("unverifiable",None)
        self.support_cache[key]=result;return result

    def body_clear(self,p,posture,width):
        key=(*np.round(p,5),posture,width)
        if key in self.cache:return self.cache[key]
        p=np.asarray(p,float)
        if (posture,width) not in self.templates:
            self.templates[posture,width]=self._body_template(posture,width)
        offsets,directions,limits,coverage=self.templates[posture,width]
        origins=offsets+p;eps=1e-4
        hits=self._cast(origins,directions)
        collision=np.flatnonzero((hits>=0)&(hits<limits-eps))
        if len(collision):
            q=int(collision[0]);result=("invalid","mesh_body_collision",(origins[q]+directions[q]*hits[q]).tolist())
        else:
            pts=coverage+p
            ind=np.floor(pts/self.res).astype(int)-self.ijk
            inside=((ind>=0)&(ind<np.array(self.lab.shape))).all(1)
            unknown=~inside
            unknown[inside]=self.lab[tuple(ind[inside].T)]==0
            result=("unverifiable","reference_body_coverage_unknown",pts[np.flatnonzero(unknown)[0]].tolist()) if unknown.any() else ("valid",None,None)
        self.cache[key]=result;return result

    def _body_template(self,posture,width):
        p=np.zeros(3)
        height=(self.body.h_stand,self.body.h_stoop,self.body.h_crawl)[posture]
        radius=(self.body.w if width==TV.NORMAL else self.body.w_side)/2-self.cfg.contact_epsilon_m
        spacing=self.cfg.body_spacing_m;eps=1e-4
        xy=np.arange(-radius,radius+spacing*.5,spacing)
        xy=np.unique(np.r_[xy,0.])
        origins=[];directions=[];limits=[];coverage=[]
        # Vertical rods plus horizontal chords detect horizontal, vertical and
        # oblique surfaces. Legs are centre-only below the shared step height.
        for dx in xy:
            for dy in xy:
                if dx*dx+dy*dy>radius*radius+1e-9:continue
                low=self.step_height+eps
                if abs(dx)+abs(dy)<1e-8:low=eps
                origins.append(p+[dx,dy,low]);directions.append([0,0,1]);limits.append(height-low)
                for z in np.arange(low,height,spacing):coverage.append(p+[dx,dy,z])
        for z in np.unique(np.r_[np.arange(self.step_height+eps,height,spacing),height-eps]):
            for offset in xy:
                extent=np.sqrt(max(radius**2-offset**2,0.))
                if extent<=eps:continue
                origins.extend((p+[-extent,offset,z],p+[offset,-extent,z]))
                directions.extend(([1,0,0],[0,1,0]));limits.extend((2*extent,2*extent))
        origins=np.asarray(origins);directions=np.asarray(directions);limits=np.asarray(limits)
        return origins,directions,limits,np.asarray(coverage)

    def validate(self,path,entry,goal):
        if not path:return dict(status="unverifiable",reason="no_proposed_path",first_failure=None)
        for index,region in ((0,entry),(-1,goal)):
            if not region_contains(path[index]["xyz"],region):
                return dict(status="invalid",reason="wrong_entry_or_goal_region_or_floor",first_failure=path[index]["xyz"])
        unknown=None;previous_height=None
        gap=any(not p.get("observed_support",True) for p in path)
        # Check the declared foot placements; intermediate movement samples are
        # body sweeps, not extra artificial foot placements on a stair riser.
        for n,point in enumerate(path):
            state,height=self.support(point["xyz"])
            if state=="invalid":return dict(status="invalid",reason="observed_no_foot_support",first_failure=point["xyz"],index=n,gap_dependency=gap)
            if state=="unverifiable" and unknown is None:unknown=("reference_support_unknown",point["xyz"],n)
            if height is not None and previous_height is not None and abs(height-previous_height)>self.step_height+1e-6:
                return dict(status="invalid",reason="reference_step_height_exceeded",first_failure=point["xyz"],index=n,gap_dependency=gap)
            previous_height=height
            check=self.body_clear(point["xyz"],point["posture"],point["width"])
            if check[0]=="invalid":return dict(status="invalid",reason=check[1],first_failure=check[2],index=n,gap_dependency=gap)
            if check[0]=="unverifiable" and unknown is None:unknown=(check[1],check[2],n)
        for n,(a,b) in enumerate(zip(path,path[1:])):
            pa,pb=np.asarray(a["xyz"]),np.asarray(b["xyz"])
            if abs(pa[2]-pb[2])>self.step_height+1e-6:
                return dict(status="invalid",reason="declared_step_height_exceeded",first_failure=pb.tolist(),index=n+1,gap_dependency=gap)
            steps=max(1,int(np.ceil(np.linalg.norm(pb-pa)/self.cfg.path_spacing_m)))
            for t in np.linspace(0,1,steps+1)[1:-1]:
                p=(1-t)*pa+t*pb
                # A step raises the body before horizontal movement; only
                # endpoint foot placements require support at the declared Z.
                p[2]=max(pa[2],pb[2])
                check=self.body_clear(p,max(a["posture"],b["posture"]),max(a["width"],b["width"]))
                if check[0]=="invalid":return dict(status="invalid",reason=check[1],first_failure=check[2],index=n,gap_dependency=gap)
                if check[0]=="unverifiable" and unknown is None:unknown=(check[1],check[2],n)
        if unknown:return dict(status="unverifiable",reason=unknown[0],first_failure=unknown[1],index=unknown[2],gap_dependency=gap)
        return dict(status="valid",reason="mesh_support_and_swept_body_verified",first_failure=None,gap_dependency=gap)
