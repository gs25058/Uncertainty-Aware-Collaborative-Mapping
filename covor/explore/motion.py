"""Conservative observed-free A*, FUEL native Polynomial and timed reservations.

A* / shortcut structure follows FUEL astar2.cpp; this is a standalone adapter,
not RACER's full planner. Native polynomial header is unmodified upstream code.
"""
import ctypes
import copy
import os
from pathlib import Path
import heapq
import numpy as np
from scipy import ndimage

EXTERNAL=Path(os.environ.get('EXTERNAL_OBSERVATION_DIR','external_observation'))

class FuelPolynomial:
    def __init__(self):
        self.lib=ctypes.CDLL(str(EXTERNAL/'standalone/libfuel_bridge.so'))
        p=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.fuel_eval.argtypes=[p,p,ctypes.c_double,ctypes.c_double,ctypes.c_int,p];self.lib.fuel_eval.restype=ctypes.c_int
    def evaluate(self,a,b,duration,t,derivative=0):
        out=np.zeros(3);ret=self.lib.fuel_eval(np.ascontiguousarray(a,dtype=float),np.ascontiguousarray(b,dtype=float),duration,t,derivative,out)
        if ret!=0:raise RuntimeError('FUEL Polynomial bridge failure')
        return out

NATIVE=FuelPolynomial()

def angle_delta(a,b):return float((b-a+np.pi)%(2*np.pi)-np.pi)

class Trajectory:
    def __init__(self,points,yaw,pitch,target_yaw,target_pitch,start,cfg,wait=0.):
        self.points=np.asarray(points,float);self.start=start;self.wait=wait;self.cfg=cfg
        self.yaw=yaw;self.pitch=pitch;self.dy=angle_delta(yaw,target_yaw);self.dp=target_pitch-pitch
        self.durations=[]
        for a,b in zip(self.points,self.points[1:]):
            L=np.linalg.norm(b-a)
            self.durations.append(max(.1,1.875*L/cfg['vmax'],np.sqrt(5.774*L/cfg['amax'])))
        self.move=sum(self.durations)
        self.rotation=max(1.875*abs(self.dy)/cfg['yaw_rate'],1.875*abs(self.dp)/cfg['pitch_rate'],.1)
        self.total=wait+max(self.move,self.rotation)
        self.end=start+self.total;self.stop_at=None
    def sample(self,t,derivative=0):
        t=min(t,self.stop_at) if self.stop_at is not None else t
        elapsed=float(t-self.start-self.wait)
        if elapsed<=0:return self.points[0].copy() if derivative==0 else np.zeros(3)
        for a,b,duration in zip(self.points,self.points[1:],self.durations):
            if elapsed<duration:return NATIVE.evaluate(a,b,duration,elapsed,derivative)
            elapsed-=duration
        return self.points[-1].copy() if derivative==0 else np.zeros(3)
    def orientation(self,t):
        s=np.clip((t-self.start-self.wait)/self.rotation,0,1);f=10*s**3-15*s**4+6*s**5
        return self.yaw+self.dy*f,self.pitch+self.dp*f
    def record(self):return dict(points=self.points.tolist(),durations=self.durations,start=self.start,wait=self.wait,end=self.end,yaw=self.yaw,pitch=self.pitch,dy=self.dy,dp=self.dp)

class FlightMap:
    def __init__(self,lab,ijk,res,cfg):
        self.lab=lab;self.ijk=np.asarray(ijk);self.res=res;self.cfg=cfg
        # Exact sphere vs voxel cube footprint, conservative margin included.
        r=cfg['radius']+cfg['margin'];n=int(np.ceil(r/res+.5));a,b,c=np.mgrid[-n:n+1,-n:n+1,-n:n+1]
        dist=np.sqrt(np.maximum(abs(a)*res-res/2,0)**2+np.maximum(abs(b)*res-res/2,0)**2+np.maximum(abs(c)*res-res/2,0)**2)
        self.safe=~ndimage.binary_dilation(lab!=1,structure=dist<=r+1e-9,border_value=1)
        self.offsets=[np.array(d) for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))]
    def avoid(self,centres):
        other=object.__new__(FlightMap);other.__dict__=self.__dict__.copy();other.safe=self.safe.copy()
        clearance=2*(self.cfg['radius']+self.cfg['margin'])
        n=int(np.ceil(clearance/self.res))+1
        for p in centres:
            q=self.index(p);lo=np.maximum(q-n,0);hi=np.minimum(q+n+1,self.safe.shape)
            cells=np.indices(tuple(hi-lo)).reshape(3,-1).T+lo
            distance=np.linalg.norm(self.world(cells)-p,axis=1)
            bad=cells[distance<clearance+self.res*.5]
            other.safe[tuple(bad.T)]=False
        other._components=None
        return other
    def index(self,p):return np.floor(np.asarray(p)/self.res).astype(int)-self.ijk
    def world(self,q):return (np.asarray(q)+self.ijk+.5)*self.res
    def point_clear(self,p):
        q=self.index(p)
        return bool(((q>=0)&(q<np.array(self.safe.shape))).all() and self.safe[tuple(q)])
    def segment(self,a,b):
        n=max(1,int(np.ceil(np.linalg.norm(np.asarray(b)-a)/(self.res*.25))))
        p=np.linspace(a,b,n+1);q=np.floor(p/self.res).astype(int)-self.ijk
        if not ((q>=0)&(q<np.array(self.safe.shape))).all():return False
        return bool(self.safe[tuple(q.T)].all())
    def route(self,a,b,max_expansions=35000):
        if not self.point_clear(a) or not self.point_clear(b):return None
        if self.segment(a,b):return [np.asarray(a),np.asarray(b)]
        start=tuple(self.index(a));dest=tuple(self.index(b))
        if not hasattr(self,'_components') or self._components is None:self._components=ndimage.label(self.safe)[0]
        if self._components[start]!=self._components[dest]:return None
        heap=[(0.,start)];cost={start:0.};parent={start:None};exp=0
        while heap:
            _,u=heapq.heappop(heap);exp+=1
            if u==dest:break
            if exp>max_expansions:return None
            for d in self.offsets:
                v=tuple(np.asarray(u)+d)
                if not all(0<=v[i]<self.safe.shape[i] for i in range(3)) or not self.safe[v]:continue
                nc=cost[u]+self.res
                if nc<cost.get(v,np.inf):
                    cost[v]=nc;parent[v]=u;heapq.heappush(heap,(nc+np.linalg.norm(np.asarray(v)-dest)*self.res,v))
        if dest not in parent:return None
        points=[];v=dest
        while v is not None:points.append(self.world(v));v=parent[v]
        points.reverse();points[0]=np.asarray(a);points[-1]=np.asarray(b)
        # Greedy line-of-sight shortcut; each tested segment is inflated FREE.
        out=[points[0]];i=0
        while i<len(points)-1:
            j=len(points)-1
            while j>i+1 and not self.segment(points[i],points[j]):j-=1
            out.append(points[j]);i=j
        return out


def reservation_conflict(traj,others,cfg,horizon=None):
    dt=cfg['reservation_dt'];end=max([traj.end]+[getattr(o,'end',traj.end) for o in others]) if horizon is None else horizon
    times=np.arange(traj.start,end+dt,dt)
    # Relative linear chord closest approach, plus conservative acceleration
    # sag bound for BOTH trajectories. Detects between-sample crossing/swap.
    p=np.array([traj.sample(t) for t in times]);limit=2*(cfg['radius']+cfg['margin'])+cfg['amax']*dt*dt/4
    for other in others:
        q=np.array([other.sample(t) for t in times]);r=p-q;d=np.diff(r,axis=0)
        f=np.clip(-np.sum(r[:-1]*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-15),0,1)
        if (np.linalg.norm(r[:-1]+f[:,None]*d,axis=1)<limit).any():return True
    return False

class Hover:
    def __init__(self,p):self.p=np.asarray(p)
    def sample(self,t):return self.p


def reserve(points,state,target,t,others,cfg):
    for wait in np.arange(0,cfg['max_wait']+.001,.5):
        tr=Trajectory(points,state['yaw'],state['pitch'],target['yaw'],target['pitch'],t,cfg,float(wait))
        if not reservation_conflict(tr,others,cfg):return tr
    return None


def truth_collision(a,b,lo,hi,radius):
    # Conservative swept sphere: 1 cm linear samples against exact AABBs.
    n=max(1,int(np.ceil(np.linalg.norm(np.asarray(b)-a)/.01)))
    p=np.linspace(a,b,n+1);d=np.maximum(np.maximum(lo[None]-p[:,None],p[:,None]-hi[None]),0)
    hit=np.linalg.norm(d,axis=2)<radius+1e-8
    return bool(hit.any())


def safe_rest_stop(traj,t,others,cfg):
    """Never replace a published trajectory with an incompatible stopped one.

    Every candidate includes its indefinite terminal hover. Sequential commit
    against ALL other current reservations preserves pairwise compatibility.
    """
    stops=traj.start+traj.wait+np.cumsum(traj.durations)
    for at in stops[stops>t+1e-8]:
        candidate=copy.copy(traj);candidate.stop_at=float(at);candidate.end=float(at)
        if not reservation_conflict(candidate,others,cfg):return candidate
    return None
