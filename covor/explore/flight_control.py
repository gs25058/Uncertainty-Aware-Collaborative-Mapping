"""Optional observed-map control experiments; no simulator geometry is accepted.

The legacy planner remains the default. A guarded controller can brake from its
actual current velocity/acceleration instead of waiting for a distant rest node.
All proposed stops must fit the observed FREE map and peer reservations.
"""
import copy
import numpy as np
from scipy import ndimage
from covor.occupancy import _dda_batch
from .motion import FlightMap,angle_delta,reservation_conflict


class WholeVoxelFlightMap(FlightMap):
    """Inflate between voxel cubes, protecting every sub-voxel flight position."""
    def __init__(self,lab,ijk,res,cfg):
        self.lab=lab;self.ijk=np.asarray(ijk);self.res=res;self.cfg=cfg
        self.offsets=[np.array(d) for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))]
        radius=cfg['radius']+cfg['margin'];n=int(np.ceil(radius/res+1))
        axes=np.mgrid[-n:n+1,-n:n+1,-n:n+1]
        distance=np.sqrt(np.sum(np.maximum(abs(axes)*res-res,0.)**2,axis=0))
        self.safe=~ndimage.binary_dilation(lab!=1,structure=distance<=radius+1e-9,border_value=1)

    def segment(self,a,b):
        _,voxels=_dda_batch(np.asarray(a,float),np.asarray(b,float).reshape(1,3),self.res)
        q=np.concatenate([voxels-self.ijk,np.array([self.index(a),self.index(b)])])
        return bool(((q>=0)&(q<self.safe.shape)).all() and self.safe[tuple(q.T)].all())

    def avoid(self,centres):
        base=super().avoid(centres);result=object.__new__(type(self));result.__dict__=base.__dict__
        return result


def make_flight_map(lab,ijk,res,cfg):
    return (WholeVoxelFlightMap if cfg.get('whole_voxel_safety') else FlightMap)(lab,ijk,res,cfg)


def split_rest_points(points,max_length):
    if not max_length:return points
    result=[np.asarray(points[0],float)]
    for a,b in zip(points,points[1:]):
        a=np.asarray(a,float);b=np.asarray(b,float)
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/max_length)))
        result.extend(a+(b-a)*(i/n) for i in range(1,n+1))
    return result


def follow_heading(state,t,dt,cfg):
    """Rate-limited travel heading, then the actual assigned observation yaw."""
    old=float(state['yaw']);tr=state.get('traj')
    target=state.get('item',{}).get('candidate',{}).get('yaw',old)
    if tr is not None:
        velocity=tr.sample(t,1)
        if np.linalg.norm(velocity[:2])>.05:
            target=float(np.arctan2(velocity[1],velocity[0]))
    return old+float(np.clip(angle_delta(old,target),-cfg['yaw_rate']*dt,cfg['yaw_rate']*dt))


def norm_extrema(coefficients,duration,derivative):
    """Exact polynomial stationary times, with endpoints, for vector norms."""
    c=np.asarray(coefficients,float)
    c=np.stack([np.polynomial.polynomial.polyder(q,m=derivative) for q in c])
    squared=np.zeros(2*c.shape[1]-1)
    for q in c:
        term=np.polynomial.polynomial.polymul(q,q);squared[:len(term)]+=term
    roots=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(squared))
    times=[0.,float(duration)]+[float(z.real) for z in roots if abs(z.imag)<1e-8 and 0<z.real<duration]
    values=np.stack([np.polynomial.polynomial.polyval(times,q) for q in c],axis=1)
    return float(np.linalg.norm(values,axis=1).max())


class BrakingTrajectory:
    """Quintic p/v/a continuity at replacement; zero terminal v/a.

    It implements the published trajectory interface and remains a stationary
    reservation after its end. It never snaps the drone to another position.
    """
    def __init__(self,p,v,a,end,duration,t,yaw,pitch,cfg):
        self.start=float(t);self.wait=0.;self.total=float(duration);self.end=self.start+self.total
        self.stop_at=self.end;self.cfg=cfg;self.yaw=float(yaw);self.pitch=float(pitch)
        self.dy=0.;self.dp=0.;self.rotation=.1
        self.points=np.array([p,end],float);self.durations=[self.total]
        p,v,a,end=map(lambda q:np.asarray(q,float),(p,v,a,end));T=self.total
        matrix=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
        residual=np.stack([end-p-v*T-a*T*T/2,-v-a*T,-a])
        self.coefficients=np.concatenate([p[:,None],v[:,None],a[:,None]/2,np.linalg.solve(matrix,residual).T],axis=1)
        self.axis_critical=[]
        for coef in self.coefficients:
            roots=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coef))
            self.axis_critical.append([float(z.real) for z in roots if abs(z.imag)<1e-8 and 0<z.real<T])

    def sample(self,t,derivative=0):
        u=float(np.clip(t-self.start,0,self.total))
        if t>=self.end:
            return self.points[-1].copy() if derivative==0 else np.zeros(3)
        return np.array([np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(q,m=derivative)) for q in self.coefficients])

    def orientation(self,t):return self.yaw,self.pitch

    def record(self):
        return dict(type='continuous_state_quintic_brake',points=self.points.tolist(),durations=self.durations,
            start=self.start,wait=0.,end=self.end,yaw=self.yaw,pitch=self.pitch,dy=0.,dp=0.,coefficients=self.coefficients.tolist())

    def within_limits(self):
        return (norm_extrema(self.coefficients,self.total,1)<=self.cfg['vmax']+1e-7
            and norm_extrema(self.coefficients,self.total,2)<=self.cfg['amax']+1e-7)


def observed_free_until(tr,t,at,flight):
    """Check complete spatial sweeps, including between sampled positions."""
    if isinstance(tr,BrakingTrajectory):
        times=np.linspace(max(0.,t-tr.start),min(tr.total,at-tr.start),max(1,int(np.ceil((at-t)/.1)))+1)
        for begin,end in zip(times,times[1:]):
            bounds=[]
            for coef,critical in zip(tr.coefficients,tr.axis_critical):
                xx=[begin,end]+[x for x in critical if begin<x<end]
                vv=np.polynomial.polynomial.polyval(xx,coef);bounds.append([min(vv),max(vv)])
            bounds=np.array(bounds);lo=flight.index(bounds[:,0]);hi=flight.index(bounds[:,1])
            if not ((lo>=0)&(hi<flight.safe.shape)).all():return False
            if not flight.safe[tuple(slice(int(a),int(b)+1) for a,b in zip(lo,hi))].all():return False
        return True
    # FUEL's legacy pieces are straight geometric segments with a monotone
    # quintic time law. Checking every crossed voxel of each piece is exact.
    ends=tr.start+tr.wait+np.cumsum(tr.durations)
    times=[t]+[float(x) for x in ends if t<x<at]+[at]
    return all(flight.segment(tr.sample(a),tr.sample(b)) for a,b in zip(times,times[1:]))


def guarded_stop(tr,state,t,others,flight,cfg):
    """Prefer an existing known-free rest node, then a continuous braking stop.

    Returning None is a real control failure. The caller retains the published
    motion, logs it, and lets the ordinary simulator count any collision.
    """
    stops=tr.start+tr.wait+np.cumsum(tr.durations)
    for at in stops[stops>t+1e-8]:
        if not observed_free_until(tr,t,float(at),flight):break
        candidate=copy.copy(tr);candidate.stop_at=float(at);candidate.end=float(at)
        if not reservation_conflict(candidate,others,cfg):return candidate,'existing_rest'
    p=tr.sample(t);v=tr.sample(t,1);a=tr.sample(t,2)
    for duration in cfg.get('braking_durations_s',[.5,.75,1.,1.25,1.5,2.,2.5,3.]):
        end=p+v*duration*.5
        candidate=BrakingTrajectory(p,v,a,end,duration,t,state['yaw'],state['pitch'],cfg)
        if not candidate.within_limits():continue
        if not flight.point_clear(end) or not observed_free_until(candidate,t,candidate.end,flight):continue
        if not reservation_conflict(candidate,others,cfg):return candidate,'continuous_brake'
    return None,'no_known_free_joint_safe_brake'
