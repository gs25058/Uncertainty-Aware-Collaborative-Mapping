"""Exact memoization of pure native Polynomial calls; no numerical approximation."""
from collections import OrderedDict
import struct
import numpy as np


class ExactPolynomialCache:
    def __init__(self,evaluate,maxsize=65536):
        self.original=evaluate;self.maxsize=maxsize;self.cache=OrderedDict();self.hits=0;self.misses=0

    def __call__(self,a,b,duration,t,derivative=0):
        aa=np.ascontiguousarray(a,dtype=float);bb=np.ascontiguousarray(b,dtype=float)
        if aa.shape!=(3,) or bb.shape!=(3,):return self.original(a,b,duration,t,derivative)
        key=(aa.tobytes(),bb.tobytes(),struct.pack('ddi',float(duration),float(t),int(derivative)))
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key].copy()
        self.misses+=1;value=self.original(a,b,duration,t,derivative)
        self.cache[key]=value.copy()
        if len(self.cache)>self.maxsize:self.cache.popitem(last=False)
        return value


def enable_exact_motion_cache():
    from .motion import NATIVE
    if isinstance(NATIVE.evaluate,ExactPolynomialCache):return NATIVE.evaluate
    cached=ExactPolynomialCache(NATIVE.evaluate);NATIVE.evaluate=cached;return cached


class DeferredDenseReads:
    """Avoid materializing arrays on ticks where nothing consumes them.

    Native inner-node occupancy is still updated on EVERY original dense call,
    preserving OctoMap's internal state. Integrations and sensor ticks are never
    deferred. Arrays are fresh at assessment, assignment, snapshot, checkpoint,
    and final ticks. This adapter is optional execution I/O, not a map model.
    """
    def __init__(self,execution,map_class):
        import weakref
        self.execution=execution;self.cache=weakref.WeakKeyDictionary()
        self.tick=0;self.needed=set();self.force=False
        self.full_reads=0;self.deferred_reads=0
        original_prepare=execution.prepare;original_restore=execution.restore_map
        original_save=execution.maybe_save;original_dense=map_class.dense

        def prepare(cfg,scene,seed,condition,policy,n,T,dt):
            result=original_prepare(cfg,scene,seed,condition,policy,n,T,dt)
            end=int(round(T/dt));next_assess=next_assign=next_snapshot=next_guard=next_sensor=0.;last_sensor=0
            guard=cfg.get('motion',{}).get('controller')=='guarded_v1'
            for tick in range(end+1):
                t=tick*dt;needed=False
                if t+1e-8>=next_sensor:
                    last_sensor=tick;next_sensor+=cfg.get('sensor_period',dt)
                if t+1e-8>=next_assess or tick==end:
                    needed=True;next_assess+=cfg['assessment_period']
                if t+1e-8>=next_snapshot:
                    needed=True;next_snapshot+=cfg['snapshot_period']
                if t+1e-8>=next_assign and t<T:
                    needed=True;next_assign+=cfg['assignment_period']
                if tick%int(round(execution.period/dt))==0 or tick==end:needed=True
                if guard and t+1e-8>=next_guard:
                    needed=True;next_guard+=cfg['motion'].get('safety_period_s',1.)
                if needed:self.needed.add(last_sensor)
            self.tick=execution.record['tick'] if execution.record is not None else 0
            return result

        def restore(m):
            self.force=True
            try:result=original_restore(m)
            finally:self.force=False
            if execution.record is not None:self.tick=execution.record['tick']+1
            return result

        def save(tick,m,runtime,logs,elapsed,lab,layer):
            assert tick==self.tick,(tick,self.tick)
            try:return original_save(tick,m,runtime,logs,elapsed,lab,layer)
            finally:self.tick=tick+1

        def dense(m):
            if self.force or self.tick in self.needed or m not in self.cache:
                result=original_dense(m);self.cache[m]=result;self.full_reads+=1
                return result
            # classify_points has exactly this one mutation. Retain it even on
            # unused intermediate reads; later integrations see identical tree.
            m.builder.tree.updateInnerOccupancy();self.deferred_reads+=1
            return self.cache[m]

        execution.prepare=prepare;execution.restore_map=restore;execution.maybe_save=save
        map_class.dense=dense
