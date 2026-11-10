import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from covor.explore.motion import FuelPolynomial
from covor.explore.execution_acceleration import ExactPolynomialCache,DeferredDenseReads


class ExactNativeControls(unittest.TestCase):
    def test_guard_and_off_sensor_consumers_get_latest_sensor_map(self):
        from types import SimpleNamespace
        class Map:
            def __init__(self):self.version=0;self.builder=SimpleNamespace(tree=SimpleNamespace(updateInnerOccupancy=lambda:None))
            def dense(self):return np.array([self.version]),{}
        class Execution:
            period=8.;record=None
            def prepare(self,*args):pass
            def restore_map(self,m):return None
            def maybe_save(self,*args):pass
        ex=Execution();d=DeferredDenseReads(ex,Map)
        cfg=dict(sensor_period=1.,assessment_period=8.,assignment_period=4.,snapshot_period=20.,motion=dict(controller='guarded_v1',safety_period_s=.5))
        ex.prepare(cfg,0,0,'clean','unknown',4,1.75,.25);m=Map();last=None
        for tick in range(8):
            if tick%4==0:m.version=tick;last=m.dense()
            if tick in (0,2,4,6,7):self.assertEqual(int(last[0][0]),m.version)
            ex.maybe_save(tick,m,{'T':1.75},[],0.,*last)
        self.assertEqual(d.needed,{0,4})

    def test_unused_dense_reads_keep_native_updates_and_all_consumers_fresh(self):
        from types import SimpleNamespace
        class Map:
            def __init__(self):
                self.version=0;self.updates=0
                self.builder=SimpleNamespace(tree=SimpleNamespace(updateInnerOccupancy=self.update))
            def update(self):self.updates+=1
            def dense(self):self.update();return np.array([self.version]),dict(version=self.version)
        class Execution:
            period=8.;record=None
            def prepare(self,*args):pass
            def restore_map(self,m):return None
            def maybe_save(self,tick,m,runtime,logs,elapsed,lab,layer):
                if tick%32==0 or tick==36:assert lab[0]==m.version
        cfg=dict(assessment_period=2.,assignment_period=1.,snapshot_period=5.)
        ex=Execution();d=DeferredDenseReads(ex,Map);ex.prepare(cfg,0,0,'clean','unknown',4,9.,.25);m=Map()
        for tick in range(37):
            m.version=tick;lab,layer=m.dense()
            if tick%4==0 or tick==36:self.assertEqual(int(lab[0]),tick)
            ex.maybe_save(tick,m,{'T':9.},[],0.,lab,layer)
        self.assertEqual(m.updates,37);self.assertEqual(d.full_reads,10);self.assertEqual(d.deferred_reads,27)

    def test_checkpoint_restore_materializes_before_advancing_tick(self):
        from types import SimpleNamespace
        class Map:
            def __init__(self):self.builder=SimpleNamespace(tree=SimpleNamespace(updateInnerOccupancy=lambda:None))
            def dense(self):return np.array([23]),{}
        class Execution:
            period=8.;record={'tick':32}
            def prepare(self,*args):pass
            def restore_map(self,m):return m.dense()
            def maybe_save(self,*args):pass
        ex=Execution();d=DeferredDenseReads(ex,Map);ex.prepare(dict(assessment_period=8.,assignment_period=4.,snapshot_period=20.),0,0,'clean','entry',4,16.,.25)
        ex.restore_map(Map());self.assertEqual(d.tick,33);self.assertEqual(d.full_reads,1)

    def test_native_results_bitwise_exact_and_outputs_independent(self):
        native=FuelPolynomial();cached=ExactPolynomialCache(native.evaluate,128);rng=np.random.default_rng(7281)
        for _ in range(32):
            a,b=rng.normal(size=(2,3));duration=float(rng.uniform(.1,10));t=float(rng.uniform(0,duration))
            for derivative in (0,1,2):
                expected=native.evaluate(a,b,duration,t,derivative)
                self.assertEqual(cached(a,b,duration,t,derivative).tobytes(),expected.tobytes())
                v=cached(a,b,duration,t,derivative);self.assertEqual(v.tobytes(),expected.tobytes());v[:]=999
                self.assertEqual(cached(a,b,duration,t,derivative).tobytes(),expected.tobytes())
        self.assertEqual(cached.hits,192)
    def test_distinct_float_bit_patterns_and_mutated_inputs_do_not_alias(self):
        native=FuelPolynomial();cached=ExactPolynomialCache(native.evaluate)
        a=np.array([0.,1.,2.]);b=np.array([2.,3.,4.]);cached(a,b,1.,.5)
        a[0]=1.;expected=native.evaluate(a,b,1.,.5)
        self.assertEqual(cached(a,b,1.,.5).tobytes(),expected.tobytes());self.assertEqual(cached.misses,2)
        cached(a,b,1.,-0.);cached(a,b,1.,+0.);self.assertEqual(cached.misses,4)


if __name__=='__main__':unittest.main()
