"""Versioned fire-scene adapter with opt-in, physically present bottom landings.

The existing gallery and frozen fire-study scenes are left unchanged. Geometry
and reference caches are isolated; policy observes only the same public prior.
"""
import hashlib,inspect,json,pickle
from pathlib import Path
from .fire_world import FireWorld
from .fire_reference import MeshHumanReference

class ReconFireWorld(FireWorld):
    def __init__(self,index,cfg):
        super().__init__(index,cfg)
        self.geometry_repairs=[]
        if cfg['scene'].get('repair_bottom_landings',False):
            for s in self.stairs:
                if s['floor']==0:continue
                x,y,f=s['x'],s['y'],s['floor']
                center=[x+.625,y-.29,f*self.storey-.10];size=[1.25,.58,.20]
                self.box('bottom_landing_repaired',center,size,f,'stair','stair')
                self.geometry_repairs.append(dict(kind='missing_bottom_landing',floor=f,center=center,size=size))
            self.scene=None
        self._references={}

    def geometry_hash(self):
        h=hashlib.sha256(json.dumps([self.entry,self.goals],sort_keys=True).encode())
        for c in self.components:
            if c['solid']:h.update(c['vertices'].tobytes());h.update(c['faces'].tobytes())
        return h.hexdigest()

    def reference_cache_path(self,res=.1):
        # Actual solids, mission regions, resolution and reference code.
        h=hashlib.sha256((self.geometry_hash()+str(res)+inspect.getsource(MeshHumanReference)).encode()).hexdigest()[:16]
        return Path(self.experiment_cfg['scene']['reference_cache'])/self.id/f'reference_recon_{res:.3f}_{h}.pkl'

    def reference(self,res=.1):
        if res in self._references:return self._references[res]
        cache=self.reference_cache_path(res)
        if cache.exists():
            with cache.open('rb') as f:result=pickle.load(f)
        else:
            result=MeshHumanReference(self,res).solve();cache.parent.mkdir(parents=True,exist_ok=True)
            with cache.open('wb') as f:pickle.dump(result,f)
        # A certified override is permitted only if it carries this geometry hash.
        certified=self.experiment_cfg['scene'].get('certified_reference_dir')
        if certified:
            path=Path(certified)/f'scene{self.index}.json'
            if path.exists():
                q=json.loads(path.read_text())
                if not isinstance(q,dict) or q.get('geometry_sha256')!=self.geometry_hash():raise ValueError('certified labels do not match this scene geometry')
                result=(q['targets'],result[1],result[2])
        self._references[res]=result;return result

    def metadata(self):
        return dict(super().metadata(),geometry_sha256=self.geometry_hash(),geometry_repairs=self.geometry_repairs,scene_revision='bottom_landing_opt_in_v1')
