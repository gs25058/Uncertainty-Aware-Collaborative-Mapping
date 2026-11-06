"""Observed-support and bounded-uncertainty graphs; default traverse is untouched.

Only this module's possible graph contains hypotheses. No hypothesis changes
labels, the confirmed graph, or evidence. Paths carry actual predecessor states.
"""
from dataclasses import dataclass, asdict
from collections import deque
import heapq
import itertools

import numpy as np
from scipy import ndimage

from . import traverse as TV
from .config import FREE, OCCUPIED, UNKNOWN


@dataclass(frozen=True)
class DecisionCfg:
    max_unknown_length_m: float = .5
    anchor_radius_m: float = .5
    anchor_height_rows: int = 2
    segment_extent_m: float = 1.5
    segment_height_m: float = .3
    # No XY or floor snap: entry/goal regions are fixed mission inputs.
    require_exact_entry: bool = True


def body_modes(lab, cfg, levels, allow_unknown):
    """Same centre, torso, surface-EDT and posture order as TV.node_modes."""
    p=np.full(lab.shape,-1,np.int8);w=p.copy()
    bad=lab==OCCUPIED if allow_unknown else lab!=FREE
    for k in levels:
        for posture,height in ((TV.STAND,cfg.h_stand),(TV.STOOP,cfg.h_stoop),(TV.CRAWL,cfg.h_crawl)):
            top=k+1+cfg.rows(height)
            if top>lab.shape[2]:continue
            centre=~bad[:,:,k+1:top].any(axis=2)
            lo=k+1+cfg.step_rows
            slab=bad[:,:,lo:top].any(axis=2)
            clear=TV._clear2d(slab,cfg.res)
            for mode,radius in ((TV.NORMAL,cfg.w/2),(TV.SIDEWAYS,cfg.w_side/2)):
                ok=centre&(clear>=radius-1e-9)&(p[:,:,k]<0)
                p[:,:,k][ok]=posture;w[:,:,k][ok]=mode
    return p,w


def surface_body_modes(lab,cfg,levels,foot_z,ijk_z,allow_unknown,head_margin=.02):
    """Conservative body slabs at measured foot height, including the head cell.

    The mixed support cell itself is resolved by the surface observation; all
    body cells above it must still be measured free for a confirmed mode.
    A fixed 2 cm head margin covers residual surface-height discretization.
    """
    p=np.full(lab.shape,-1,np.int8);w=p.copy()
    bad=lab==OCCUPIED if allow_unknown else lab!=FREE
    for k in levels:
        z=foot_z[:,:,k]
        for posture,height in ((TV.STAND,cfg.h_stand),(TV.STOOP,cfg.h_stoop),(TV.CRAWL,cfg.h_crawl)):
            tops=np.ceil((z+height+head_margin)/cfg.res-1e-9).astype(int)-ijk_z
            lows=np.floor((z+cfg.step)/cfg.res+1e-9).astype(int)-ijk_z
            for lo,top in np.unique(np.stack((lows,tops),-1).reshape(-1,2),axis=0):
                if top>lab.shape[2] or top<=k+1:continue
                centre=~bad[:,:,k+1:top].any(2)
                slab=bad[:,:,max(k+1,lo):top].any(2)
                group=(lows==lo)&(tops==top)
                for mode,radius in ((TV.NORMAL,cfg.w/2),(TV.SIDEWAYS,cfg.w_side/2)):
                    r=int(np.ceil(radius/cfg.res+.5));di,dj=np.mgrid[-r:r+1,-r:r+1]
                    # Exact disc-vs-square footprint; outside the map is not
                    # observed free. EDT's half-cell shortcut misses corners.
                    distance=np.hypot(np.maximum(abs(di)*cfg.res-cfg.res*.5,0),
                                      np.maximum(abs(dj)*cfg.res-cfg.res*.5,0))
                    footprint=distance<=radius+1e-9
                    clear=~ndimage.binary_dilation(slab,structure=footprint,border_value=1)
                    ok=group&centre&clear&(p[:,:,k]<0)
                    p[:,:,k][ok]=posture;w[:,:,k][ok]=mode
    return p,w


def local_segments(uncertain, ijk, res, cfg):
    """Connected local patches; split only components exceeding stated extent.

    A small connected doorway remains one patch across all its floor voxels.
    Long clouds split into fixed world tiles; IDs sorted by minimum voxel.
    """
    cc,n=ndimage.label(uncertain,np.ones((3,3,3)))
    objects=ndimage.find_objects(cc)
    patches=[]
    tile=np.maximum(np.rint(np.array([cfg.segment_extent_m,cfg.segment_extent_m,cfg.segment_height_m])/res).astype(int),1)
    for cid,slices in enumerate(objects,1):
        if slices is None:continue
        start=np.array([s.start for s in slices])
        xyz=np.argwhere(cc[slices]==cid)+start
        extent=(xyz.max(0)-xyz.min(0)+1)*res
        if (extent[:2]<=cfg.segment_extent_m+1e-8).all() and extent[2]<=cfg.segment_height_m+1e-8:
            patches.append(xyz);continue
        tiles=np.floor_divide(xyz+ijk,tile)
        for key in np.unique(tiles,axis=0):
            pts=xyz[(tiles==key).all(axis=1)]
            low=pts.min(0);shape=pts.max(0)-low+1
            sub=np.zeros(shape,bool);sub[tuple((pts-low).T)]=True
            labels,num=ndimage.label(sub,np.ones((3,3,3)))
            for label in range(1,num+1):patches.append(np.argwhere(labels==label)+low)
    patches.sort(key=lambda x:tuple(x.min(0)))
    region=np.zeros(uncertain.shape,np.int32);records=[]
    for sid,pts in enumerate(patches,1):
        region[tuple(pts.T)]=sid
        records.append(dict(id=sid,voxel_min=(pts.min(0)+ijk).tolist(),
                            voxel_max=(pts.max(0)+ijk).tolist(),n_nodes=len(pts)))
    return region,records


class DecisionGraph:
    def __init__(self,labels,ijk_min,res=.1,cfg=None,body=None,*,support_layer=None,surface_geometry=False,foot_contact_offset=.005):
        self.lab=np.asarray(labels,np.uint8)
        self.ijk=np.asarray(ijk_min,int);self.res=res
        self.cfg=cfg or DecisionCfg();self.body=body or TV.TravCfg(res=res)
        self.support_layer=support_layer
        self.surface_geometry=surface_geometry
        self.foot_contact_offset=foot_contact_offset
        self.foot_z=np.broadcast_to((np.arange(self.lab.shape[2])+self.ijk[2]+1)*res,self.lab.shape).copy()
        self.surface_candidates=np.zeros(self.lab.shape,bool)
        if surface_geometry:
            if support_layer is None:raise ValueError('surface_geometry requires observed surface heights')
            use=np.asarray(support_layer['support'])&np.isfinite(support_layer['height'])
            self.foot_z[use]=support_layer['height'][use]+foot_contact_offset
            self.surface_candidates=(support_layer['votes']>0)&(support_layer['evidence']>0)&np.isfinite(support_layer['height'])
            self.foot_z[self.surface_candidates]=support_layer['height'][self.surface_candidates]+foot_contact_offset
        if support_layer is None:
            self.support,self.cp,self.cw=TV.node_modes(self.lab,self.body)
        else:
            self.support=np.asarray(support_layer['support'],bool)
            if self.support.shape!=self.lab.shape:raise ValueError('support/map shape mismatch')
            self.cp,self.cw=body_modes(self.lab,self.body,np.unique(np.nonzero(self.support)[2]),False)
            if surface_geometry:
                self.cp,self.cw=surface_body_modes(self.lab,self.body,np.unique(np.nonzero(self.support)[2]),self.foot_z,self.ijk[2],False)
            self.cp[~self.support]=-1;self.cw[~self.support]=-1
        self.confirmed=self.cp>=0
        anchors3=self.support|self.surface_candidates
        levels=np.unique(np.nonzero(anchors3)[2])
        st=self.cfg.anchor_height_rows
        levels=sorted({int(k+d) for k in levels for d in range(-st,st+1)
                       if 0<=k+d<self.lab.shape[2]-self.body.rows(self.body.h_crawl)})
        near=np.zeros(self.lab.shape,bool)
        for k in levels:
            anchors=anchors3[:,:,max(k-st,0):min(k+st+1,self.lab.shape[2])].any(2)
            if anchors.any():
                distance,index=ndimage.distance_transform_edt(~anchors,return_indices=True)
                near[:,:,k]=distance*res<=self.cfg.anchor_radius_m+1e-9
                if surface_geometry:
                    # Infer a candidate at an observed adjacent surface height,
                    # not at every arbitrary +/- two-row height. No confirmation.
                    lo,hi=max(k-st,0),min(k+st+1,self.lab.shape[2])
                    heights=np.where(anchors3[:,:,lo:hi],self.foot_z[:,:,lo:hi],np.nan)
                    # Select the anchor height nearest this level, never a mean
                    # between two floors. XY nearest anchor is deterministic.
                    target=(k+self.ijk[2]+.5)*res
                    delta=np.where(np.isfinite(heights),abs(heights-target),np.inf)
                    choice=np.argmin(delta,axis=2)
                    z=np.take_along_axis(heights,choice[:,:,None],axis=2)[:,:,0]
                    z=z[tuple(index)]
                    safe=np.where(np.isfinite(z),z,target)
                    belongs=np.isfinite(z)&(np.floor((safe-foot_contact_offset)/res).astype(int)-self.ijk[2]==k)
                    near[:,:,k]&=belongs|anchors3[:,:,k]
                    infer=near[:,:,k]&~anchors3[:,:,k]
                    self.foot_z[:,:,k][infer]=z[infer]
        self.pp,self.pw=body_modes(self.lab,self.body,levels,True)
        if surface_geometry:self.pp,self.pw=surface_body_modes(self.lab,self.body,levels,self.foot_z,self.ijk[2],True)
        self.possible=(self.pp>=0)&near&((self.lab!=FREE)|self.support|self.surface_candidates)
        self.possible|=self.confirmed
        self.pp[self.confirmed]=self.cp[self.confirmed]
        self.pw[self.confirmed]=self.cw[self.confirmed]
        self.region,self.segments=local_segments(self.possible&~self.confirmed,self.ijk,res,self.cfg)
        self.xyz=np.argwhere(self.possible)
        self.ids=np.full(self.lab.shape,-1,np.int32)
        self.ids[tuple(self.xyz.T)]=np.arange(len(self.xyz))
        self._neighbors={}
        self._reach_transitions=None
        self._proof=None

    def mission_nodes(self,region,confirmed=False):
        x0,y0,x1,y1=region["xy_bounds"]
        h0,h1=region["floor_z_bounds"]
        p=(self.xyz+self.ijk+.5)*self.res
        feet=self.foot_z[tuple(self.xyz.T)]
        mask=(p[:,0]>=x0)&(p[:,0]<=x1)&(p[:,1]>=y0)&(p[:,1]<=y1)&(feet>=h0)&(feet<=h1)
        if confirmed and len(p):mask&=self.confirmed[tuple(self.xyz.T)]
        return np.flatnonzero(mask).tolist()

    def _raw_neighbors(self,u):
        if u in self._neighbors:return self._neighbors[u]
        i,j,k=self.xyz[u];st=self.body.step_rows
        out=[]
        for di,dj in itertools.product((-1,0,1),repeat=2):
            if di==dj==0:continue
            a,b=i+di,j+dj
            if not (0<=a<self.lab.shape[0] and 0<=b<self.lab.shape[1]):continue
            for kk in range(max(0,k-st),min(self.lab.shape[2]-1,k+st)+1):
                v=int(self.ids[a,b,kk])
                if v<0:continue
                if self.surface_geometry and abs(self.foot_z[i,j,k]-self.foot_z[a,b,kk])>self.body.step+1e-9:continue
                corners=[]
                if di and dj:
                    for x,y in ((i,b),(a,j)):
                        lo,hi=max(kk-st,0),min(kk+st+1,self.lab.shape[2])
                        ids=self.ids[x,y,lo:hi];ids=ids[ids>=0]
                        if self.surface_geometry:
                            ids=np.array([n for n in ids if abs(self.foot_z[tuple(self.xyz[n])]-self.foot_z[i,j,k])<=self.body.step+1e-9
                                 and abs(self.foot_z[tuple(self.xyz[n])]-self.foot_z[a,b,kk])<=self.body.step+1e-9],int)
                        corners.append(ids.tolist())
                    if not all(corners):continue
                length=self.res*(2**.5 if di and dj else 1.)
                out.append((v,length,corners))
        self._neighbors[u]=out
        return out

    def edges(self,u,confirmed_only=False,closed=None):
        if closed and self.region[tuple(self.xyz[u])]==closed:return
        for v,length,corners in self._raw_neighbors(u):
            deps=set()
            for node in (u,v):
                sid=int(self.region[tuple(self.xyz[node])])
                if sid:deps.add(sid)
            if closed in deps:continue
            ok=True
            for choices in corners:
                known=[n for n in choices if self.confirmed[tuple(self.xyz[n])]]
                available=known or [n for n in choices if self.region[tuple(self.xyz[n])]!=closed]
                if not available:ok=False;break
                sid=int(self.region[tuple(self.xyz[available[0]])])
                if sid:deps.add(sid)
            if not ok or (confirmed_only and deps):continue
            yield v,length,tuple(sorted(deps))

    def search(self,start_nodes,confirmed_only=False,closed=None,target_nodes=None):
        """Dijkstra; optional targets stop at the cheapest requested goal.

        Without targets the complete reached set is preserved for area/cut
        evaluation. Single-goal probes can avoid exploring the whole building.
        """
        # Run length resets only at an observed support with observed body room.
        # Integer milligrid distances distinguish axial and diagonal steps.
        starts=[(int(n),0) for n in start_nodes
                if (not confirmed_only or self.confirmed[tuple(self.xyz[n])])
                and (closed is None or self.region[tuple(self.xyz[n])]!=closed)]
        dist={s:0. for s in starts};parent={s:None for s in starts};via={}
        heap=[(0.,*s) for s in starts];heapq.heapify(heap)
        best={};limit=int(round(self.cfg.max_unknown_length_m/self.res*1000))
        targets=None if target_nodes is None else set(map(int,target_nodes))
        if targets==set():return dict(best=best,parent=parent,via=via,dist=dist)
        while heap:
            d,u,run=heapq.heappop(heap);state=(u,run)
            if d>dist.get(state,np.inf):continue
            if u not in best:best[u]=state
            if targets is not None and u in targets:break
            for v,length,deps in self.edges(u,confirmed_only,closed):
                uncertain=bool(deps)
                step=int(round(length/self.res*1000))
                nr=(run+step) if (run or uncertain) else 0
                if nr>limit:continue
                if self.confirmed[tuple(self.xyz[v])]:nr=0
                vp,vw=self.pp[tuple(self.xyz[v])],self.pw[tuple(self.xyz[v])]
                up,uw=self.pp[tuple(self.xyz[u])],self.pw[tuple(self.xyz[u])]
                cost=length*max(TV._factor(vp,vw,self.body),TV._factor(up,uw,self.body))
                rise=(abs(self.foot_z[tuple(self.xyz[v])]-self.foot_z[tuple(self.xyz[u])])
                      if self.surface_geometry else abs(self.xyz[v,2]-self.xyz[u,2])*self.res)
                cost+=self.body.c_rise*rise
                ns=(v,nr);nd=d+cost
                if nd<dist.get(ns,np.inf):
                    dist[ns]=nd;parent[ns]=state;via[ns]=deps
                    heapq.heappush(heap,(nd,v,nr))
        return dict(best=best,parent=parent,via=via,dist=dist)

    def path(self,search,targets):
        states=[search["best"][u] for u in targets if u in search["best"]]
        if not states:return [],[]
        node=min(states,key=lambda s:(search["dist"][s],s));chain=[];regions=set()
        while node is not None:
            chain.append(node[0]);regions.update(search["via"].get(node,()))
            sid=int(self.region[tuple(self.xyz[node[0]])])
            if sid:regions.add(sid)
            node=search["parent"][node]
        chain.reverse();out=[]
        for u in chain:
            i,j,k=self.xyz[u];p=(self.xyz[u]+self.ijk+.5)*self.res
            out.append(dict(node=int(u),ijk=(self.xyz[u]+self.ijk).tolist(),
                xyz=[float(p[0]),float(p[1]),float(self.foot_z[i,j,k])],
                posture=int(self.pp[i,j,k]),width=int(self.pw[i,j,k]),
                observed_support=bool(self.support[i,j,k]),confirmed=bool(self.confirmed[i,j,k]),
                segment=int(self.region[i,j,k])))
        return out,sorted(regions)

    def reachability_search(self,start_nodes,closed=None):
        """Exact reachability for cut probes, with dominated run states removed.

        At one node a smaller consecutive-unknown distance permits every
        continuation of a larger one. Costs do not affect a cut's reachability.
        Immutable predecessor records retain an auditable feasible witness.
        Normal route selection still uses the original cost-optimal search.
        """
        if self._reach_transitions is None:
            regions=self.region[tuple(self.xyz.T)];known=self.confirmed[tuple(self.xyz.T)]
            transitions=[]
            for u in range(len(self.xyz)):
                edges=[];ru=int(regions[u])
                for v,length,corners in self._raw_neighbors(u):
                    options=[]
                    for choices in corners:
                        if any(known[n] for n in choices):continue
                        options.append(tuple(sorted({int(regions[n]) for n in choices})))
                    rv=int(regions[v]);forced=tuple(x[0] for x in options if len(x)==1)
                    edges.append((v,int(round(length/self.res*1000)),rv,forced,options,bool(ru or rv or options)))
                transitions.append((ru,edges))
            self._reach_transitions=(transitions,known)
        transitions,known=self._reach_transitions
        run_at=[2**31-1]*len(self.xyz)
        queue=deque();parent={};via={};dist={};best={};serial=0
        for u in start_nodes:
            if closed is not None and self.region[tuple(self.xyz[u])]==closed:continue
            state=(int(u),serial);serial+=1;run_at[u]=0
            parent[state]=None;dist[state]=0.;queue.append((state,0))
        limit=int(round(self.cfg.max_unknown_length_m/self.res*1000))
        while queue:
            state,run=queue.popleft();u=state[0]
            if run>run_at[u]:continue
            best[u]=state
            ru,edges=transitions[u]
            for v,step,rv,forced,options,uncertain in edges:
                if closed is not None and (rv==closed or closed in forced):continue
                nr=run+step if run or uncertain else 0
                if nr>limit:continue
                if known[v]:nr=0
                if nr>=run_at[v]:continue
                run_at[v]=nr;child=(int(v),serial);serial+=1
                deps={s for s in (ru,rv) if s}
                for choices in options:deps.add(next(s for s in choices if s!=closed))
                parent[child]=state;via[child]=tuple(sorted(deps));dist[child]=dist[state]+step/1000*self.res
                if nr==0:queue.appendleft((child,nr))
                else:queue.append((child,nr))
        return dict(best=best,parent=parent,via=via,dist=dist)

    def blockage_proof(self,entry,goal):
        """Conservative certificate using a MORE permissive column relaxation.

        All heights and unlimited unknown runs are admitted, corner/step limits
        relaxed. An escape to an XY boundary precludes a blocked verdict. A
        complete observed floor/ceiling enclosure is additionally required.
        In incomplete scans this deliberately usually returns no certificate.
        """
        if self._proof is None:
            levels=range(self.lab.shape[2]-self.body.rows(self.body.h_crawl))
            p,_=body_modes(self.lab,self.body,levels,True)
            permissive=((p>=0)&(self.lab!=FREE)).any(2)
            cc,_=ndimage.label(permissive,np.ones((3,3)))
            self._proof=(permissive,cc)
        mask,cc=self._proof
        starts=self.mission_nodes(entry,True)
        if not starts:return None
        comp=int(cc[tuple(self.xyz[starts[0],:2])])
        if comp==0:return None
        component=cc==comp
        if component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():return None
        # Require observed floor and roof with no unknown interior over the
        # reachable relaxed component: no absent-floor or vertical-escape proof.
        for i,j in np.argwhere(component):
            supports=np.flatnonzero(self.support[i,j])
            if not len(supports):return None
            k=int(supports[0]);roof=np.flatnonzero(self.lab[i,j,k+1:]==OCCUPIED)+k+1
            if not len(roof):return None
            if (self.lab[i,j,k+1:roof[0]]==UNKNOWN).any():return None
        nodes=self.mission_nodes(goal)
        if nodes and any(cc[tuple(self.xyz[n,:2])]==comp for n in nodes):
            reason="fully_observed_height_or_body_disconnection"
        else:reason="observed_enclosing_obstacle_cut"
        return dict(reason=reason,enclosed_columns=int(component.sum()),
                    relaxation="all heights; unlimited unknown; 8-neighbour column relaxation")

    def decide(self,entry,goals,with_influence=False,progress=None):
        starts=self.mission_nodes(entry,True)
        possible_starts=self.mission_nodes(entry)
        confirmed=self.search(starts,True);possible=self.search(possible_starts)
        decisions=[];dependencies={}
        for goal in goals:
            cp,_=self.path(confirmed,self.mission_nodes(goal,True))
            pp,deps=self.path(possible,self.mission_nodes(goal))
            proof=None
            if cp:verdict="passable";reason="observed_support_and_body_path"
            elif pp:verdict="unverified";reason="short_uncertain_connections"
            elif not starts:verdict="unverified";reason="entry_not_observed_in_fixed_region"
            elif not self._region_inside(goal):verdict="unverified";reason="goal_outside_evaluation_scope"
            else:
                proof=self.blockage_proof(entry,goal)
                verdict="blocked" if proof else "unverified"
                reason=proof["reason"] if proof else "insufficient_coverage_or_possible_scope_escape"
            decisions.append(dict(goal=goal["id"],verdict=verdict,reason=reason,
                                  path=cp,possible_path=pp,uncertain_segments=deps,blockage_certificate=proof))
            if pp:dependencies[goal["id"]]=set(deps)
        impact={s["id"]:[] for s in self.segments};cut_evidence={}
        if with_influence:
            # Exact pruning: closing a segment outside a retained path cannot
            # destroy that path, so only those segment/goal pairs need a search.
            for sid in sorted(set().union(*dependencies.values()) if dependencies else set()):
                if progress:progress(int(sid))
                shut=self.reachability_search(possible_starts,closed=sid)
                evidence=[]
                for goal in goals:
                    if sid not in dependencies.get(goal["id"],set()):continue
                    closed_path,closed_deps=self.path(shut,self.mission_nodes(goal))
                    evidence.append(dict(goal=goal["id"],open_reachable=True,closed_reachable=bool(closed_path),
                                         closed_path=closed_path,closed_path_segments=closed_deps))
                    if not closed_path:
                        impact[sid].append(goal["id"])
                cut_evidence[sid]=dict(closed_segment=int(sid),other_segments="unchanged possible state",
                    n_open_reached_nodes=len(possible["best"]),n_closed_reached_nodes=len(shut["best"]),goals=evidence)
        return dict(decisions=decisions,confirmed=confirmed,possible=possible,impact=impact,
                    cut_evidence=cut_evidence,n_confirmed_nodes=int(self.confirmed.sum()),n_possible_nodes=len(self.xyz))

    def _region_inside(self,goal):
        x0,y0,x1,y1=goal["xy_bounds"]
        low=self.ijk[:2]*self.res;high=(self.ijk[:2]+self.lab.shape[:2])*self.res
        return low[0]<=x0<=x1<high[0] and low[1]<=y0<=y1<high[1]

    def region_observation_cells(self,sid,max_cells=512):
        pts=np.argwhere(self.region==sid)
        support=[];body=[]
        for i,j,k in pts:
            support.append((i,j,k))
            posture,width=int(self.pp[i,j,k]),int(self.pw[i,j,k])
            height=(self.body.h_stand,self.body.h_stoop,self.body.h_crawl)[posture]
            rad=(self.body.w if width==TV.NORMAL else self.body.w_side)/2
            r=int(np.ceil(rad/self.res+.5))
            top=k+1+self.body.rows(height);torso_low=k+1+self.body.step_rows
            if self.surface_geometry:
                top=int(np.ceil((self.foot_z[i,j,k]+height+.02)/self.res-1e-9))-self.ijk[2]
                torso_low=max(k+1,int(np.floor((self.foot_z[i,j,k]+self.body.step)/self.res+1e-9))-self.ijk[2])
            for di in range(-r,r+1):
                for dj in range(-r,r+1):
                    # Same half-voxel surface convention as the body clearance.
                    if np.hypot(di,dj)*self.res>rad+self.res*.5:continue
                    low=k+1 if di==dj==0 else torso_low
                    for kk in range(low,min(top,self.lab.shape[2])):
                        a,b=i+di,j+dj
                        if 0<=a<self.lab.shape[0] and 0<=b<self.lab.shape[1]:body.append((a,b,kk))
        def sample(points):
            a=np.unique(np.asarray(points,dtype=int).reshape(-1,3),axis=0)
            n=len(a)
            if n>max_cells:a=a[np.linspace(0,n-1,max_cells).astype(int)]
            return a,n
        return sample(support),sample(body)

    def area_result(self,result):
        shape=self.lab.shape[:2];reach=np.zeros(shape,bool)
        for n in result["best"]:reach[tuple(self.xyz[n,:2])]=True
        return reach
