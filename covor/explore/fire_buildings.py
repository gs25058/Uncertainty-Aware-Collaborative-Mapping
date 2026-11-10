"""Fixed, multistorey fire-damaged mesh scenes, independent of legacy experiments.

Collision geometry contains the actual openings, stairs, closed door leaves and
fallen debris. Fire/smoke annotations are visualization/source metadata, not solid
objects or calibrated fire physics. No reference reachability labels are invented.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
import trimesh


CATALOG = [
    ('apartment', '01 · 공동주택', 2, 22., 16., '거실 화재와 층간 바닥 붕괴', 'home'),
    ('school', '02 · 학교 교사', 3, 26., 18., '교실 화재·복도 벽 파손·천장 낙하', 'school'),
    ('office', '03 · 사무동', 3, 24., 18., '사무실 화재와 무너진 천장 패널', 'office'),
    ('hotel', '04 · 소형 호텔', 3, 23., 17., '잠긴 객실과 손상된 보조 계단', 'hotel'),
    ('clinic', '05 · 의원 건물', 2, 24., 17., '진료실 화재·잔해가 쌓인 대기 공간', 'clinic'),
    ('retail', '06 · 상가 건물', 2, 26., 19., '매장 화재·쇼윈도 파손·큰 바닥 구멍', 'retail'),
    ('workshop', '07 · 작업장', 2, 25., 18., '자재 화재·지붕 붕괴·작업대 잔해', 'workshop'),
    ('dormitory', '08 · 기숙사', 3, 25., 17., '객실 화재와 길게 이어지는 복도', 'hotel'),
    ('duplex', '09 · 복층 주거동', 2, 20., 16., '큰 벽 붕괴와 불탄 실내 가구', 'home'),
    ('library', '10 · 도서관 별관', 3, 26., 19., '서가 화재·개방형 열람 공간·층간 개구부', 'library'),
]

PALETTE = dict(slab=[.49,.47,.43], wall=[.73,.69,.59], exterior=[.57,.48,.40],
    frame=[.19,.23,.25], glass=[.22,.48,.53], door=[.36,.22,.12], locked=[.48,.17,.12],
    debris=[.44,.40,.35], steel=[.22,.23,.23], furniture=[.40,.28,.19],
    fabric=[.29,.38,.36], stair=[.62,.59,.53], tile=[.68,.66,.58], char=[.095,.085,.075],
    pipe=[.34,.36,.35], marker=[.17,.50,.36], flame=[1.,.33,.025], flame_core=[1.,.83,.17])


class FireBuilding:
    def __init__(self, index=0, seed=None, fire_scale=1.):
        code,title,levels,w,h,description,theme=CATALOG[index]
        self.index=index;self.seed=int(seed if seed is not None else 71100+index)
        self.rng=np.random.default_rng(self.seed);self.id=f'{index+1:02d}_{code}'
        self.title=title;self.description=description;self.theme=theme
        self.levels=levels;self.width=w;self.depth=h;self.storey=3.3
        self.components=[];self.features=[];self.rooms=[];self.fires=[];self.stairs=[]
        self.scene=None;self.textures=None;self.mesh=None
        self.bounds=np.array([[-2.,-3.,-1.2],[w+2.,h+2.,levels*self.storey+1.]])
        self._build()
        self.fire_scale=float(fire_scale)
        if self.fire_scale<1:raise ValueError('fire_scale must be >=1')
        if self.fire_scale>1:self._expand_fire()

    def _expand_fire(self):
        """Increase combustion sources after construction; never change solids/RNG.

        Local source strengths are dimensionless scenario metadata for subsequent
        controlled observation studies, not calibrated heat/smoke concentrations.
        """
        import open3d as o3d
        base=list(self.fires);scene,_=self.render_scene(False)
        self.components=[c for c in self.components if c['kind']!='fire_visual']
        self.features=[f for f in self.features if f['kind']!='fire'];self.fires=[]
        saved_rng=self.rng;self.rng=np.random.default_rng(self.seed+700003)
        for original in base:
            f=original['floor'];z=f*self.storey;px,py,_=original['position']
            rooms=[r for r in self.rooms if r['floor']==f and r['bounds'][1]>self.depth/2]
            def rank(r):
                x0,y0,x1,y1=r['bounds'];inside=x0<=px<=x1 and y0<=py<=y1
                return (not inside,abs(r['target'][0]-px),r['id'])
            rooms=sorted(rooms,key=rank)[:min(3,int(np.ceil(self.fire_scale)))]
            for j,room in enumerate(rooms):
                x0,y0,x1,y1=room['bounds'];x,y=(px,py) if j==0 else room['target'][:2]
                # Attach the source to the existing floor/furniture surface.
                hit=scene.cast_rays(o3d.core.Tensor(np.array([[x,y,z+1.20,0,0,-1]],np.float32)))['t_hit'].numpy()[0]
                base_z=z+1.20-float(hit)+.035 if np.isfinite(hit) else z+.04
                radius=min(original['radius']*np.sqrt(self.fire_scale)*(1 if j==0 else .91),(x1-x0)/2-.08,(y1-y0)/2-.08)
                height=min(original['height']*1.38,z+3.01-base_z)
                fire=dict(id=f'fire_{f}_{j}',floor=f,position=[x,y,base_z],radius=radius,height=height,
                          smoke_height=max(.2,z+3.02-base_z),smoke_opacity=.24,room_bounds=[x0,y0,x1,y1],
                          room_id=room['id'],source_strength_relative=self.fire_scale,smoke_density_relative=2.1,
                          visual_only=True,role='primary' if j==0 else 'adjacent_room_involvement')
                self.fires.append(fire)
                self.features.append(dict(id=f'fire_feature_{f}_{j}',kind='fire',position=fire['position'],floor=f,label='확대 화재' if j==0 else '인접 구역 화재'))
                # Broad irregular lobes rather than one oversized cone.
                for k in range(11):
                    angle=k*2.39996;r=radius*.67*np.sqrt((k+.5)/11)
                    xx=np.clip(x+r*np.cos(angle),x0+.30,x1-.30);yy=np.clip(y+r*np.sin(angle),y0+.30,y1-.30)
                    cr=min(.24+radius*.17,min(xx-x0,x1-xx,yy-y0,y1-yy)*.83)
                    surface=scene.cast_rays(o3d.core.Tensor(np.array([[xx,yy,z+1.20,0,0,-1]],np.float32)))['t_hit'].numpy()[0]
                    lobe_z=z+1.20-float(surface)+.025 if np.isfinite(surface) else base_z
                    hh=min(height*self.rng.uniform(.68,1.0),z+3.01-lobe_z)
                    cone=trimesh.creation.cone(radius=cr,height=hh,sections=7)
                    cone.vertices+=np.array([xx,yy,lobe_z]);cone.vertices[cone.vertices[:,2]>lobe_z+.05,0]+=.09*np.sin(k)
                    self.add('large_flame_visual_only',cone.vertices,f,'fire_visual','flame_core' if k%3==0 else 'flame',solid=False)
        self.rng=saved_rng

    def add(self, name, vertices, floor, kind, material, solid=True):
        vertices=np.asarray(vertices,float)
        hull=ConvexHull(vertices)
        faces=hull.simplices.copy()
        for i,face in enumerate(faces):
            a,b,c=vertices[face]
            if np.dot(np.cross(b-a,c-a),hull.equations[i,:3])<0:faces[i]=face[::-1]
        color=np.clip(np.asarray(PALETTE[material])*self.rng.uniform(.90,1.08),0,1)
        self.components.append(dict(name=name,vertices=vertices,faces=faces,planes=hull.equations,
                                    floor=int(floor),kind=kind,material=material,color=color,solid=solid))

    def box(self,name,center,size,floor,kind,material,rotation=None,solid=True):
        size=np.asarray(size,float)
        if (size<=1e-6).any():return
        vertices=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])*size
        if rotation is not None:vertices=vertices@np.asarray(rotation).T
        self.add(name,vertices+center,floor,kind,material,solid)

    def aabb(self,name,lo,hi,floor,kind,material):
        lo=np.asarray(lo);hi=np.asarray(hi);self.box(name,(lo+hi)/2,hi-lo,floor,kind,material)

    def rod(self,name,a,b,r,floor,kind='detail',material='steel'):
        a=np.asarray(a,float);b=np.asarray(b,float);d=b-a;n=d/np.linalg.norm(d)
        u=np.cross(n,[0,0,1] if abs(n[2])<.9 else [1,0,0]);u/=np.linalg.norm(u);v=np.cross(n,u)
        circle=np.array([u*np.cos(t)+v*np.sin(t) for t in np.linspace(0,2*np.pi,7)[:-1]])*r
        self.add(name,np.concatenate([a+circle,b+circle]),floor,kind,material)

    def feature(self,kind,position,label,floor,**extra):
        self.features.append(dict(id=f'feature_{len(self.features):03d}',kind=kind,position=list(position),label=label,floor=floor,**extra))

    def wall(self,name,axis,fixed,start,end,z,floor,doors=(),windows=(),breach=None,exterior=False):
        """Merge wall raster runs; windows and irregular breach are real voids."""
        thick=.24 if exterior else .17;step=.20
        ss=np.linspace(start,end,max(2,int(np.ceil((end-start)/step))+1));zz=np.linspace(0,3.08,17)
        for k in range(len(zz)-1):
            mid=(zz[k]+zz[k+1])/2;present=np.ones(len(ss)-1,bool)
            for j in range(len(present)):
                s=(ss[j]+ss[j+1])/2
                if any(abs(s-c)<width/2 and mid<2.16 for c,width in doors):present[j]=False
                if any(abs(s-c)<width/2 and .86<mid<2.27 for c,width in windows):present[j]=False
                if breach is not None:
                    c,r=breach;wiggle=.10*np.sin(k*2.7+self.seed)+.065*np.sin(j*2.1)
                    if ((s-c)/(r+wiggle))**2+((mid-1.15)/1.55)**2<1:present[j]=False
            # Runs reduce mesh count while retaining a broken masonry outline.
            for a in np.flatnonzero(present&~np.r_[False,present[:-1]]):
                b=a+1
                while b<len(present) and present[b]:b+=1
                lo=[ss[a],fixed-thick/2,z+zz[k]] if axis=='x' else [fixed-thick/2,ss[a],z+zz[k]]
                hi=[ss[b],fixed+thick/2,z+zz[k+1]] if axis=='x' else [fixed+thick/2,ss[b],z+zz[k+1]]
                self.aabb(f'{name}_{k}_{a}',lo,hi,floor,'exterior' if exterior else 'wall','exterior' if exterior else 'wall')
        for center,width in windows:
            def p(s,d,h):return [s,fixed+d,z+h] if axis=='x' else [fixed+d,s,z+h]
            for side in (-1,1):self.box(name+'_jamb',p(center+side*width/2,0,1.58),[.065,.31,1.48] if axis=='x' else [.31,.065,1.48],floor,'window','frame')
            for height in (.86,2.27):self.box(name+'_sill',p(center,0,height),[width+.14,.34,.075] if axis=='x' else [.34,width+.14,.075],floor,'window','frame')
            self.box(name+'_mullion',p(center,0,1.58),[.045,.10,1.42] if axis=='x' else [.10,.045,1.42],floor,'window','frame')
            # Remaining triangular shards, not an invisible intact window sheet.
            for sign in (-1,1):
                v=np.array([p(center+sign*width*.46,0,.91),p(center+sign*width*.08,0,.91),p(center+sign*width*.46,0,1.26)])
                normal=np.array([0,.018,0]) if axis=='x' else np.array([.018,0,0])
                self.add(name+'_glass_shard',np.r_[v-normal,v+normal],floor,'window','glass')
        if breach:
            pos=[breach[0],fixed,z+.6] if axis=='x' else [fixed,breach[0],z+.6]
            self.feature('wall_collapse',pos,'墙体坍塌',floor)
            self.rubble(pos[:2],z,floor,18,.85)
            for dz in (.55,1.1,1.75):
                p=[breach[0]-.9,fixed,z+dz] if axis=='x' else [fixed,breach[0]-.9,z+dz]
                q=np.array(p)+([.65,.07,.13] if axis=='x' else [.07,.65,.13])
                self.rod('exposed_wall_rebar',p,q,.013,floor)

    def rubble(self,xy,z,floor,count=22,spread=1.):
        for i in range(count):
            angle=self.rng.uniform(0,2*np.pi);r=self.rng.uniform(.15,spread)
            center=np.array([xy[0]+r*np.cos(angle),xy[1]+r*np.sin(angle),z])
            size=self.rng.uniform([.14,.12,.10],[.55,.48,.32])
            vertices=self.rng.normal(size=(9,3));vertices/=np.maximum(np.linalg.norm(vertices,axis=1,keepdims=True),1e-6);vertices*=size
            vertices[:,2]-=vertices[:,2].min();self.add('broken_masonry',vertices+center,floor,'debris','debris')

    def slab(self,floor,holes):
        w,h=self.width,self.depth;z=floor*self.storey
        xs=np.linspace(0,w,int(w/.35)+1);ys=np.linspace(0,h,int(h/.35)+1)
        for j in range(len(ys)-1):
            y=(ys[j]+ys[j+1])/2;centers=(xs[:-1]+xs[1:])/2
            mask=np.ones(len(centers),bool)
            for hole in holes:
                if hole['type']=='rect':
                    x0,y0,x1,y1=hole['bounds'];mask&=~((centers>=x0)&(centers<=x1)&(y>=y0)&(y<=y1))
                else:
                    x0,y0,rx,ry=hole['ellipse'];theta=np.arctan2((y-y0)/ry,(centers-x0)/rx)
                    mask&=(((centers-x0)/rx)**2+((y-y0)/ry)**2>1+.13*np.sin(theta*7+self.seed))
            for a in np.flatnonzero(mask&~np.r_[False,mask[:-1]]):
                b=a+1
                while b<len(mask) and mask[b]:b+=1
                self.aabb('roof_slab' if floor==self.levels else 'floor_slab',[xs[a],ys[j],z-.22],[xs[b],ys[j+1],z],floor,'roof' if floor==self.levels else 'slab','slab')

    def door(self,name,x,y,z,floor,locked=False,width=1.03):
        for dx in (-width/2,width/2):self.box(name+'_frame',[x+dx,y,z+1.08],[.07,.25,2.16],floor,'door','frame')
        self.box(name+'_header',[x,y,z+2.14],[width+.14,.25,.09],floor,'door','frame')
        if locked:
            self.box(name+'_leaf',[x,y,z+1.055],[width-.06,.065,2.11],floor,'door','locked')
            for dz in (.42,1.0,1.62):self.box(name+'_panel',[x,y-.038,z+dz],[width-.24,.018,.30],floor,'door','door')
            self.rod('locked_door_handle',[x+width*.30,y-.08,z+1.0],[x+width*.30-.13,y-.08,z+1.0],.018,floor)
            self.box('lock_plate',[x+width*.30,y-.09,z+.90],[.055,.035,.08],floor,'detail','steel')
            self.feature('locked_door',[x,y,z+1.2],'잠긴 문',floor,state='locked_closed',opening_width_m=width,can_open_during_scene=False)
        else:
            angle=np.deg2rad(self.rng.uniform(68,100));r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            self.box(name+'_open_leaf',[x-width/2+(width-.07)*np.cos(angle)/2,y+(width-.07)*np.sin(angle)/2,z+1.05],[width-.07,.055,2.10],floor,'door','door',r)

    def staircase(self,x,y,floor,damaged=False):
        z=floor*self.storey;rise=self.storey/20;tread=.29;fw=1.25;gap=.20
        for flight in range(2):
            for i in range(10):
                # The secondary stair may lose two treads; the main stair stays intact.
                if damaged and flight==1 and i in (4,5):continue
                sy=y+i*tread if flight==0 else y+(9-i)*tread
                top=z+(i+1+flight*10)*rise
                self.box('stair_tread',[x+flight*(fw+gap)+fw/2,sy+tread/2,top-.075],[fw,tread,.15],floor,'stair','stair')
            a=[x+flight*(fw+gap)+fw/2,y-.05,z+.01 if flight==0 else z+self.storey+.01]
            b=[a[0],y+2.9,z+self.storey/2+.01]
            self.rod('stair_stringer',a,b,.075,floor,'stair','steel')
            for side in (0,fw):
                xx=x+flight*(fw+gap)+side
                for i in (0,3,6,9):
                    yy=y+(i if flight==0 else 9-i)*tread+.13;hh=z+(i+1+10*flight)*rise
                    self.rod('stair_guard_post',[xx,yy,hh],[xx,yy,hh+.92],.018,floor,'stair')
                self.rod('stair_handrail',[xx,y,z+.92 if flight==0 else z+self.storey+.92],[xx,y+2.9,z+self.storey/2+.92],.024,floor,'stair')
        self.box('half_landing',[x+(2*fw+gap)/2,y+3.25,z+self.storey/2-.10],[2*fw+gap,.70,.20],floor,'stair','stair')
        self.box('upper_landing',[x+fw+gap+fw/2,y-.29,z+self.storey-.10],[fw,.58,.20],floor,'stair','stair')
        self.feature('stairs',[x+fw,y+1.4,z+1.0],'파손 계단' if damaged else '층간 계단',floor,upper_floor=floor+1,rise_m=rise,tread_m=tread,width_m=fw,missing_treads=2 if damaged else 0)
        self.stairs.append(dict(floor=floor,x=x,y=y,damaged=damaged,rise_m=rise,tread_m=tread))

    def furniture(self,room):
        x0,y0,x1,y1=room['bounds'];f=room['floor'];z=f*self.storey
        x=(x0+x1)/2;y=(y0+y1)/2
        def b(n,c,s,mat='furniture'):self.box(n,[c[0],c[1],z+c[2]],s,f,'furniture',mat)
        home_mode=int(room['id'].split('_')[-1])%3
        if self.theme=='home' and home_mode==0:
            b('sofa_base',[x,y,.32],[2.0,.85,.46],'fabric');b('sofa_back',[x,y+.37,.74],[2.,.18,.60],'fabric')
            for dx in (-.95,.95):b('sofa_arm',[x+dx,y,.60],[.18,.90,.42],'fabric')
            b('coffee_table',[x,y-1.10,.30],[1.10,.60,.10])
            for dx in (-.4,.4):b('coffee_table_leg',[x+dx,y-1.10,.15],[.06,.44,.30],'steel')
            b('television',[x,y0+.45,1.0],[1.25,.08,.72],'frame');b('tv_cabinet',[x,y0+.45,.34],[1.7,.48,.68])
        elif self.theme=='home' and home_mode==1:
            b('kitchen_cabinets',[x,y1-.40,.45],[max(1.,x1-x0-.6),.65,.90],'tile')
            b('kitchen_worktop',[x,y1-.40,.94],[max(1.,x1-x0-.5),.72,.07],'slab')
            b('hob',[x-.5,y1-.40,.985],[.55,.50,.025],'frame')
            b('sink',[x+.55,y1-.40,.985],[.55,.45,.035],'steel');self.table(x,y-.60,z,f,1.25,.75)
        elif self.theme in ('home','hotel'):
            b('bed_frame',[x,y,.26],[1.55,2.05,.40]);b('mattress',[x,y,.52],[1.5,2.,.22],'fabric');b('pillow',[x,y+.65,.67],[.80,.42,.12],'tile')
            b('wardrobe',[x0+.45,y1-.5,1.02],[.65,.90,2.04]);b('side_table',[x+1.15,y+.45,.32],[.48,.55,.64])
        elif self.theme=='school':
            for dx in (-.85,.85):
                for dy in (-1.0,.8):self.table(x+dx,y+dy,z,f,.90,.60)
            b('blackboard',[x,y1-.14,1.55],[max(.8,x1-x0-1),.07,1.0],'fabric')
        elif self.theme=='clinic':
            b('examination_bed',[x,y,.62],[.8,1.95,.18],'tile')
            for dx in (-.28,.28):b('bed_support',[x+dx,y,.31],[.09,1.3,.62],'steel')
            b('medical_cabinet',[x0+.5,y1-.4,.85],[.8,.60,1.7],'tile')
        elif self.theme in ('retail','library'):
            for dx in (-.9,.9):
                b('shelf_left',[x+dx-.44,y,.95],[.07,2.0,1.9]);b('shelf_right',[x+dx+.44,y,.95],[.07,2.0,1.9])
                for hh in (.15,.65,1.15,1.65):
                    b('shelf',[x+dx,y,hh],[.95,2.,.07])
                    for row in (-.65,0,.65):b('books_or_stock',[x+dx,y+row,hh+.20],[.7,.40,.32],'fabric')
        else:
            self.table(x,y,z,f,1.8,.85)
            b('cabinet',[x0+.5,y1-.45,.8],[.65,.85,1.6],'steel')
            if self.theme=='office':
                b('monitor',[x,y+.15,1.04],[.65,.07,.4],'frame');b('monitor_stem',[x,y+.15,.82],[.05,.06,.15],'steel')

    def table(self,x,y,z,f,w,d):
        self.box('table_top',[x,y,z+.76],[w,d,.07],f,'furniture','furniture')
        for dx in (-w*.40,w*.40):
            for dy in (-d*.37,d*.37):self.box('table_leg',[x+dx,y+dy,z+.36],[.055,.055,.72],f,'furniture','steel')
        self.box('chair_seat',[x,y-.8,z+.43],[.46,.46,.07],f,'furniture','fabric')
        self.box('chair_back',[x,y-1.02,z+.72],[.46,.07,.60],f,'furniture','fabric')
        for dx in (-.17,.17):self.box('chair_legs',[x+dx,y-.8,z+.21],[.035,.35,.42],f,'furniture','steel')

    def _build(self):
        w,h=self.width,self.depth;c=h/2;corr=1.3 if self.theme!='retail' else 1.65
        self.corridor=(c-corr,c+corr)
        room_count=4 if self.theme in ('hotel','office') else 3
        cuts=np.r_[.30,np.linspace(4.25,w-4.25,room_count+1),w-.30]
        cuts=np.unique(np.round(cuts,3));self.room_cuts=cuts.tolist()
        # Rectangular stair wells are genuine slab openings on upper levels.
        wells=[dict(type='rect',bounds=[.85,c+corr+.03,3.70,c+corr+3.92]),dict(type='rect',bounds=[w-3.70,c+corr+.03,w-.85,c+corr+3.92])]
        self.holes={}
        for f in range(self.levels+1):
            holes=[] if f==0 else [dict(a) for a in wells]
            hx=cuts[2]+(cuts[3]-cuts[2])*.72;hy=(c-corr)*.50
            holes.append(dict(type='ellipse',ellipse=[hx+(f%2)*1.2,hy,1.1+.12*(self.index%3),.85+.1*(self.index%2)]))
            if f==self.levels:
                holes.append(dict(type='ellipse',ellipse=[w*.72,c+corr+2.7,2.0,1.25]))
            self.holes[f]=holes;self.slab(f,holes)
            if f<self.levels:
                hole=holes[-1]['ellipse'];self.feature('floor_hole',[hole[0],hole[1],f*self.storey+.02],'바닥 붕괴',f,ellipse=hole,depth_to_lower_level_m=self.storey if f else 1.0)
                if f==0:self.box('pit_bottom',[hole[0],hole[1],-1.0],[hole[2]*2.5,hole[3]*2.5,.18],0,'foundation','slab')
            if f>0:
                hole=holes[-1]['ellipse'];z=(f-1)*self.storey
                self.rubble(hole[:2],z,f-1,28,1.5)
                for q in range(3):
                    angle=self.rng.uniform(-.5,.5);r=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
                    self.box('fallen_ceiling_panel',[hole[0]+(q-1)*.65,hole[1],z+.45],[1.25,.85,.14],f-1,'debris','slab',r)
                self.feature('ceiling_collapse',[hole[0],hole[1],z+.65],'천장 붕괴·잔해',f-1)
        for f in range(self.levels):
            z=f*self.storey
            # Exterior windows, real entrance opening, and one torn exterior bay.
            window_x=[(a+b)/2 for a,b in zip(cuts[:-1],cuts[1:])]
            self.wall('south_facade','x',0,0,w,z,f,doors=[(w/2,1.85)] if f==0 else (),windows=[(x,1.45) for x in window_x if f or abs(x-w/2)>1.7],breach=(w*.24,1.25) if f==(self.index+1)%self.levels else None,exterior=True)
            self.wall('north_facade','x',h,0,w,z,f,windows=[(x,1.5) for x in window_x],breach=(w*.72,1.3) if f==self.index%self.levels else None,exterior=True)
            for xx in (0,w):self.wall('side_facade','y',xx,0,h,z,f,windows=[(h*.25,1.4),(h*.75,1.4)],exterior=True)
            # Concrete columns and beams express the surviving structure.
            for xx in (0,w,4.1,w-4.1):
                for yy in (0,h,c-corr,c+corr):self.box('concrete_column',[xx,yy,z+1.54],[.32,.32,3.08],f,'column','slab')
            for yy in (c-corr,c+corr):self.box('ceiling_beam',[w/2,yy,z+2.91],[w,.28,.30],f,'beam','slab')
            for side,(y0,y1) in enumerate(((.25,c-corr),(c+corr,h-.25))):
                for i,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
                    is_stair=side==1 and i in (0,len(cuts)-2)
                    if not is_stair:
                        doorx=(a+b)/2;boundary=y1 if side==0 else y0
                        breach=(a+(b-a)*.7,.85) if (i==2 and side==1 and f==(self.index+1)%self.levels) else None
                        self.wall('corridor_wall','x',boundary,a,b,z,f,doors=[(doorx,1.03)],breach=breach)
                        locked=i==1 and side==1 or i==len(cuts)-3 and side==0 and f==self.levels-1
                        self.door('room_door',doorx,boundary,z,f,locked)
                        room=dict(id=f'L{f+1}_{side}_{i}',floor=f,bounds=[a+.15,y0+.18,b-.15,y1-.18],target=[(a+b)/2,(y0+y1)/2,z+.04])
                        self.rooms.append(room);self.furniture(room)
                    open_bay=self.theme in ('retail','workshop','library') and i%2==0
                    if i>0 and not open_bay:self.wall('room_partition','y',a,y0,y1,z,f,breach=((y0+y1)/2,1.0) if i==2 and side==0 and f==0 else None)
            if f<self.levels-1:
                self.staircase(1.0,c+corr+.30,f)
                self.staircase(w-3.70,c+corr+.30,f,damaged=(self.index in (3,6,8) and f==0))
            # Lighting housings, services, hanging ceiling material and soot.
            for xx in np.arange(3,w-1,4):
                self.box('corridor_luminaire',[xx,c,z+2.97],[1.1,.18,.055],f,'detail','tile')
            self.rod('service_pipe',[.5,c+.65,z+2.82],[w-.5,c+.65,z+2.82],.042,f,'detail','pipe')
            rx=cuts[2]+.65;ry=c+corr+(h-c-corr)*.52
            fire=dict(id=f'fire_{f}',floor=f,position=[rx,ry,z+.55],radius=.65+.10*(self.index%3),height=1.8+.10*f,smoke_height=2.85,visual_only=True)
            self.fires.append(fire);self.feature('fire',fire['position'],'실내 화재',f)
            self.box('burnt_furniture',[rx,ry,z+.3],[1.3,.85,.60],f,'furniture','char')
            for j in range(4):
                height=fire['height']*(.65+.1*j)
                cone=trimesh.creation.cone(radius=.23+.035*j,height=height,sections=7)
                cone.vertices[:,:2]+=[rx+(j-1.5)*.22,ry+.12*np.sin(j*2)]
                cone.vertices[:,2]+=z+.57
                cone.vertices[cone.vertices[:,2]>z+.58,0]+=.13*np.sin(j+1)
                self.add('flame_visual_only',cone.vertices,f,'fire_visual','flame_core' if j==2 else 'flame',solid=False)
            # Charred strips lie against actual corridor wall, not floating decals.
            for q in range(7):
                height=self.rng.uniform(.6,2.7)
                self.box('soot_on_wall',[rx+(q-3)*.18,c+corr+.091,z+height/2],[.19,.014,height],f,'detail','char')
            self.box('fallen_low_beam',[w*.71,c,z+1.42],[1.4,1.65,.22],f,'debris','slab',trimesh.transformations.euler_matrix(.06,.15,.12)[:3,:3])
            for dx in (-.55,.55):self.rod('hanging_beam_rebar',[w*.71+dx,c-.6,z+1.5],[w*.71+dx-.12,c-.65,z+3.05],.017,f,'detail')
            self.feature('low_clearance',[w*.71,c,z+1.4],'낮게 처진 보',f,approx_clearance_m=1.2)
        # Entrance apron, steps and four staging positions.
        self.aabb('south_sidewalk',[-1.2,-1.2,-.20],[w+1.2,0,0],0,'foundation','slab')
        self.aabb('north_sidewalk',[-1.2,h,-.20],[w+1.2,h+1.2,0],0,'foundation','slab')
        self.aabb('west_sidewalk',[-1.2,0,-.20],[0,h,0],0,'foundation','slab')
        self.aabb('east_sidewalk',[w,0,-.20],[w+1.2,h,0],0,'foundation','slab')
        self.box('entry_apron',[w/2,-1.25,-.10],[5.,2.5,.20],0,'foundation','slab')
        self.starts=np.array([[w/2-1.2+i*.8,-1.4,1.15] for i in range(4)])
        self.feature('entry',[w/2,-.5,.12],'출입구',0)
        self.entry=dict(xy_bounds=[w/2-.3,.35,w/2+.3,.95],floor_z_bounds=[-.05,.20])
        self.goals=[dict(id=r['id'],xy_bounds=[r['target'][0]-.25,r['target'][1]-.25,r['target'][0]+.25,r['target'][1]+.25],floor_z_bounds=[r['floor']*self.storey-.1,r['floor']*self.storey+.25]) for r in self.rooms]
        for a in self.features:
            if a['label']=='墙体坍塌':a['label']='벽 붕괴'

    def prior(self):
        return dict(bounds=self.bounds.tolist(),exploration_box=[[.2,.2,.05],[self.width-.2,self.depth-.2,self.levels*self.storey-.2]],entry=self.entry,goals=self.goals,
                    launch_free_box=[[self.width/2-1.8,-2.1,.65],[self.width/2+1.8,-.4,1.7]],
                    assumption='shared mission target regions; internal geometry is not planner input; reachability unlabelled')

    def render_scene(self,textured=False):
        if textured:raise ValueError('Multistorey suite currently supplies colored geometry and direct-depth raycasting, not legacy stereo textures')
        if self.scene is None:
            import open3d as o3d
            self.scene=o3d.t.geometry.RaycastingScene()
            vertices=[];faces=[];offset=0
            for c in self.components:
                if not c['solid']:continue
                vertices.extend(c['vertices']);faces.extend(c['faces']+offset);offset+=len(c['vertices'])
            self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices,np.float32)),o3d.core.Tensor(np.asarray(faces,np.uint32)))
        return self.scene,None

    def labels(self,res=.15):
        """Conservative convex half-space voxelization, not route ground truth."""
        ijk=np.floor(self.bounds[0]/res).astype(int);shape=np.ceil(self.bounds[1]/res).astype(int)-ijk
        labels=np.ones(shape,np.uint8)
        for c in self.components:
            if not c['solid']:continue
            vertices=c['vertices'];a=np.maximum(np.floor(vertices.min(0)/res).astype(int)-ijk,0);b=np.minimum(np.ceil(vertices.max(0)/res).astype(int)-ijk,shape)
            if (b<=a).any():continue
            q=np.indices(b-a).reshape(3,-1).T+a;points=(q+ijk+.5)*res;planes=c['planes']
            mask=np.ones(len(q),bool)
            for normal in planes:
                mask&=points@normal[:3]+normal[3]<=res*.5*abs(normal[:3]).sum()+1e-7
            ids=q[mask];labels[tuple(ids.T)]=2
        return labels,ijk

    def metadata(self):
        return dict(id=self.id,title=self.title,description=self.description,seed=self.seed,units='metres',up_axis='Z',levels=self.levels,storey_height_m=self.storey,
                    fire_scale=self.fire_scale,fire_scenario='multi-room involvement' if self.fire_scale>1 else 'localized furniture fire',
                    footprint_m=[self.width,self.depth],bounds=self.bounds.tolist(),rooms=self.rooms,stairs=self.stairs,fires=self.fires,features=self.features,
                    slab_openings=self.holes,prior=self.prior(),human_model=dict(stand_height_m=1.9,stoop_height_m=1.4,crawl_height_m=.9,width_m=.7,sideways_width_m=.45,step_m=.18),
                    reference_reachability='not yet certified; no target labels asserted',fire_model='visual/source annotations only, no thermal or structural dynamics',
                    voxelization='0=unknown (unused GT),1=air,2=solid; conservative cell/convex-plane test; mesh is authoritative',
                    glass_model='remaining shards are opaque collision surfaces; window openings contain no invisible full sheet',components=len(self.components))

    def export(self,out,res=.15):
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        scene=trimesh.Scene();meshes=[]
        for i,c in enumerate(self.components):
            mesh=trimesh.Trimesh(c['vertices'],c['faces'],process=False)
            mesh.visual.face_colors=np.r_[np.round(c['color']*255),255].astype(np.uint8)
            scene.add_geometry(mesh,node_name=f'{i:04d}_{c["kind"]}_L{c["floor"]+1}_{c["name"]}');meshes.append(mesh)
        self.mesh=trimesh.util.concatenate([mesh for mesh,c in zip(meshes,self.components) if c['solid']])
        (out/'scene.glb').write_bytes(scene.export(file_type='glb'));self.mesh.export(out/'scene.ply')
        meta=self.metadata();(out/'scene.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        lab,ijk=self.labels(res);np.savez_compressed(out/'gt_voxel.npz',labels=lab,ijk=ijk,res=res)
        geometry=[dict(name=c['name'],floor=c['floor'],kind=c['kind'],material=c['material'],solid=c['solid'],vertices=c['vertices'].round(6).tolist(),faces=c['faces'].tolist()) for c in self.components]
        (out/'collision_geometry.json').write_text(json.dumps(geometry,separators=(',',':'))+'\n')
        return meta
