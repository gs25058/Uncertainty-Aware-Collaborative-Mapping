#!/usr/bin/env python3
"""Prepare an isolated short raw-IR replay for VINS-Fusion (clean and uniform)."""
import argparse
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from covor.data import MILUV_ROOT
from covor.occupancy import load_stereo_calib,StereoDepth
from covor.synth.render import _depth_rectified
from covor.smoke import SmokeCfg,stereo_smoke,clean_right_depth
from common import write_json,write_csv,digest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--duration",type=float,default=20)
    ap.add_argument("--image-stride",type=int,default=2)
    ap.add_argument("--beta",type=float,default=.8)
    ap.add_argument("--robot",default="ifo001")
    args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    data=Path(MILUV_ROOT)/"data"/"default_3_zigzag_0"
    source=data/args.robot
    calib=load_stereo_calib(args.robot);sd=StereoDepth(calib)
    cv2.setNumThreads(1)
    # For VIO keep raw geometry and original camera calibration. Reproject ONLY
    # the synthetic transmission to raw pixel coordinates; never warp raw I.
    R1,R2,P1,P2,*_=cv2.stereoRectify(calib.K_l,calib.D_l.reshape(1,-1),
                calib.K_r,calib.D_r.reshape(1,-1),calib.size,
                np.ascontiguousarray(calib.R),calib.T.reshape(3,1),
                flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    v,u=np.indices((calib.size[1],calib.size[0]),dtype=np.float32)
    pixels=np.stack([u,v],axis=-1).reshape(-1,1,2)
    warps=[]
    for K,D,R,P in ((calib.K_l,calib.D_l,R1,P1),(calib.K_r,calib.D_r,R2,P2)):
        q=cv2.undistortPoints(pixels,K,D,R=R,P=P).reshape(*u.shape,2)
        warps.append((q[:,:,0],q[:,:,1]))
    common=set(p.name for p in (source/"infra1").glob("*.jpeg"))&set(p.name for p in (source/"infra2").glob("*.jpeg"))
    files=sorted(common,key=lambda f:float(f[:-5]))
    files=[f for f in files if float(f[:-5])<=args.duration][::args.image_stride]
    configs={"clean":SmokeCfg(),"uniform_strong":SmokeCfg(beta=args.beta)}
    cfgsrc=Path(MILUV_ROOT)/"config"/"vins"/args.robot
    for name in configs:
        d=out/name;d.mkdir(exist_ok=True)
        (d/"image0").mkdir(exist_ok=True);(d/"image1").mkdir(exist_ok=True)
        for cam in ("cam0.yaml","cam1.yaml"):
            (d/cam).write_bytes((cfgsrc/cam).read_bytes())
        text=(cfgsrc/"vins.yaml").read_text()
        text=re.sub(r'^output_path:.*',f'output_path: "{d}"',text,flags=re.M)
        text=re.sub(r'^pose_graph_save_path:.*',f'pose_graph_save_path: "{d}"',text,flags=re.M)
        text=re.sub(r'^show_track:.*','show_track: 0',text,flags=re.M)
        (d/"vins.yaml").write_text(text)
    rows=[]
    for i,f in enumerate(files):
        t=float(f[:-5])
        raw=[cv2.imread(str(source/c/f),0) for c in ("infra1","infra2")]
        left,right=sd.rectify_pair(*raw)
        zl,_,_=_depth_rectified(sd,left,right)
        zr=clean_right_depth(sd,left,right)
        for name,cfg in configs.items():
            _,_,meta=stereo_smoke(sd,left,right,zl,zr,np.eye(4),t,cfg)
            from covor.smoke import attenuate,ray_directions,rectified_camera_poses
            dirs=ray_directions(sd,left.shape)
            Ts=rectified_camera_poses(sd,np.eye(4))
            for cam,(img,z,T,(mx,my)) in enumerate(zip(raw,(zl,zr),Ts,warps)):
                if cfg.beta==0:
                    fog=img.copy()
                else:
                    _,trans,_=attenuate((left,right)[cam],z,dirs,T,t,cfg)
                    traw=cv2.remap(trans,mx,my,cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=float(np.exp(-cfg.beta*cfg.fallback_z)))
                    fog=np.rint(np.clip(traw*img+(1-traw)*255*cfg.airlight,0,255)).astype(np.uint8)
                dest=out/name/f"image{cam}"/f"{i:06d}.png"
                cv2.imwrite(str(dest),fog)
                if cfg.beta==0:
                    assert np.array_equal(cv2.imread(str(dest),0),img)
        rows.append(dict(index=i,timestamp=t,source=f,
                         left_sha256=digest(source/"infra1"/f),right_sha256=digest(source/"infra2"/f)))
        if (i+1)%50==0:
            print(i+1,"VIO stereo pairs prepared",flush=True)
    write_csv(out/"frames.csv",rows)
    write_json(out/"design.json",dict(args=vars(args),frames=len(rows),
        configs={k:v.metadata() for k,v in configs.items()},data=str(data),
        raw_geometry=True,clean_bit_exact=True,rectified_transmission_inverse_warp=True))


if __name__=="__main__":main()
