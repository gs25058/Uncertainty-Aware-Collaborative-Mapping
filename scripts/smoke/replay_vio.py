#!/usr/bin/env python3
"""Run under the existing covins ROS environment; writes only --input outputs."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import re
import subprocess
import time
import xmlrpc.client

import cv2
import rospy
from sensor_msgs.msg import Image,Imu


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--rate",type=float,default=.5,help="replay speed relative to real time")
    ap.add_argument("--port",type=int,default=11329)
    ap.add_argument("--run-id",default=None,help="isolated repeat folder; preserves first run")
    ap.add_argument("--order",default="clean,uniform_strong")
    args=ap.parse_args()
    root=Path(args.input).resolve()
    design=json.loads((root/"design.json").read_text())
    runroot=root/"repeats"/args.run_id if args.run_id else root
    runroot.mkdir(parents=True,exist_ok=True)
    robot=design["args"]["robot"]
    frames=list(csv.DictReader(open(root/"frames.csv")))
    data=Path(design["data"])
    shifts={k.strip():int(v) for k,v in (line.split(":") for line in (data/"timeshift.yaml").read_text().splitlines())}
    epoch=shifts["timeshift_s"]+shifts["timeshift_ns"]/1e9
    imu=list(csv.DictReader(open(data/robot/"imu_px4.csv")))
    imu=[r for r in imu if float(r["timestamp"])<=float(frames[-1]["timestamp"])+.1]
    env=os.environ.copy()
    env.update(ROS_MASTER_URI=f"http://127.0.0.1:{args.port}",ROS_HOSTNAME="127.0.0.1",ROS_IP="127.0.0.1",
               ROS_HOME=str(runroot/"ros"),ROS_LOG_DIR=str(runroot/"ros"),
               OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    os.environ.update(env)
    (runroot/"ros").mkdir(exist_ok=True)
    corelog=open(runroot/"roscore.log","w")
    core=subprocess.Popen(["roscore","-p",str(args.port)],stdout=corelog,stderr=subprocess.STDOUT,
                           env=env,start_new_session=True)
    node=None
    status=[]
    try:
        for _ in range(100):
            if core.poll() is not None:
                raise RuntimeError("roscore failed; see roscore.log")
            try:
                xmlrpc.client.ServerProxy(env["ROS_MASTER_URI"]).getPid("smoke_replay")
                break
            except OSError:
                time.sleep(.1)
        else:raise RuntimeError("ROS master did not become ready")
        rospy.init_node("controlled_smoke_replay",anonymous=True,disable_signals=True)
        pubs=[rospy.Publisher(f"/{robot}/camera/infra{j+1}/image_raw",Image,queue_size=100) for j in range(2)]
        ipub=rospy.Publisher(f"/{robot}/mavros/imu/data_raw",Imu,queue_size=2000)
        for name in args.order.split(","):
            if name not in design["configs"]:raise ValueError("unknown replay condition")
            d=runroot/name
            if args.run_id:
                d.mkdir(parents=True,exist_ok=True)
                for cam in ("cam0.yaml","cam1.yaml"):
                    (d/cam).write_bytes((root/name/cam).read_bytes())
                cfg=(root/name/"vins.yaml").read_text()
                cfg=re.sub(r'^output_path:.*',f'output_path: "{d}"',cfg,flags=re.M)
                (d/"vins.yaml").write_text(cfg)
            log=open(d/"vins.log","w")
            node=subprocess.Popen([os.environ.get("VINS_NODE","vins_node"),str(d/"vins.yaml")],
                                  stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            for _ in range(100):
                if node.poll() is not None:
                    raise RuntimeError(f"VINS failed for {name}; see vins.log")
                if ipub.get_num_connections() and all(p.get_num_connections() for p in pubs):break
                time.sleep(.1)
            else:raise RuntimeError("VINS did not subscribe to images/IMU")
            t0=time.monotonic();it=0
            for frame in frames:
                t=float(frame["timestamp"])
                wait=t/args.rate-(time.monotonic()-t0)
                if wait>0:time.sleep(wait)
                while it<len(imu) and float(imu[it]["timestamp"])<=t+.04:
                    row=imu[it];msg=Imu();msg.header.stamp=rospy.Time.from_sec(epoch+float(row["timestamp"]))
                    for attr,prefix in ((msg.angular_velocity,"angular_velocity"),(msg.linear_acceleration,"linear_acceleration")):
                        for axis in "xyz":setattr(attr,axis,float(row[f"{prefix}.{axis}"]))
                    ipub.publish(msg);it+=1
                for j,pub in enumerate(pubs):
                    img=cv2.imread(str(root/name/f"image{j}"/f'{int(frame["index"]):06d}.png'),0)
                    msg=Image();msg.header.stamp=rospy.Time.from_sec(epoch+t)
                    msg.height,msg.width=img.shape;msg.encoding="mono8";msg.step=img.shape[1]
                    msg.data=img.tobytes();pub.publish(msg)
            for row in imu[it:]:
                msg=Imu();msg.header.stamp=rospy.Time.from_sec(epoch+float(row["timestamp"]))
                for attr,prefix in ((msg.angular_velocity,"angular_velocity"),(msg.linear_acceleration,"linear_acceleration")):
                    for axis in "xyz":setattr(attr,axis,float(row[f"{prefix}.{axis}"]))
                ipub.publish(msg)
            # Wait for estimator queue to settle, capped at 40 seconds.
            last=-1;stable=0
            for _ in range(40):
                time.sleep(1)
                p=d/"vio.csv";size=p.stat().st_size if p.exists() else 0
                stable=stable+1 if size==last else 0;last=size
                if stable>=5:break
            os.killpg(node.pid,signal.SIGINT);node.wait(timeout=10);node=None
            log.close()
            p=d/"vio.csv";n=sum(1 for _ in open(p)) if p.exists() else 0
            status.append(dict(condition=name,run_id=args.run_id or "first",n_poses=n,n_stereo_pairs=len(frames),
                               elapsed_s=time.monotonic()-t0,replay_rate=args.rate))
            (runroot/"run.json").write_text(json.dumps(status,indent=2)+"\n")
            print(status[-1],flush=True)
            time.sleep(2)
    finally:
        for p in (node,core):
            if p is not None and p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
        corelog.close()


if __name__=="__main__":main()
