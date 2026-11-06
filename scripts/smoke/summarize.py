#!/usr/bin/env python3
"""Publish measured CSVs/figures and a compact report without rerunning inference."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import write_csv,write_json,digest


def read(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def table(rows,keys,labels=None):
    def fmt(v):
        try:
            f=float(v)
            return str(int(f)) if f.is_integer() else f"{f:.4f}"
        except (ValueError,TypeError):return str(v)
    lines=["| "+" | ".join(labels or keys)+" |","|"+"---|"*len(keys)]
    lines.extend("| "+" | ".join(fmt(r.get(k,"")) for k in keys)+" |" for r in rows)
    return "\n".join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    src=Path(args.input).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    for sub in ("miluv","miluv_baseline","miluv_seedcheck","corridor","corridor_baseline","vio","reference_examples"):
        p=src/sub
        if not p.exists():continue
        dest=out/sub;dest.mkdir(exist_ok=True)
        for name in ("summary.csv","frames.csv","goals.csv","candidate_scores.csv","manifest.json",
                     "baseline_check.json","design.json","run.json","evaluation.json","download_report.json","sources.json"):
            if (p/name).exists():shutil.copy2(p/name,dest/name)
        for pimg in p.glob("*.png"):
            shutil.copy2(pimg,dest/pimg.name)
        if (p/"examples").exists():
            (dest/"examples").mkdir(exist_ok=True)
            for pimg in (p/"examples").glob("*.png"):
                # Include comparison panels; raw stereo remains in results/.
                if not any(x in pimg.stem for x in ("_left","_right")):
                    shutil.copy2(pimg,dest/"examples"/pimg.name)
    miluv=read(src/"miluv"/"summary.csv")
    corridor=read(src/"corridor"/"summary.csv")
    if (out/"corridor"/"summary_conservative.csv").exists():
        corridor=read(out/"corridor"/"summary_conservative.csv")
    frames=read(src/"corridor"/"frames.csv")
    sensor=[]
    for condition in dict.fromkeys(r["condition"] for r in frames):
        rr=[r for r in frames if r["condition"]==condition and r["stage"]=="initial"]
        weights=np.array([float(r["n_joint"]) for r in rr])
        sensor.append(dict(condition=condition,n_frames=len(rr),
          valid_fraction=np.mean([float(r["valid_fraction"]) for r in rr]),
          joint_fraction=np.mean([float(r["joint_fraction"]) for r in rr]),
          depth_absdiff_mean_m=np.average([np.nan_to_num(float(r["depth_absdiff_mean_m"])) for r in rr],weights=np.maximum(weights,1)),
          fill_left=np.mean([float(r["fill_left"]) for r in rr]),
          fill_right=np.mean([float(r["fill_right"]) for r in rr])))
    write_csv(out/"corridor"/"sensor_summary.csv",sensor)
    seedcheck=read(src/"miluv_seedcheck"/"summary.csv")
    vio=read(src/"vio"/"summary.csv")
    for repeat in sorted((src/"vio"/"repeats").glob("*")):
        if not repeat.is_dir():continue
        vio.extend(read(repeat/"summary.csv"))
        dest=out/"vio"/"repeats"/repeat.name;dest.mkdir(parents=True,exist_ok=True)
        for f in repeat.iterdir():
            if f.is_file() and f.suffix in (".csv",".json",".png"):
                shutil.copy2(f,dest/f.name)
    write_csv(out/"vio"/"all_runs.csv",vio)
    if not miluv or len(corridor)<28:
        raise RuntimeError("primary experiments are incomplete; report refused")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for field in ("uniform","heterogeneous"):
        rows=[miluv[0]]+[r for r in miluv if r["field"]==field and r["condition"]!="clean"]
        x=[float(r["beta"]) for r in rows]
        for a,k in zip(ax,("valid_fraction","depth_absdiff_mean_m","occupied_iou_clean")):
            a.plot(x,[float(r[k]) for r in rows],"o-",label=field)
    for a,label in zip(ax,("Valid SGBM depth fraction","Mean |Z - clean Z| [m]","Occupied IoU with clean map")):
        a.set_xlabel("attenuation setting beta [1/m]");a.set_ylabel(label);a.grid(alpha=.3);a.legend()
    fig.tight_layout();fig.savefig(out/"miluv_changes.png",dpi=150);plt.close(fig)
    # Same world bounds and camera angle for all occupied-voxel 3D panels.
    for dataset in ("miluv","corridor"):
        fig=plt.figure(figsize=(14,5))
        for i,name in enumerate(("clean","heterogeneous_medium","heterogeneous_strong")):
            suffix="_initial" if dataset=="corridor" else ""
            q=np.load(src/dataset/f"map_{name}{suffix}.npz")
            lab=q["labels"];lo=q["ijk_min"]
            p=(np.argwhere(lab==2)+lo+.5)*.1
            p=p[(p[:,2]>=-.1)&(p[:,2]<=2.2)]
            p=p[::max(1,len(p)//6000)]
            a=fig.add_subplot(1,3,i+1,projection="3d")
            a.scatter(p[:,0],p[:,1],p[:,2],c=p[:,2],s=1,cmap="viridis",vmin=-.1,vmax=2.2)
            a.set_xlim(lo[0]*.1,(lo[0]+lab.shape[0])*.1)
            a.set_ylim(lo[1]*.1,(lo[1]+lab.shape[1])*.1);a.set_zlim(-.1,2.2)
            a.set_box_aspect((lab.shape[0]*.1,lab.shape[1]*.1,2.3))
            a.view_init(elev=45,azim=-65);a.set_title(name)
            a.set_xlabel("x [m]");a.set_ylabel("y [m]");a.set_zlabel("z [m]",labelpad=1)
            a.set_zticks([0,1,2]);a.tick_params(axis="z",labelsize=7)
        fig.tight_layout();fig.savefig(out/dataset/"occupancy_3d_comparison.png",dpi=150);plt.close(fig)
    ref=src/"reference_examples"
    paths=[ref/"real_0.jpg",ref/"synthetic_0.png",src/"miluv"/"examples"/"heterogeneous_strong_smoke_left.png"]
    if all(p.exists() for p in paths):
        fig,ax=plt.subplots(1,3,figsize=(15,4))
        for a,p,title in zip(ax,paths,("SmokeSeer real.zip RGB example","SmokeSeer synthetic.zip example","Our MILUV IR approximation")):
            a.imshow(Image.open(p),cmap="gray");a.set_title(title);a.axis("off")
        fig.tight_layout();fig.savefig(out/"reference_visual_check.png",dpi=130);plt.close(fig)
    text=f'''# 모의 연기와 단일 재관측 — 실제 실행 결과

기준 커밋 `dffbb8ed68d8c298bdb6c08792245239784be23a`, `paper/conference-2026`.
실험 ID `smoke_20261103`. 원본 데이터·기존 결과·기존 SLAM/지도/판정 구현은 변경하지 않았다.
새 코드는 `covor/smoke.py`, `covor/reobserve.py`, `scripts/smoke/`다.

확인한 실제 입력 경로:

- IR 원본: `/src/gs25058/cr_RNE/miluv/data/default_3_zigzag_0/ifo00{{1,2,3}}/infra{{1,2}}/*.jpeg`
- 원시 VIO: `/src/gs25058/ws/vins_ws/run/out/default_3_zigzag_0/<robot>/vio.csv`
- 고정 융합 포즈: `vo_output/occ_default_3_zigzag_0_<robot>_af.npz`
- 복도 메시: `meshes/2026-09-15/2026. 9. 15.floorfix.obj`와 같은 폴더의 MTL·텍스처
- 복도 GT·원시 합성 관측/포즈: `results/synth_corridor915f/gt_voxel_fs.npz`,
  `data_seed0/`, `vins_seed0/`, `gt_traj.npz`

MILUV 실제 사용 시간은 드론별 약 5.87–106.31 / 6.39–106.73 / 6.82–107.17초다.
전체 시퀀스의 모든 프레임을 사용한 실험은 아니다.

핵심 결과: MILUV 강한 비균일 감쇠에서 유효 깊이 비율은 69.84%→66.24%,
공동 유효 픽셀의 평균 깊이 차이는 0.227 m, occupied exact IoU는 0.258이었다.
복도 중간 균일 감쇠에서 경로 중심 한 쌍은 통과 확인 목표를 3→4개로 늘렸고
가까운 후보·미관측량 중심 방식은 변화가 없었다. 중간 비균일 감쇠에서는 경로 중심 방식이
도달률을 9.07%→10.43%로 늘렸지만 통과 확인 목표는 4개로 같았다.
7조건 중 경로 중심 방식의 통과 확인 수 증가는 1조건, 감소는 1조건, 나머지는 변화가 없었다.
따라서 이번 기준선은 조건부 개선을 보였으며 일반적 우월성을 입증하지 않는다.

## 1. MILUV에서 측정한 깊이·지도 변화

`default_3_zigzag_0`, 드론 3대 각 48쌍, 융합 포즈 인덱스 0부터 stride 16.
실제 이미지 이름·시간·SHA256과 포즈 파일 SHA256은 `miluv/manifest.json`, 프레임별 수치는 `miluv/frames.csv`.
포즈는 기존 anchor-free 융합 출력으로 고정했다. 깊이 항만 가중하는 기존 OccCfg,
0.10 m 격자, 기본 SGBM, 4픽셀 역투영 간격, 기존 동료 드론 마스킹을 공통 적용했다.
깨끗한 지도는 **비교 기준이며 독립적인 참 점유지도가 아니다**.

{table(miluv,["condition","beta","valid_fraction","depth_absdiff_mean_m","n_occupied","n_free","n_unknown","occupied_iou_clean","free_iou_clean"])}

깊이 차이는 무처리와 해당 조건이 **함께 유효한 픽셀**에서 계산한 절대 차이의 평균이다.
`joint_fraction`도 함께 저장했다. 겹침은 voxel 허용오차 없는 exact IoU다.
모든 조건의 occupied/free 인덱스 합집합을 포함하는 공통 직육면체 안에서 unknown을 센다.
분모는 {miluv[0]["domain_voxels"]}복셀로 고정돼 있다. SGBM 유효 픽셀이 적게 줄더라도
대응되는 깊이와 벽 위치는 상당히 달라질 수 있다. 이 겹침 감소만으로 실제 장애물 정확도를 판단할 수 없다.

중간 비균일 연기의 추가 시드 반복(같은 프레임·포즈):

{table(seedcheck,["condition","seed","valid_fraction","depth_absdiff_mean_m","occupied_iou_clean","free_iou_clean"])}

그림: `miluv_changes.png`, `miluv/map_comparison.png`, `miluv/examples/`.

## 2. 합성 복도에서 검증한 진입 판정

장면 `corridor915f`, 기존 `gt_voxel_fs.npz`, 드론 3대 각 100쌍(stride 12), 총 300쌍.
원래 합성 데이터 seed 0과 C_3drone 추정 포즈를 사용했다.
**참 포즈에서 관측을 생성하고 추정 포즈로 적분**했다. 추정 포즈로 렌더하는 과거의 낙관적 규약은 사용하지 않았다.
스캔 텍스처로 생성한 mono8 스테레오에 연기를 합성하고 SGBM을 다시 계산했다.
기존 `clean3d.clean`과 `TV.traverse`, 기본 `TravCfg`로 도달률·오도달률을 평가했다.
GT의 도달 가능 컬럼 수는 {corridor[0]["n_gt_reach"]}이다. 전체 GT 격자가 분모이며, 관측 범위만으로 분모를 줄이지 않았다.

{table([r for r in corridor if r["policy"]=="initial"],["condition","reach_recall","false_reach_rate","n_map_reach","n_false_reach","confirmed_reachable_goals","unverified_goals"])}

복도의 센서 단계 수치:

{table(sensor,["condition","valid_fraction","joint_fraction","depth_absdiff_mean_m","fill_left","fill_right"])}

연기에 따라 지표가 단조롭게 악화한다고 가정하지 않았다. 잘못된 점유 증거가 줄거나
관측이 더해져 통로가 좁아지는 경우도 있어, 도달 증가를 정확도 개선과 동일시하지 않는다.
목표는 GT 결과를 보기 전에 지정한 22개 좌표이며 목표 좌표 스냅은 하지 않는다.
GT가 실제 도달 가능한 목표, 관측된 고형물로 막힌 목표, GT 미확인 목표를 `goals.csv`에서 구분한다.
이 목표 집합은 GT 도달 16개, 관측된 고형물로 막힌 목표 1개, 나머지 GT 비도달/미확인 5개다.
따라서 목표 단위 오확인 0건은 작은 표본의 결과이며, 일반적인 안전성을 입증하지 않는다.

## 3. 같은 한 쌍 예산의 재관측

조건마다 초기 300쌍과 후보 24개(드론당 8개)가 동일하다. 후보는 초기 프레임과 겹치지 않는다.
이는 **합성 관측 기록의 재생 실험**이며, 실제 비행·이동 비용·충돌 없는 후보 접근을 검증하지 않는다.
nearest는 각 드론의 첫 초기 카메라 포즈로부터의 유클리드 거리를 최소화한다.
unknown은 현재 지도에서 예측한 미관측 복셀 수를 최대화한다.
route는 입구와 여러 목표를 연결하는 현재 지도의 낙관적 경로 주변·장애물 경계에서
미관측 영역을 보여줄 것으로 예상되는 후보를 고른다. 이 2D 경로는 선택용 근사이고 최종 판정은 기존 3D 모델이다.
모두 현재 occupied에서 멈추는 공통 카메라 광선 예측을 사용한다.

`rank_candidates()`에는 현재 지도, 후보 추정 포즈·시간, 보정값, 임무 목표만 전달된다.
후보 영상·깊이·참 포즈·GT는 입력에 없다. 점수와 선택을 JSON에 기록한 **뒤에**
선택된 후보의 실제 영상을 생성한다. 미래 영상 접근 시 예외를 내는 테스트도 통과했다.
초기 log-odds는 전체 `.ot`로 복제해 세 방식에 동일한 초기 증거를 제공한다(`.bt`는 쓰지 않는다).

**통과 확인**은 몸 공간과 발판이 관측된 경로가 있을 때만 인정한다.
기존 판정기의 0.5 m 미관측 바닥 건넘으로만 이어지는 목표는 **미확인**이다.
이를 판별하려고 같은 판정기를 `gap=0`으로도 실행했다. 기존 `gap=0.5` 도달 지표는 별도 보존했다.
경로를 못 찾았다는 이유만으로 막혔다고 단정하지 않는다.

**최종 대조에서 추가한 보수적 확인 규칙(사후 처리).** 강한 비균일 조건에서만 입구가
0.224 m 떨어진 발판으로 스냅됐다(다른 조건은 0 m). 이를 지정 입구의 통과 확인으로
세지 않도록 확인 판정에 `snap_m=0`을 적용했다. 이 조건의 확인 목표는 4→0개,
전부 미확인이 됐다. 기존 도달률·오도달률은 기존 스냅 규약 그대로 보존한다.
아래 표의 확인 수는 `summary_conservative.csv`, 목표별 판정은 `goals_conservative.csv`다.
원래 근처 발판에서 시작한 확인 수는 같은 CSV의 `legacy_confirmed_reachable_goals`와
원래 `summary.csv`·`goals.csv`에 남아 있다. 이 규칙 추가는 입력이나 후보 선택을 바꾸지 않는다.

{table([r for r in corridor if r["policy"]!="initial"],["condition","policy","candidate_id","reach_recall","false_reach_rate","confirmed_reachable_goals","delta_true_confirmed","false_confirmed_gt_unreachable","false_confirmed_gt_blocked","gap_permitted_true_goals","delta_gap_true_goals","false_gap_permitted_gt_unreachable"])}

통과 확인 수는 실제 GT 도달 가능 목표에서만 센다.
false-confirmed는 GT 비도달 목표를 확인했다고 표시한 수이며,
그중 관측된 고형물에 막힌 목표를 `false_confirmed_gt_blocked`로 따로 센다.
GT 미관측을 실제 벽이라고 부르지 않는다. `goals.csv`에는 모든 목표의 전후 판정이 있다.
도달 면적·목표 확인 수·오도달을 함께 봐야 한다. 한 장면·한 후보 예산의 규칙 기반 기준선으로,
경로 중심 방식의 일반적 우월성을 주장하지 않는다.

그림: `corridor/entry_comparison.png`, `corridor/reobservation_comparison.png`,
`corridor/map_comparison.png`. 선택별 원래 지도와 `.ot`는 `results/smoke_20261103/corridor/`에 보존했다.

## 4. 별도 VIO 재실행

{table(vio,["condition","run_id","n_input_stereo","n_poses","first_t","last_t","common_ate_yaw_rmse_m","max_output_gap_s"]) if vio else "VIO 결과 파일이 아직 없으므로 완료 결과를 기재하지 않는다."}

VIO는 ifo001의 첫 20초, 2프레임마다 한 쌍과 원래 PX4 IMU를 재생한 별도 실험이다.
무처리와 beta=0.8 균일 감쇠를 비교한다. 원래 카메라 보정과 원시 영상 좌표계를 유지했다.
무처리 raw 영상은 원본 JPEG를 디코딩한 값과 비트 단위로 같고 PNG로 저장했다.
연기 조건은 rectified 좌표의 투과율만 raw 좌표로 역투영한 뒤 원시 영상에 혼합했다.
원본 VINS 출력 경로를 쓰는 과거 `run_seq.sh`는 호출하지 않았다.
평가는 정제 mocap과 공통 시간 구간에서 yaw+이동 정렬하며 스케일 정렬은 없다.
첫 실행에서 연기 조건의 오차가 더 작게 나와 같은 입력의 실행 순서를 바꿔 반복했다.
역순 반복에서도 두 조건의 결과는 같았다(무처리 0.3206 m, 감쇠 0.0155 m).
순서 영향은 보이지 않았지만, 한 짧은 구간의 특징점·초기화 효과를 일반화할 수 없어
실제 연기가 VIO를 개선한다는 근거로 해석하지 않는다.
이 짧은 1대 실험은 3대의 연기 속 VIO·UWB 전체 재융합이나 장시간 추적 성공을 검증하지 않는다.
ROS 로컬 통신은 샌드박스에서 차단돼 승인된 샌드박스 밖 실행을 사용했다.

## 5. 모의 모델·선행연구·변경점

- [Foggy Cityscapes Demo](https://github.com/sakaridis/fog_simulation-SFSU_synthetic/blob/master/source/Demo_fog_simulation_Cityscapes.m),
  [haze_linear.m](https://github.com/sakaridis/fog_simulation-SFSU_synthetic/blob/master/source/Fog_simulation/haze_linear.m),
  [transmission_homogeneous_medium.m](https://github.com/sakaridis/fog_simulation-SFSU_synthetic/blob/master/source/Fog_simulation/transmission_homogeneous_medium.m):
  깊이 Z를 광선 거리로 바꾸고 `t=exp(-beta*distance)`, `I=t*J+(1-t)*A`를 적용하는 원리를 참고했다.
  Cityscapes 깊이 보완·RGB 산란광 추정·guided filtering은 이식하지 않았다.
- [Estimating Fog Parameters §V-A](https://arxiv.org/html/2511.20865v1#S5.SS1),
  [저장소](https://github.com/SenseRoboticsLab/estimating-fog-parameters):
  좌우 각각의 거리 지도를 쓰며 같은 쌍의 감쇠 설정을 공유하는 기준을 따랐다.
  ORB-SLAM2나 그 논문의 추정기는 도입하지 않았다.
- [SmokeSeer Resources](https://github.com/cmu-ci-lab/SmokeSeer#resources):
  공개 `real.zip`와 `synthetic.zip`에서 예시 RGB 프레임을 HTTP 부분 다운로드해 직접 확인했다.
  멤버 경로와 전송 바이트는 `reference_examples/download_report.json`에 있다.
  `reference_visual_check.png`에서 실제 연기의 국소적인 두꺼운 가림·거친 경계,
  합성 예시의 회색 베일과 우리 근사의 차이를 비교했다. 학습·3DGS는 실행하지 않았다.

MILUV는 실외 가시광 영상이 아니라 **적외선 스테레오 mono8**이다.
따라서 이 실험은 **거리 의존 감쇠와 대비 저하를 모사한 센서 저하 실험**이다.
실제 화재 검증이 아니며 beta를 화재 연기 농도·가시거리로 환산하지 않는다.
카메라 응답·파장별 산란이 보정되지 않아 intensity 영역에서 gamma=1로 혼합했다.
A=0.8(204/255), beta=0 / 0.15 / 0.4 / 0.8 [m^-1]는 실험 설정값이다.

비균일 조건은 seed 0의 공유 3D 장을 사용한다. 세 사인 모드의 파장 1.2–4.0 m,
진폭 0.55/0.30/0.10, 평균 밀도 1, 범위 0.05–1.95, 이류 속도 (0.06,0.025,0.015) m/s다.
각 카메라 광선의 장 밀도를 해석적으로 적분해 투과율을 만든다.
좌우·프레임별 독립 랜덤 마스크는 없다. 시간은 각 데이터의 상대 초이다.
MILUV 장 좌표에는 기존 융합 포즈, 복도 합성에는 참 카메라 포즈를 쓴다.
장 자체의 연속성을 검증했지만, 추정 깊이와 포즈의 흔들림으로 합성 투과율은 불연속일 수 있다.

깨끗한 왼쪽 깊이는 원래 SGBM, 오른쪽 깊이는 좌우 영상을 반전해 오른쪽 기준으로
동일 SGBM을 수행해 얻었다. 각 카메라에서 깊이가 없는 픽셀은 가장 가까운 유효 Z로 채웠다.
카메라 전체가 무효이면 Z=5 m를 **합성에만** 쓴다. 오른쪽과 왼쪽의 fill 비율을 기록했다.
이 방식은 빈 영역에 Voronoi 경계 같은 인공적 투과율 무늬를 남길 수 있고,
좌우의 추정 깊이 오차도 연기 합성에 영향을 준다. 따라서 순수한 실제 연기 효과로 해석할 수 없다.
깊이 보완값이나 깨끗한 깊이는 연기 조건의 지도 적분에 **한 점도 사용하지 않는다**.
모든 지도의 입력은 해당 조건 영상에서 새로 계산한 SGBM 깊이다.

## 6. 검증·재현·보존

작은 MILUV 기준 실행: 3대 각 4프레임에서 기존 `process_robot`과
**96,061개 leaf의 log-odds가 정확히 같았다**. 원시 입력에 대한 `StereoDepth.depth`와
새 rectified 경로의 깊이·분산·유효 마스크도 픽셀 단위로 같았다.
복도 작은 기준 실행은 3대 각 4프레임으로 생성→SGBM→지도→판정을 완료했다.
그 작은 초기 관측의 도달률은 0이며, 전체 구간 성능으로 인용하지 않는다.

새 검증 10개, 기존 ray traversal 9개, 기존 사람형 판정 11개(총 30개)를 직접 실행했다.
결과 로그는 이 폴더의 `test_*.log`다. covor 환경에는 pytest가 없어 각 파일의 기존
직접 실행 runner를 사용했다. 시험을 통과시키기 위한 기존 구현 변경은 없다.

```bash
export MPLCONFIGDIR=/tmp/covor_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/src/gs25058/miniconda3/envs/covor/bin/python
O=results/smoke_20261103
$PY scripts/smoke/run_miluv.py --out $O/miluv_baseline --frames 4 --stride 6 --levels clean --baseline-check
$PY scripts/smoke/run_miluv.py --out $O/miluv --frames 48 --stride 16 --seed 0
$PY scripts/smoke/run_miluv.py --out $O/miluv_seedcheck --frames 48 --stride 16 --levels heterogeneous_medium --seeds 1,2
$PY scripts/smoke/run_corridor.py --out $O/corridor_baseline --frames 4 --stride 12 --levels clean --skip-selection
$PY scripts/smoke/run_corridor.py --out $O/corridor --frames 100 --stride 12 --seed 0 --candidates 24
$PY scripts/smoke/prepare_vio.py --out $O/vio --duration 20 --image-stride 2 --beta 0.8
(
  source /src/gs25058/ws/vins_ws/env.sh
  source /src/gs25058/ws/vins_ws/devel/setup.bash
  python scripts/smoke/replay_vio.py --input $O/vio --rate 0.5
  python scripts/smoke/replay_vio.py --input $O/vio --rate 0.5 --run-id reversed1 --order uniform_strong,clean
)
$PY scripts/smoke/evaluate_vio.py --input $O/vio
$PY scripts/smoke/evaluate_vio.py --input $O/vio --run-id reversed1
python3 scripts/smoke/fetch_reference_examples.py --out $O/reference_examples
$PY tests/test_smoke_reobserve.py
$PY tests/test_ray_traversal.py
$PY tests/test_entry_traverse.py
$PY scripts/smoke/audit.py --input $O --out reports/smoke_20261103
$PY scripts/smoke/summarize.py --input $O --out reports/smoke_20261103
```

VINS는 별도 covins 환경, 그 외 추론은 covor 환경이다. 실제 버전은 `environment.json`.
큰 cache·포즈·3D 격자는 results/ 아래, 검토용 CSV·그림·이 문서는 reports/ 아래에 있다.
원래 메시와 기준 결과는 읽기만 했다. 산출물 SHA256은 `artifacts.json`에 있다.
'''
    (out/"RESULTS.md").write_text(text)
    write_json(out/"artifacts.json",{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(out.rglob("*")) if p.is_file() and p.name!="artifacts.json"})
    print(out/"RESULTS.md")


if __name__=="__main__":main()
