# HANDOFF — 세션 인수인계

> 새 세션은 **이 문서 하나만 먼저 읽으면** 프로젝트의 현재 상태·함정·다음 단계와
> "무엇을 더 읽어야 하는지"를 파악할 수 있다. 상세 내용은 각 전용 문서로 포인터만
> 건다(중복 금지). 마지막 수정: 2026-08-01.

---

## 1. 프로젝트 한 줄 요약 + 현재 단계

**불확실성 인지 다중드론 협업 SLAM 기반 항법용 occupancy map** 만들기. 여러 드론이
VIO + UWB 거리로 협업 정합한 포즈와 **그 포즈의 불확실성(Σ)** 을 함께 써서, 확신이
낮은 관측일수록 약하게 반영하는 가중 occupancy map을 만드는 것이 핵심 novelty.

- 계획서: `reference files/연구계획서_불확실성인지_다중드론_점유지도.md`
  (원논문: 같은 폴더의 CoVOR-SLAM PDF, 데이터셋: MILUV `2504.14376v1.pdf`)
- **현재 단계**: 프론트엔드를 ORB-SLAM3 mono → **VINS-Fusion(스테레오+IMU VIO)** 으로
  교체 완료(4개 궤적 확보). **바로 다음 = fusion 레이어 SE(3) 전환**(§6).

---

## 2. 저장소·환경 상태

- **브랜치**: `vins-frontend` / **HEAD**: `a0a5e07` (이 HANDOFF 커밋 전 기준).
  `occupancy-pipeline` 브랜치와 같은 커밋에서 분기했고 **자체 커밋은 이 문서가 처음**.
- **미커밋(이전 occupancy 작업 잔여, 이번 전환과 무관)**: `M .gitignore`,
  `M OCCUPANCY_PIPELINE.md`, `?? scripts/export_occupancy_web.py`, `?? web/`,
  세션 export `.txt` 2개. HANDOFF 커밋에는 **포함하지 않았다**(별도 정리 대상).
- 원격: `origin git@github.com:gs25058/Uncertainty-Aware-Collaborative-Mapping.git`

### ★ 경고: VINS 산출물은 리포 밖에 있다 (git 미추적 → 유실 위험) ★
프론트엔드 빌드·궤적·평가 스크립트 전부 **리포지토리 바깥** `/src/gs25058/ws/vins_ws/`
에 있다. `git`이 추적하지 않으므로 이 디렉토리가 지워지면 4개 궤적과 하네스가 사라진다.

```
/src/gs25058/ws/vins_ws/
├── env.sh                     # covins 환경 소싱 + CERES_DIR_HINT
├── src/VINS-Fusion/           # 빌드 소스(@be55a93, 빌드-호환 패치 적용됨)
├── devel/lib/vins/vins_node   # 빌드 산출 바이너리
├── VINS_BUILD_NOTES.md        # 빌드 호환 패치 10건 기록
├── VINS_RESULTS.md            # 4개 궤적 위치/회전 오차 상세 + 재현 명령
└── run/
    ├── run_seq.sh  eval_traj.py  plot_all.py
    ├── cfg/<seq>/<robot>/{vins.yaml,cam0.yaml,cam1.yaml}
    └── out/<seq>/<robot>/vio.csv          # ★ raw local-frame 궤적 = fusion 입력 ★
```

### 실행 환경 (이미 검증됨, 재조사 불필요)
- conda(micromamba) env **`covins`** = RoboStack **ROS1 Noetic**(roscore/roslaunch/
  rosbag/catkin_make + Ceres 1.14 / OpenCV 4.6 / Boost 1.78 / glog 0.7).
- **Docker·sudo 불가**(데몬 접근 없음) → conda-Noetic이 유일한 실행 경로. 재시도 금지.
- catkin 워크스페이스: `/src/gs25058/ws/vins_ws` (`source env.sh` 후 사용).
- **`ROSCORE_PORT` 오버라이드**로 포트 충돌 회피(기본 11321).

### 재현 명령
```bash
cd /src/gs25058/ws/vins_ws/run
# 1개 로봇 실행: run_seq.sh <seq> <robot> <wall_s> <log> [rosbag args]
ROSCORE_PORT=11321 bash run_seq.sh default_3_zigzag_0 ifo002 420 run_ifo002.log "-s 5"
# 평가(회전 포함) — 반드시 COVOR env python (covins엔 pandas/scipy 없음)
/src/gs25058/miniconda3/envs/covor/bin/python eval_traj.py default_3_zigzag_0 ifo002
# 4개 종합 플롯
/src/gs25058/miniconda3/envs/covor/bin/python plot_all.py   # -> vins_all_trajectories.png
```

---

## 3. 완료된 것 + 핵심 수치

| 단계 | 상태 | 상세 문서 |
|---|---|---|
| CoVOR-SLAM 재현(ORB-SLAM3 mono) | 완료·**비교군 보존** | `FAITHFUL_REPRODUCTION.md` |
| 불확실성 인지 occupancy 파이프라인 | 구현·검증 완료 | `OCCUPANCY_PIPELINE.md`, `DIAGNOSIS.md` |
| VINS-Fusion 프론트엔드 전환 | 4개 궤적 확보 | `ws/vins_ws/VINS_RESULTS.md`, `VINS_BUILD_NOTES.md` |

**CoVOR 재현(ORB-SLAM3 mono, 비교군)**: VO baseline ATE **0.167 / 0.782 / 0.606 m**
(ifo001/002/003, Sim3 정렬), 회전 median **21–34°**, 커버리지 **94 / 93 / 98 %**,
맵브레이크 **7–13회**.

**Occupancy 파이프라인**: 가중 log-odds `w = exp(−trΣ/α)·exp(−σ_Z²/β)`, 안전 비대칭
`|l_free| < |l_occ|`, unknown≠free. weighted vs uniform ablation에서 uniform-free
**8,497셀 중 473셀(5.6%)** 억제 — 원거리·고드리프트 관측에 **공간적으로 집중**(무작위
아님). GT-pose 대조로 남은 지도 결함이 mapping이 아니라 **프론트엔드 포즈 오차**임을 확정.

**VINS-Fusion 전환(요약표, 상세는 VINS_RESULTS.md)** — VIO, no loop closure:

| seq / robot | poses | 커버리지 | 위치 RMSE | **tilt(roll/pitch)** | yaw | yaw drift |
|---|---|---|---|---|---|---|
| zigzag / ifo001 | 4357 | 전구간 연속 | 0.207 m | **1.13°** | 3.41° | −0.42°/min |
| zigzag / ifo002 | 4347 | 전구간 연속 | 0.231 m | **1.14°** | 2.76° | +0.45°/min |
| zigzag / ifo003 | 4339 | 전구간 연속 | 0.259 m | **0.89°** | 4.73° | −3.86°/min |
| obstacles / ifo001 | 3060 | 전구간 연속 | 0.107 m | **0.74°** | 2.53° | +2.86°/min |

핵심: **맵브레이크 소멸**(ORB의 7–13회 → 0, ifo002는 트래킹손실 1460회 로봇), **tilt
~1°** (ORB 21–34° 대비 ~20–30배↓, IMU 중력 관측 작동), yaw만 느리게 드리프트(미관측 축).

---

## 4. ★★ 값비싸게 배운 교훈 / 다시 밟으면 안 되는 함정 ★★

**이 문서에서 가장 중요한 섹션.** 대부분 다른 문서에 없음. 각 항목 = "하지 말 것 + 왜".

1. **ORB-SLAM3 `loopClosing: 0`은 map merging까지 끈다**(`mbActiveLC` 하나로 묶임).
   드론 영상은 트래킹 손실이 잦아 지도가 ~10조각으로 갈라지고 궤적이 4~15%만 남는다.
   "순수 VO 만들려면 루프클로저만 끄면 된다"는 접근 → **실패**. (VINS 전환이 이걸 해소.)
2. **Vicon(mocap) 정렬 궤적을 fusion 입력으로 쓰지 말 것.**
   `_vio_loop_aligned_and_shifted.csv` 계열은 GT로 이미 정렬돼 있어, 쓰면 UWB의
   역할(드론 간 정합)을 GT가 대신해 실험이 무의미해진다. **에러 없이 조용히 실패하고
   결과가 오히려 좋아 보여 특히 위험.** fusion 입력은 반드시 raw local-frame
   `run/out/<seq>/<robot>/vio.csv`. (평가용 정렬은 `eval_traj.py` 안에서만, 파일로 저장 안 함.)
3. **Σ는 VINS가 아니라 우리 GTSAM 융합에서 나온다**(계획서 §2.5: marginal covariance
   H⁻¹). **VINS 출력에 공분산 컬럼이 없는 건 정상**이고 문제가 아니다. 찾으려 하지 말 것.
4. **광선 free 증거 중복 계산 버그**: 호길이 등간격 샘플링 → 한 voxel에 2~4회 누적,
   동시에 모서리를 스치는 셀은 누락. 안전 비대칭이 코드에서 뒤집혀 있었음. `_dda_batch`로
   수정 완료. **회귀 테스트는 voxel 개수가 아니라 셀별 log-odds 값을 비교할 것** — 개수만
   보는 테스트가 이 버그를 그대로 통과시켰다.
5. **OctoMap `write_bt`는 파괴적**(max-likelihood 변환 + pruning). **분류·플롯을 먼저
   하고 `write_bt`는 마지막에.** 순서를 어겨 free 카운트가 틀리게 보고된 적 있음.
6. **`tau_occ = l_occ`인 이유**: 단발 관측 기여가 `w·l_occ ≤ l_occ`이므로, 이 임계값이면
   관측 1회로 occupied가 **원리적으로 불가능**. 임의 튜닝이 아니라 성질을 만족하는 최소값.
   임의로 낮추지 말 것.
7. **UWB 실측 품질**: RMSE 0.22–0.24 m, bias +0.05, std 0.22, p99 ~0.9 m.
   inter-range(0.17)가 anchor(0.26)보다 정확. **CSV의 std 컬럼은 실제보다 낙관적**이라
   그대로 쓰면 과신. 일반 원리: **VO 오차 > UWB 유효정확도일 때만 융합이 이득**(그래서
   저드리프트 ifo001은 융합이 오히려 손해).
8. **시퀀스를 random으로 바꾸는 건 해결책이 아님**(검토 완료). 드리프트가 안 쌓이는
   이유는 궤적 모양이 아니라 **방이 좁아 재관측이 잦기 때문**.
9. **MILUV 앵커 6개가 모두 ~1.7 m 높이 → 수직(VDOP) 관측 불가.** 완벽한 gt_range로도
   수직을 못 정한다. height 팩터가 보완(zRMSE 0.30→0.05)하나 **계획서 범위 밖 확장**이라
   논문 충실 구성에선 기본 OFF.
10. **`pkill`/`pgrep -f` self-매칭 주의**: `pgrep -f 'rosbag play'`가 자기 셸의 명령줄을
    매칭해 실제로 없는 "고아 프로세스"를 죽이려다 **exit 144**(자기 셸 종료)를 낸다.
    `pgrep -x <정확한이름>`을 쓰거나 패턴에서 `$$`를 배제할 것.
11. **`obstacles_1_random3_0b`는 `ifo001.bag`만 존재**(ifo002/003 bag·mocap 없음).
    이 시퀀스로는 3대 협업 실험 불가 — 단일 VIO 하드케이스 샘플로만 사용.
12. (이번 세션) **VINS `vio.csv` 쿼터니언은 w-first** `t,x,y,z,qw,qx,qy,qz,vx,vy,vz`.
    위치만 볼 땐 무관하나 **회전 오차 계산 시 순서를 틀리면 조용히 잘못된 값**이 나온다.
13. (이번 세션) **회전 오차는 best constant two-sided offset** `R_gt≈A·R_vins·B`(A=Vicon←VINS
    world, B=IMU→marker body) 제거 후 잔차로 측정. 이는 ORB-SLAM3 비교군과 **동일 방법론**
    이라 직접 비교 가능. 단측 정렬만 하면 body 장착 오프셋이 오차로 새어든다.
14. (이번 세션) **BLAS 스레드 과다구독으로 VINS가 ~30배 느려짐**(0.03x 실시간). `run_seq.sh`
    안의 `OPENBLAS/OMP/MKL_NUM_THREADS=1`이 필수 — 빼면 bag이 timeout 안에 안 끝난다.

---

## 5. 폐기된 접근과 그 이유 (지금 구조가 왜 이런지)

- **ORB-SLAM3 순수 VO 시도** → 지도 파편화로 폐기(함정 1).
- **ORB-SLAM3 IMU 모드 우회** → 계획서가 VINS-Fusion을 명시 + MILUV가 config/런처 제공
  + conda-Noetic으로 실행 가능 → **불필요**. (VINS가 안 됐을 때만 고려하기로 했으나 됐음.)
- **Docker 경로** → 데몬 접근·sudo 불가로 폐기, **conda-Noetic**으로 우회.
- **Sim(3) 재파라미터화**(`Pose3` + 스칼라 스케일) → CoVOR 재현용이었음. VINS는 metric이라
  **스케일 제거 예정**(§6). 계획서는 처음부터 SE(3).

---

## 6. 지금 열려 있는 작업 = 다음 단계: fusion 레이어 SE(3) 전환

4개 metric·중력정렬·전구간 VIO 궤적이 확보됐으므로 fusion에서 Sim(3) 스케일 장치를 제거:

- **`covor/factors.py`**: 스케일 변수 `Sc`·`scale_prior`·스케일 random-walk·odometry
  factor의 스케일 항 제거 → 순수 `Pose3` between-factor.
- **`scripts/fuse_and_dump.py`**: 현재 rank-deficient 회피용 "모든 변수 약한 prior(σ=10)"는
  UWB가 위치만 구속해 회전 게이지가 자유롭기 때문. **VINS는 중력으로 roll/pitch가 고정**되어
  남는 게이지가 yaw + 위치뿐 → 정칙화 단순화. 그 결과 `tr(Σ)`가 정칙화 인공물이 아니라
  **실제 정합 불확실성**이 되어 occupancy 가중에 의미 있게 들어간다.
- **`covor/occupancy.py`**: **무변경**(`(pose, tr_sigma_pos, depth)` 인터페이스 독립).
- **입력**: raw `vio.csv` 4개 — `/src/gs25058/ws/vins_ws/run/out/{default_3_zigzag_0/
  ifo001,ifo002,ifo003 ; obstacles_1_random3_0b/ifo001}/vio.csv`.

이후 후보(한 줄씩):
- occupancy 재생성 + weighted-vs-uniform ablation 재측정(새 프론트엔드 기준).
- **yaw 드리프트를 UWB inter-range로 잡을 수 있는지** 검토(inter가 anchor보다 정확).
- σ_Z 공간 분산 반영(설계 확장 → **승인 필요**).
- Phase 4: 사람/동적 진입 시나리오 변환.

---

## 7. 파일·디렉토리 역할 맵

### 리포 내 (`/src/gs25058/cr_RNE/covor_slam/`)
| 경로 | 역할 |
|---|---|
| `HANDOFF.md` | (이 문서) 세션 인수인계 진입점 |
| `FAITHFUL_REPRODUCTION.md` | CoVOR 논문 충실 재현·ATE 표·충실성 체크리스트(비교군) |
| `OCCUPANCY_PIPELINE.md` | 가중 occupancy 설계·결함 진단·GT-pose 대조·ablation |
| `DIAGNOSIS.md` | 융합 이득 조건, UWB 품질, 시퀀스/앵커 분석 |
| `covor/factors.py` | GTSAM 팩터(현재 Sim3 스케일 포함 → SE(3) 전환 대상) |
| `covor/occupancy.py` | 불확실성 가중 occupancy 모듈(인터페이스 독립, 무변경) |
| `scripts/fuse_and_dump.py` | 융합 실행 + Σ 추출(정칙화 단순화 대상) |
| `reference files/` | 계획서(.md) + CoVOR·MILUV 논문 PDF |

### 리포 밖 (`/src/gs25058/ws/vins_ws/`) — ★ git 미추적, 유실 주의 ★
| 경로 | 역할 |
|---|---|
| `VINS_RESULTS.md` | 4개 궤적 위치/회전 오차 상세 + 재현 명령 |
| `VINS_BUILD_NOTES.md` | 빌드 호환 패치 10건(알고리즘 변경 아님) |
| `run/run_seq.sh` | VINS 실행 하네스(seq/robot 파라미터화, 스레드 캡) |
| `run/eval_traj.py` | mocap 대비 위치+회전(roll/pitch/yaw) 평가 |
| `run/plot_all.py` | 4개 종합 플롯(top-down + tilt/yaw vs time) |
| `run/out/<seq>/<robot>/vio.csv` | **raw local-frame 궤적 = fusion 입력** |
| `run/cfg/<seq>/<robot>/` | 로봇별 vins.yaml + cam calib(output_path 리다이렉트) |

### 데이터셋 (`/src/gs25058/cr_RNE/miluv/`)
`data/<seq>/<robot>.bag`(스테레오 compressed + IMU), `data/<seq>/<robot>/mocap.csv`(GT),
`data/<seq>/timeshift.yaml`, `config/vins/<robot>/`(IMU 노이즈·extrinsic 완비).

---

## 8. 불변 원칙 (매 작업에서 지킬 것)

- **계획서 설계 준수**: 가중 log-odds, 안전 비대칭(`|l_free|<|l_occ|`), unknown≠free,
  weighted on/off ablation 토글 유지.
- **OctoMap/OpenCV/VINS의 C++ 알고리즘 수정 금지**(빌드 호환 패치는 예외 — 반드시 기록).
- **CoVOR 재현 결과(`FAITHFUL_REPRODUCTION.md`, ORB-SLAM3 산출물) 훼손 금지** — 비교군.
- **원인 확정 없이 파라미터 튜닝 금지.** 목표는 "보기 좋은 결과"가 아니라 "정확한 원인 제거".
- **측정 없이 추측으로 수정하지 말 것.**
