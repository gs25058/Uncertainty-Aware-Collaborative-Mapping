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
- **현재 단계**: 프론트엔드 VINS-Fusion 교체 + **fusion 레이어 SE(3) 전환 완료**(§9).
  **융합이 3대 모두 VIO를 2.1~4.0× 이긴다**(§9 헤드라인). GT 대조군 재실행 완료
  (§10). **바로 다음 = §4.9 1/2/3-drone ablation** — 단, 착수 전 차단요인 2건
  해소 필요(§10): 쌍 리스트 ablation 미구현 + GT 상한선 프레임 오류.

---

## 2. 저장소·환경 상태

- **브랜치**: `vins-frontend`. `occupancy-pipeline`과 같은 커밋(`a0a5e07`)에서 분기.
- **커밋됨**: SE(3) 전환 일체(§9) — `covor/{data,factors,fusion}.py`,
  `scripts/{fuse_and_dump,build_occupancy,selftest_fusion,sweep_height}.py`,
  `run_covor.py`, `HANDOFF.md`, `OCCUPANCY_PIPELINE.md`.
- **여전히 미커밋(별도 정리 대상, 이번 작업과 무관)**: 웹 뷰어 유닛
  `?? scripts/export_occupancy_web.py` + `?? web/occupancy_viewer.template.html`
  (`web/occupancy_viewer.html`은 1 MB 생성물이라 gitignore), `M .gitignore`(그 규칙),
  세션 export `.txt`, `untitled.txt`.
  ⚠ `OCCUPANCY_PIPELINE.md`에는 **이 뷰어를 설명하는 문단이 이미 커밋돼 있다** —
  스크립트를 함께 커밋하거나 문단을 빼야 정합성이 맞는다.
- **git 미추적 산출물**(재생성 가능, `.gitignore`): `vo_output/occ_*.npz`, `results/`.
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
| zigzag / ifo001 | 4357 | 전구간 연속 | 0.171 m | **0.90°** | 3.96° | +1.06°/min |
| zigzag / ifo002 | 4347 | 전구간 연속 | 0.202 m | **0.84°** | 0.78° | −0.13°/min |
| zigzag / ifo003 | 4339 | 전구간 연속 | 0.242 m | **0.63°** | 6.00° | −4.48°/min |
| obstacles / ifo001 | 3060 | 전구간 연속 | 0.104 m | **0.69°** | 2.44° | +2.94°/min |

> 위 수치는 **2026-08-01에 `timeshift_ns` 누락을 고친 뒤의 값**(함정 15). 이전 값은
> 0.207/0.231/0.259/0.107 m 였다. 전후 대조·검증은 `VINS_RESULTS.md` §3.0.

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
   그대로 쓰면 과신. 일반 원리: **VO 오차 > UWB 유효정확도일 때만 융합이 이득**.
   ⚠ **"유효정확도 = RMSE 0.22 m"라는 당시 해석은 갱신됐다**(§9 경계조건). range를
   수천 개 쓰면 랜덤 성분은 1/√N로 평균되고 **계통 bias만 남으므로 실효 바닥은
   ~0.09 m**다. 그래서 ORB 시대엔 손해였던 ifo001이 지금은 2.1× 이득이다. 당시
   손해의 실제 주범은 VO 정확도가 아니라 **UWB의 89.9%를 버리던 연관 손실**이었다.
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
15. (이번 세션) **MILUV 시각 변환은 `timeshift_s + timeshift_ns/1e9`** (`miluv/utils.py:316`).
    `timeshift_s`만 쓰면 zigzag 0.256 s, obstacles 0.067 s 계통 편이가 생긴다. mocap이
    조밀해 최근접 매칭이 **항상 성공하므로 경고 없이** 오차만 부풀린다. 검증법: 올바른
    변환이면 VINS 포즈 시각이 스테레오 이미지 파일명 시각과 **정확히 0.000000 s** 일치.
16. (이번 세션) **VIO 궤적을 상대포즈 팩터로 바꾸면 절대 roll/pitch가 버려진다.**
    "VINS가 중력으로 tilt를 고정하니 그래프도 고정될 것"이라는 가정은 **틀렸다** —
    코드가 인코딩해야 한다(`gravity_prior`). 안 넣으면 tilt가 0.6~0.9°→3~4°로 악화하고
    **ATE까지 40~45% 나빠진다**(range 잔차가 엉뚱한 회전으로 흡수됨). §9 A/B 참조.
17. (이번 세션) **mocap에도 결함이 있다**: 0.13 s 사이 173~180° 회전(물리적 불가능)이
    ifo001/ifo002에서 ~0.5%, ifo003는 0%. 노이즈 σ를 RMS로 재면 이 꼬리에 지배당해
    로봇별로 5.4°/7.1°/0.74°처럼 흩어진다. **배제하거나 robust 추정을 쓸 것** —
    배제 후 0.734/0.776/0.735°로 일치하고, 배제 대상이 0건인 ifo003가 전후 동일값
    (0.735°)이라 절차 자체가 교차검증된다.
18. (이번 세션) **`gtsam.Marginals`는 이 그래프에서 병목이 아니다**(선형, 6.5k 노드 1.8 s).
    반면 **`GaussianFactorGraph.optimizeDensely()`는 절대 쓰지 말 것** — 39,138 DoF에서
    밀집 행렬 ~30 GB를 잡고 사실상 안 끝난다. 진단 스크립트에서 한 번 밟았다.

---

## 5. 폐기된 접근과 그 이유 (지금 구조가 왜 이런지)

- **ORB-SLAM3 순수 VO 시도** → 지도 파편화로 폐기(함정 1).
- **ORB-SLAM3 IMU 모드 우회** → 계획서가 VINS-Fusion을 명시 + MILUV가 config/런처 제공
  + conda-Noetic으로 실행 가능 → **불필요**. (VINS가 안 됐을 때만 고려하기로 했으나 됐음.)
- **Docker 경로** → 데몬 접근·sudo 불가로 폐기, **conda-Noetic**으로 우회.
- **Sim(3) 재파라미터화**(`Pose3` + 스칼라 스케일) → CoVOR 재현용이었음. VINS는 metric이라
  **스케일 제거 예정**(§6). 계획서는 처음부터 SE(3).

---

## 6. (완료됨 → §9) fusion 레이어 SE(3) 전환 — 당시 계획

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
- ~~절대 tilt를 그래프에 넣기~~ → **완료**(§9, `gravity_prior`).
- **occupancy α 재보정 → 재생성 + weighted-vs-uniform ablation 재측정.**
  α를 안 고치면 ablation이 무의미해진다. §9 말미 참조 — 착수 전 합의 필요.
- **초기 정렬을 앵커 range 기반으로 전환**(현재는 mocap Umeyama, 초기 추정치에만 사용).
  VINS가 metric·중력정렬이라 yaw+위치 4-DoF만 세우면 되고, 논문 충실성에도 나음.
  단 SE(3) 전환이 정상 동작함을 확인한 뒤에 대조군을 갖고 시도할 것.
- **UWB bias를 종류별로 분리**(`bias_mode` 활용). 실측 bias가 앵커 **+0.087 m** vs
  inter **+0.004 m**로 10배 다른데 지금은 하나로 취급(off)한다. 균일 상수 제거는
  오히려 악화했다(NEES 9.7→11.6). DIAGNOSIS의 "inter RMSE 0.17 < anchor 0.26"과
  일관. 융합 오차 바닥(~0.09 m)을 낮출 유일한 경로지만 ATE가 이미 좋고 과신은
  1.8×(허용 범위)라 급하지 않음.
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

---

## 9. SE(3) 전환 완료 + 중력 prior (2026-08-01)

### ★ 헤드라인: 융합이 두 입력을 모두 크게 이긴다 ★

이 프로젝트는 "융합하면 오히려 나빠진다"(DIAGNOSIS.md)는 미스터리로 시작했다.
부호가 뒤집혔다.

| robot | VIO 단독 | 융합 후 | 개선 | tilt | yaw | yaw drift |
|---|---|---|---|---|---|---|
| ifo001 | 0.1715 | **0.0809** | **2.1×** | 0.90→0.89° | 3.94→1.97° | +1.07→+1.21 °/min |
| ifo002 | 0.2019 | **0.0832** | **2.4×** | 0.85→0.85° | 0.78→1.29° | −0.12→+0.19 °/min |
| ifo003 | 0.2421 | **0.0613** | **4.0×** | 0.64→0.64° | 6.00→1.46° | **−4.48→+0.17 °/min** |

> **단위 주의.** 위는 VIO·융합 **모두 rigid(SE3) 정렬 ATE**로 통일한 값이다.
> 섞으면 개선폭이 부풀려진다: VIO에 timeshift 수정 **전** 값(0.207/0.231/0.259)을,
> 융합에 레버암 보정 절대오차(0.078/0.078/0.062)를 쓰면 2.7/3.0/4.2×가 나오는데
> 이는 서로 다른 정의를 비교한 것이다. 참고로 융합의 두 정의는 사실상 일치한다
> (정렬 ATE 0.0809 vs 절대오차 0.078) — 앵커가 절대 프레임을 실제로 고정하고
> 있다는 증거다.

기존 DIAGNOSIS의 "융합이 VO에 짐"이 뒤집혔다. 주된 원인은 UWB 연관률이
**10.1% → 98.0%** 로 오른 것(ORB 키프레임 1.4~1.8 Hz는 `range_tol=0.05 s` 안에
24,787개 중 2,497개만 잡았다. 22,290개를 버리고 있었다).

**yaw 기여 주장 성립**: 드리프트가 실제로 있던 유일한 로봇 ifo003에서
−4.48 → +0.17 °/min로 **26배 평탄화**. 나머지 둘은 애초에 드리프트가 없어 평탄해질 게
없었다. 균일한 개선이 아니라 **결함이 있던 곳에만 나타나는 개선**이라 방어하기 좋다.
(ifo002는 yaw가 0.78→1.29°로 약간 나빠지는데, 함정 7의 "VO 오차 < UWB 유효정확도면
융합이 손해"와 같은 패턴이다.)

### 중력 prior — 왜 필요했나 (A/B 대조)

VINS는 중력으로 roll/pitch를 관측하지만, 그 궤적을 **상대포즈 between-factor로
바꾸면 절대 성분이 버려진다.** 그래프의 절대 방위 구속은 로봇당 첫 노드 1개(6,523개
중 3개)뿐이었고, 상대 회전 체인(σ=0.0131 rad/step)의 누적 여유는 13 s에 7.5°,
전구간 35°. UWB는 tilt를 못 본다(|l|=0.231 m에서 5° 기울여도 태그는 0.020 m 이동 —
노이즈 플로어 0.05 m 미만). 그래서 tilt가 자유롭게 배회했다.

`gravity_prior`(GTSAM `Pose3AttitudeFactor`, 2-DoF, yaw 자유)를 노드마다 추가:

| | tilt(융합) | ATE(융합) | optimize |
|---|---|---|---|
| gravity **OFF** | 3.67 / 4.08 / 3.01° | 0.1449 / 0.1342 / 0.0849 | 547 s |
| gravity **ON** | **0.89 / 0.85 / 0.64°** | **0.0809 / 0.0832 / 0.0613** | **235 s** |

tilt가 VIO 값과 완전히 일치(0.89 vs 0.90 등)하게 복원됐다. **부수 효과가 더 크다**:
tilt가 자유일 때 range 잔차가 엉뚱한 기체 회전으로 흡수되고 있었고, tilt를 고정하니
range가 위치로만 설명돼 **ATE가 추가로 40~45% 개선**되고 수렴도 2.3배 빨라졌다.

σ_tilt = **0.0119 rad (0.68°)** 는 실측값(VIO tilt 잔차 크기가 Rayleigh 분포,
σ = median/1.1774 → 로봇별 0.764/0.731/0.548°). 꼬리는 가우시안보다 무겁고
(p90/median 2.17~2.82 vs 1.82) 네이티브 팩터가 robust 커널을 거부하므로,
이 σ는 **몸통을 기술할 뿐 꼬리는 기술하지 않는다**는 점을 유의.

### 경계조건 검증: 왜 부호가 뒤집혔나

DIAGNOSIS의 원리는 **"VO 오차 > UWB 유효정확도일 때만 융합이 이득"** 이고, 거기서
유효정확도를 **UWB RMSE 0.22 m**로 잡았다. 그 기준만으로는 지금 결과가 설명되지 않는다
— ifo001은 VIO 0.1715 < 0.22 인데도 2.1× 개선됐다. 실제로 확인된 설명은 두 겹이다.

**(1) ORB 시대의 실패는 "VO가 UWB보다 정확해서"가 아니라 파이프라인 결함이었다.**
- UWB의 **89.9%를 버리고 있었다**(키프레임 1.4~1.8 Hz, `range_tol=0.05 s` →
  24,787개 중 2,497개만 연관). 실측 확인.
- 맵브레이크로 게이지 자유 세그먼트가 생겨 σ=10 blanket prior가 필요했다.
- Sim(3) 스케일 자유도가 남아 있었다.

**(2) 융합 오차의 바닥은 UWB의 RMSE가 아니라 UWB의 계통 bias가 정한다.**
range 수천 개를 쓰면 **랜덤 성분은 1/√N로 평균되지만 공통 bias는 평균되지 않는다.**
- UWB 랜덤 성분: std **0.221 m**
- UWB 계통 bias: 전체 **+0.054 m**, 앵커만 **+0.087 m**, inter는 **+0.004 m**
- 관측된 융합 오차: **0.062 ~ 0.083 m** ← 앵커 bias 규모와 일치

즉 "0.22 m 문턱"은 range를 몇 개 못 쓰던 시절의 유효값이었고, 연관률이 98%가 된
지금의 실효 바닥은 **bias 수준(~0.09 m)** 이다. VIO 0.17~0.24 m가 전부 그보다
크므로 3대 모두 이득이 나는 것이고, 결과가 0.06~0.08 m에 수렴한 것도 이 예측과 맞는다.

**따라서 향후 개선 여지는 σ를 줄이는 게 아니라 bias를 분리하는 데 있다**(앵커와
inter의 bias가 10배 다르므로 `bias_mode`를 종류별로 쓰는 것 — 아래 후보 참조).

### occupancy α = 0.3 **유지** (변경하지 않음) — 근거 기록

Σ가 정칙화 인공물이 아님을 확인했다(정칙화 없이 Marginals 1.8 s 분해,
tr(Σ_pos) 0.0010~0.0026 m², σ_pos ≈ 3.2~5.1 cm). 그러자 α=0.3에서
`w_pose = exp(−0.0017/0.3) ≈ 0.994` (전체 0.991~0.997)로 포즈 항이 **사실상
무동작**이 된다. 처음엔 이걸 "α 미스매치"로 보고 α≈0.0017을 제안했으나 **철회**한다.

**철회 이유 1 — 그건 규칙의 재적용이 아니라 규칙 변경이다.**
ORB 시대의 실제 규칙은 α를 median trΣ의 **6배**로 잡은 것이었다
(median 0.05, α=0.3 → median w = 0.85). α = median(×1)로 잡으면 median w = 0.37로,
중간 품질 관측을 훨씬 세게 깎는 **새 규칙**이다. 그리고 "ablation 차이가 나도록"
α를 고른 뒤 그 차이로 방법론을 검증하면 **순환논증**이 된다.

**철회 이유 2 — 측정해보니 Σ의 판별력이 약하고 로봇 간 부호가 갈린다.**
NEES = 9.7(교정값 3) → σ가 **1.8× 과신**(SLAM 기준 완만). 그런데 균일 스케일 오차는
α가 통째로 흡수하므로 문제가 아니다. 진짜 문제는 **순위**다:
- trΣ 십분위별 실제 오차가 **U자형**(최저 십분위 0.074 m ≈ 최고 십분위 0.093 m)
- 자세(yaw) 층화로 교란을 제거하면 ifo002/ifo003는 층별 rho **+0.15~+0.60**으로
  양의 상관이 드러나지만, **ifo001은 −0.41~+0.26으로 부호가 갈린다**. 층 18개 중
  12개 양수, 중앙값 **+0.26** — 신호는 있으나 약하고 불일치.

→ 이 상태에서 α를 낮춰 포즈 항을 "깨우면" **반정보적 가중을 주입**하게 된다.
α=0.3에서 포즈 항이 **무해하게 잠자는 것이 올바른 동작**이다.

**그리고 α=0.3은 이미 "판별 정보가 있을 때만 켜지는" 범위에 맞춰져 있다.**
§4.9의 1/2/3-drone ablation에서는 UWB가 없는 로봇이 odometry-only로 남아 trΣ가
체인을 따라 발산한다(0.1~1.0 m² 이상). α=0.3이면 w가 0.72 → 0.036으로 **강하게
판별**한다. ORB 시대에 trΣ 0.05(정상) vs 100~216(게이지 자유)을 가르던 값과 같은 역할.

**포즈 항의 서사**: 포즈 항의 역할은 *구속된 곳과 안 된 곳을 가르는 것*이다. 3대가
전부 UWB로 구속된 단일 실행 안에는 가를 것이 없다. **실패가 아니라 조건의 부재**이며,
입증 무대는 §4.9의 1/2/3-drone 비교다. 현재 novelty를 지탱하는 축은 **깊이 항**
(15배 분산, σ_Z = Z²/(f·B)라는 명확한 물리 근거)이다.


---

## 10. GT 대조군 재실행 결과 + §4.9 차단요인 (2026-08-06)

상세는 `OCCUPANCY_PIPELINE.md` 말미 두 절. 요약:

**지도 품질이 프론트엔드 개선을 따라왔다.** 부유큐브 **1,101 → 48 (23배↓)**,
사물 기둥 회수 **9~10/13 → 13/13**. ORB 시대 격차의 원인이 회전 오차였다는
진단이 확인됐다.

**★ GT 대조군이 더는 상한선이 아니다 ★** fused의 부유큐브(48)가 GT 대조군(116)
보다 **적다**. 융합 포즈가 mocap을 이길 리는 없고, 원인은 측정됐다 —
`gt_pose_control.py`가 mocap 강체 포즈를 px4-IMU 포즈로 간주하는데 **두 프레임이
다르다**. fused와 GT 카메라 포즈가 **0.057~0.144 m(0.6~1.4 voxel), 2.9~4.1°**
차이난다. 이건 `eval_traj.py`가 two-sided `B`로 적합하는 바로 그 마커↔IMU 장착
오프셋이다. **§4.9의 상한선으로 쓰려면 먼저 고쳐야 한다**(`T_wb = T_w_marker · B`,
`B`와 레버암은 VINS-vs-mocap만으로 추정 가능 → GT 누수 없음).

**occupied −41% 해석 = 사전 규칙상 AMBIGUOUS.** 제거된 셀의 GT 상태가 허용오차에
따라 뒤집히고(GT-occupied 19.4% @0.5vox → 42.2% @1vox → 71.7% @2vox), 하필 GT
기준 자체의 프레임 오차가 그 판정 구간에 들어간다. 원거리 집중도 예측 방향이지만
약하다(3.00 m vs 2.56 m). **단, 정합 무관 집계는 깨끗하다**: uniform이 GT 대비
occupied를 **+36.6%** 부풀리는 반면 weighted는 **−9.4%**, 사물 부피도 uniform
+20.8% vs weighted +4.7%. weighted가 GT에 **3.9~4.4배 가깝고** 사물은 13/13 전부
유지. 즉 제거된 셀이 실제 장애물은 아니다. **β는 조정하지 않았다.**

**§4.9 착수 전 차단요인 2건**
1. **쌍 리스트 ablation이 구현돼 있지 않다.** `fuse_and_dump.py --drones`는
   **출력만 필터링**하고 그래프를 바꾸지 않는다(그 위 주석은 §4.9 ablation을 한다고
   주장하지만 사실이 아니다). `Cfg.inter_pairs` / `Cfg.anchor_robots` 신설 필요.
2. **GT 상한선 프레임 오류**(위).
