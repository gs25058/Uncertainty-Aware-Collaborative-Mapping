# 발표 슬라이드용 그림

모두 PNG, 300 dpi, 흰 배경, 그림 안 제목 없음, 글자 14 pt 이상. 드론 색 #1f77b4 / #d62728 / #2ca02c, GT 검은 점선.
글꼴은 NanumGothic (Noto Sans CJK KR 미설치). 알고리즘·파라미터는 손대지 않았고, 그림에 쓴 수치는 모두
생성 시점에 아래 출처와 대조한다 (불일치면 스크립트가 assert로 멈춘다).

`PY=/src/gs25058/miniconda3/envs/covor/bin/python` (covor env), 그림 3만 시스템 `python3` (Playwright 설치됨).

| 그림 | 입력 데이터 | 생성 명령 | 사용한 수치 출처 |
|---|---|---|---|
| `s1_traj.png` (2977×1570) | VINS 원시 궤적 `/src/gs25058/ws/vins_ws/run/out/default_3_zigzag_0/<robot>/vio.csv` (`covor.data.load_vins`); 융합 포즈 `vo_output/occ_default_3_zigzag_0_<robot>_af.npz` (조건 C, 앵커 없음); GT `covor.data._mocap_splines` (정제 mocap) | `$PY scripts/slides/s1_traj.py` | 드론별 4-DoF ATE 0.1517 / 0.2218 / 0.3010 m = `results/results_49_collaboration.csv` 조건 C `ate_ifo00{1,2,3}` (재계산값 일치 assert). 세 값의 평균 0.2248 = `reference files/RESULTS_SUMMARY.md` §7.2 조건 C ATE_per |
| `s3_entry3d.png` (3840×2160) | `web/traverse_3d_corridor915f.html` (사람형 판정 뷰어, `scripts/entry/export_traverse_3d.py` 출력), 변형 `sgbm` = SGBM (기존 규약) | `python3 scripts/slides/s3_entry3d.py` | 그림에 수치 없음. 판정은 `RESULTS_traversability.md` §6 · §7.1 표의 SGBM (기존 규약) 행 (recall 0.680, false-reach 0.0173)과 같은 실행 |
| `s5_trsigma.png` (3063×1584) | `results/results_49_collaboration.csv` (A_1drone_af, B_2drone_af, C_3drone_af, D_3drone_anch) | `$PY scripts/slides/s5_trsigma.py` | tr(Σ) 중앙값 0.940 / 0.130 / 0.074 / 0.0015 m², IoU@1vox 0.306 / 0.537 / 0.518 / 0.584, false-free 5.978 / 4.999 / 5.268 / 4.426 % = `reference files/RESULTS_SUMMARY.md` §7.2 표 및 `paper/README.md` 표 (CSV 원값 0.93967 / 0.12998 / 0.07428 / 0.00151, 반올림 일치 assert) |

그림 2 (실제 복도 사진 vs 복셀 격자)는 이미 따로 있어 여기서 만들지 않는다.

## 그림별 메모

**그림 1.** (a)는 각 드론 VIO를 자기 좌표계 그대로 그린다 (어떤 정렬도 없음). (b)는 앵커 없는 융합이라 절대
좌표가 고정되지 않으므로 (융합 좌표계가 mocap에서 0.71–0.82 m 떨어져 있음) 세 드론 **공통의** yaw+이동 변환
하나만 맞춰서 GT 위에 올렸다 — 드론별로 따로 맞추면 이 그림이 보여 줄 드론 간 정합 자체가 가려진다.
표시된 ATE는 문서·CSV와 같은 드론별 4-DoF ATE. MILUV 시계는 `timeshift_s + timeshift_ns/1e9`, 쿼터니언 w-first,
mocap 정렬 VIO 파일은 쓰지 않는다 (`covor.data` 로더가 처리).

**그림 3.** 지정된 `web/entry_map_3d_corridor915.html`(띠 모델) 대신 `web/traverse_3d_corridor915f.html`(사람형
모델)을 썼다. 요구한 색 규칙(서서/숙여서/기어서/옆으로, 미도달 회색)과 데이터(`synth_corridor915f`)가 사람형
뷰어의 것이기 때문이다. 판정은 바꾸지 않고 표시만 바꿨다: 흰 배경, 안개·UI 패널 숨김, 천장을 바닥+0.60 m에서
자름(뷰어 자체의 높이 자르기), "설 수 있으나 미도달" 칸을 회색으로 재색칠, 오른쪽 아래 범례.
범례에 "관측 안 된 바닥 건넘 (≤ 0.5 m)"을 넣었다 — SGBM 도달 면적의 63 %(38.4/60.7 m²)가 이 칸이라
빼면 그림이 판정을 과장한다 (`RESULTS_traversability.md` §9). SGBM 지도에는 북쪽 복도가 없어 화면을 지도가
있는 영역으로 잘랐다. 헤드리스 Chrome + SwiftShader(`--use-angle=swiftshader`), 뷰포트 2560×1440, DSF 1.5,
카메라 theta 0°, 천정각 55°(수평에서 35° 위), r 22 m. 다른 변형은 `--variant gt|sgbm_rt|ideal`.

**그림 5.** A/C 비 0.93967/0.07428 = 12.7 → "13배 감소" (문서 표기와 같음). y축 로그, 눈금은 일반 숫자.
오른쪽 IoU·false-free는 측정값 그대로이며 축을 0에서 시작하지 않는 대신 두 패널 모두 값 라벨을 붙였다.
B→C에서 IoU 0.537→0.518, false-free 4.999→5.268 %로 오히려 약간 나빠지는 것(첫 쌍 이후 포화)이 그대로 보인다.
D(3대+앵커)는 참고용이라 연회색 빗금 막대·속 빈 점.
