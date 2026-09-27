# 학회 발표문(2–3쪽) 작성 자료

브랜치 `paper/conference-2026` (2026-09-27 기준 `entry-map`의 모든 결과를 포함). 이 폴더는 **논문을 쓰는 데
필요한 것만** 모았다: 구성안, 수치(출처 명시), 그림, 데이터, 재현 명령. 수치는 모두 저장소 문서에서 그대로
옮겼고 이 폴더에서 새로 계산한 값은 없다. 원 문서가 우선한다.

---

## 1. 제목 후보

1. UWB 협업 다중 드론 점유지도 기반 구조대원 진입 가능성 판정
2. 불확실성 인지 다중 드론 협업 매핑과 사람형 이동 가능성 판정 — 실제 건물 스캔 기반 평가
3. Uncertainty-aware multi-drone occupancy mapping with human-like traversability for first-responder entry maps

## 2. 초록 초안 (수정해서 쓸 것)

> 카메라와 IMU만 탑재한 드론 3대가 UWB 상호 거리로 정합하여, GPS가 불가능한 실내에서 구조대원이 진입 경로를
> 판단할 수 있는 지도를 만드는 방법을 제안한다. 앵커 없는 드론 간 거리만으로 정합 불확실성(tr Σ)이 13배
> 감소했고, 깊이 불확실성 가중은 점유 과잉을 GT 대비 +47.8 %에서 −1.9 %로 줄였다(MILUV 실측). 점유지도에서
> 이동 가능 영역을 판정할 때 기존의 "단일 바닥 + 고정 높이 밴드" 방식은 스테레오(SGBM) 깊이의 바닥 흔들림을
> 장애물로 오인해 실제 건물 복도에서 통과 가능 영역을 전혀 찾지 못했다. 이를 발판·몸통 부피·자세(서기·숙이기·
> 기기)·한 걸음 높이로 판정하는 사람형 모델로 바꾸자, 실제 건물 스캔 위 합성 비행의 SGBM 지도에서 GT 도달
> 영역의 68.0 %를 안전 기준(오도달 1.73 %) 안에서 찾았다. 다만 추정 포즈 오차를 지도 생성에 물리적으로 반영하면
> 도달률이 6.5–8.9 %로 떨어져, 앵커 없는 협업의 포즈 드리프트 처리가 남은 핵심 과제임을 보인다.

## 3. 기여 (논문에 쓸 형태)

| # | 기여 | 핵심 수치 | 출처 |
|---|---|---|---|
| C1 | 앵커 없는 UWB 드론 간 거리만으로 협업 정합 | tr(Σ) 0.940 → 0.130 → 0.074 m² (C(N,2) 0→1→3, 13×); 정합 격차 0.0088 m (앵커 6개 조건 0.0074 / 앵커만 0.0116) | `reference files/RESULTS_SUMMARY.md` §7.2–7.3 |
| C2 | 깊이 불확실성 가중 점유지도 | occupied 총량 GT 대비 weighted −1.9 % vs uniform +47.8 % | 같은 문서 §6.6 |
| C3 | 사람형 이동 가능성 판정(진입지도 v2) | SGBM 지도 도달률 0.000(밴드 모델) → 0.680, 오도달 0.0173 | `RESULTS_traversability.md` §6, `RESULTS_sgbm.md` §14.4 |
| C4 | 실제 건물 스캔 기반 평가 규약과 그 한계 | 물리 규약에서 도달 0.089(ideal) / 0.065(SGBM) | `RESULTS_traversability.md` §7.1 |

2–3쪽이면 **C1 + C3를 본론**, C2를 한 문장, C4를 한계·향후 과제로 두는 구성을 권한다.

## 4. 권장 구성 (2–3쪽)

| 절 | 분량 | 내용 | 그림·표 |
|---|---|---|---|
| 1. 서론 | 0.3쪽 | 재난 실내 진입, GPS 불가, 카메라+IMU 드론, 협업 필요성, "점유지도 ≠ 진입 가능성" | — |
| 2. 시스템 | 0.5쪽 | VIO(VINS) → UWB 협업 정합(앵커 없음, Σ 추출) → 불확실성 가중 점유지도 → 진입 판정 | (그림 1: 파이프라인, 직접 작성) |
| 3. 사람형 이동 가능성 | 0.6쪽 | 발판, 몸통 원판, 자세 3종×폭 2종, 걸음 0.18 m, 틈 0.5 m, Dijkstra | 표 A |
| 4. 실험 | 0.4쪽 | MILUV 실측(C1·C2), 실제 건물 스캔(9.15 복도 41.6 m) 위 합성 비행 3대, SGBM | 그림 2(스캔), 그림 3(도면) |
| 5. 결과 | 0.7쪽 | 표 B(협업), 표 C(진입 판정), 음성 대조군, 물리 규약 | 그림 4(주 결과), 그림 5(대조군) |
| 6. 결론·한계 | 0.3쪽 | §9의 한계 문장 | — |

## 5. 방법 요약

**협업 정합.** 각 드론 VINS-Fusion 궤적 + UWB 드론 간 거리 → SE(3) 포즈 그래프(GTSAM), 중력 prior, 게이지 G1.
앵커 없음(재난 현장 상정). 마지막 포즈의 공분산 Σ를 지도 단계로 전달.

**불확실성 인지 점유지도.** 가중 log-odds, w = exp(−tr Σ/α)·exp(−σ_Z²/β), 안전 비대칭(l_occ +0.85, l_free
−0.40, τ_occ = l_occ). 0.10 m 격자. 확립된 것은 **깊이 항**뿐이고 포즈 항 가중은 음성 결과(§9).

**사람형 이동 가능성** (`DESIGN_traversability.md`, 구현 `covor/entry/traverse.py`):

표 A. 값은 기존 설계값 또는 외부 기준 (숙이기 높이만 새 값, 사전 고정)

| 항목 | 값 | 근거 |
|---|---|---|
| 발판 | occupied 복셀 바로 위가 free | 전역 바닥 없음 |
| 몸 폭 / 옆으로 | 0.70 m / 0.45 m | SCBA 착용 성인 |
| 자세 높이 | 서기 1.90 / 숙이기 1.40 / 기기 0.90 m | 기존 설계, 인체 치수 |
| 한 걸음 | 0.18 m (0.10 m 격자에서 2행) | 건축 기준 단높이 상한 |
| 틈 건너기 | 관측 안 된 바닥 0.5 m까지 | 조심스러운 보폭 (사후 변경, 등록 후 실행) |
| unknown | 어디서나 막힘 | 안전 원칙 |
| 경로 | Dijkstra, 자세·폭 비용 계수 | 비용은 경로 선택만, 도달 여부 불변 |

## 6. 실험 설정

- **MILUV 실측** (C1·C2): 드론 3대(ifo001–003), UWB, mocap GT. 조건 A(1대)…C(3대 anchor-free), D·E(앵커).
- **실제 건물 스캔 기반 합성 비행** (C3·C4): Polycam 스캔(2026-09-15, 2층 중앙 계단·201호·복도, 41.6 × 11.4 m)
  → GT 복셀(0.10 m). 스캔의 바닥 이음매·표류를 장면에서 보정(`scripts/scan/repair_floor.py`, `RESULTS_sgbm.md`
  §14) → 장면 `corridor915f`. 드론 3대 120 s, 고도 1.10/1.25/1.40 m, 복도 길이 방향 3구역, 합성 VIO·UWB,
  앵커 없는 융합(ATE 0.121 / 0.093 / 0.085 m, σ_xy 중앙값 0.752 m). 깊이: SGBM(스캔 텍스처로 렌더한 스테레오,
  MILUV 기본 파라미터, |ΔZ| 중앙값 0.005–0.024 m) 및 ideal(참고).
- **평가**: 같은 판정 모델을 GT 격자와 지도에 똑같이 적용. 진입점 = 남단 발사점 (−2.35, −19.55) m.
  reach recall = |지도 도달 ∩ GT 도달| / |GT 도달|, false-reach = 지도는 도달, GT는 어떤 자세로도 설 수 없는 칸 비율.

## 7. 결과 표

표 B. 협업 인과 사슬 (MILUV, anchor-free) — `RESULTS_SUMMARY.md` §7.2

| 조건 | C(N,2) | tr(Σ) 중앙 [m²] | IoU@1vox | false-free |
|---|---|---|---|---|
| A (1대) | 0 | 0.940 | 0.306 | 5.978 % |
| B (2대) | 1 | 0.130 | 0.537 | 4.999 % |
| C (3대) | 3 | 0.074 | 0.518 | 5.268 % |
| D (3대+앵커) | 3+앵커 | 0.0015 | 0.584 | 4.426 % |

ρ(tr Σ, false-free) = +0.900. 융합 정확도(앵커 조건): VINS 0.171/0.202/0.242 m → 0.079/0.085/0.061 m (2.1–4.0×, §5.5).

표 C. 진입 판정 — 9.15 복도 보정 장면 (`paper/data/`)

| 지도 (깊이) | 밴드 모델 walk recall | 사람형 reach recall | 사람형 IoU | 사람형 false-reach |
|---|---|---|---|---|
| ideal | 0.609 | 0.675 | 0.661 | 0.0067 |
| **SGBM** | **0.000** | **0.680** | 0.633 | 0.0173 |

두 recall은 정의가 다르므로(밴드 모델: walk 밴드 통과 가능 셀, 사람형: 진입점에서 도달한 컬럼) **방향만** 비교한다.
밴드 모델이 SGBM에서 0인 원인: SGBM 바닥 점이 두 복셀 행에 걸쳐(5,484 / 3,965) 단일 바닥 규칙이 낮은 행을 골라
실제 바닥이 밴드 안 장애물이 됨(`RESULTS_sgbm.md` §14.5). 3D 형상은 SGBM도 ideal 수준(precision 0.922, IoU@1 0.605 vs 0.610).

표 D. 음성 대조군 (프레임별 위치 흔들림, ideal) — `paper/data/traverse_negative_control.csv`

| σ | 0 | 0.20 m | 0.40 m |
|---|---|---|---|
| reach recall | 0.675 | 0.298 | 0.013 |

표 E. 평가 규약의 영향 (§9 한계와 함께 쓸 것) — `paper/data/traverse_physical_protocol.csv`

| 규약 | ideal | SGBM |
|---|---|---|
| 추정 포즈로 렌더·적분 (표 C) | 0.675 | 0.680 |
| **참 포즈로 렌더, 추정 포즈로 적분 (물리적)** | **0.089** | **0.065** |

## 8. 그림과 캡션 초안 (`paper/figures/`)

| 파일 | 캡션 초안 |
|---|---|
| `floorplan_2F_scanned_region.png` | 대상 건물 2층 평면도. 스캔 범위는 중앙 계단, 201호, 복도. |
| `fig_scan_views.png` | 실제 건물 스캔(텍스처 메시)의 눈높이 렌더: (A) 로비, (B) 계단실 출입구, (C) 바닥 메시 이음매, (D–F) 복도·교실 문. |
| `fig_scan_vs_band_model.png` | 스캔 평면(천장 제거) 위 밴드 모델 판정. 가짜 턱(①)이 복도를 끊고, 계단실(②)·닫힌 문(③)·미스캔 경계(④⑤)를 표현하지 못함. |
| `fig_traverse_main.png` | 사람형 판정 결과: GT / SGBM / ideal / 흔들림 5 cm. 초록 서기, 노랑 숙이기, 주황 기기, 파랑 옆으로. |
| `fig_negative_control.png` | 흔들림 0 / 0.20 / 0.40 m에서 도달 영역 감소. |
| `fig_physical_protocol.png` | 참 포즈로 렌더하고 추정 포즈로 적분한 지도: 복도 전체를 덮지만 설 수 있는 칸이 파편화. |
| `fig_miluv_weighted_vs_uniform.png` | MILUV 실측 3대: 불확실성 가중 vs 표준(w = 1) 점유지도, z = 0.8 m 단면. |
| `fig_room909_entry_map.png` | 다른 장면(방 909 스캔) 위 진입지도, 밴드 모델, 0.05 m. |

**인터랙티브 3D 뷰어** `web/traverse_3d_corridor915f.html` (더블클릭으로 열림): GT / SGBM(기존 규약) /
SGBM(물리 규약) / ideal. 발표 화면 캡처용. 타일은 발판 높이에, 색은 도달 자세.

## 9. 반드시 적을 한계 (정직한 서술 — 문장 그대로 써도 됨)

1. **평가 규약의 상한.** 합성 지도는 추정 포즈에서 렌더하고 같은 포즈로 적분했다. 이 규약은 포즈 오차가 지도에
   남기는 번짐을 구조적으로 지우므로 표 C는 상한이다. 물리적 규약(표 E)에서는 앵커 없는 협업의 절대 위치 오차
   (드론별 0.22 / 0.62 / 1.46 m)가 지도를 파편화한다.
2. **사후 변경.** 사람형 모델의 판정(성립)은 세 번의 사후 변경(틈 건너기, 바닥 지지 GT, 대조군 재정의)을 각각
   실행 전에 등록한 뒤 얻었다(`DESIGN_traversability.md` §7–9).
3. **포즈 불확실성 가중은 음성 결과.** 포즈 항 가중은 안전 비대칭을 무력화해 false-free를 늘렸다
   (`RESULTS_SUMMARY.md` §8). 확립된 가중은 깊이 항뿐이다.
4. **GT는 스캔에서 왔다.** 스캔의 바닥 이음매·표류는 보정했고(§14), 둘러싸임 기준 밖의 층계참은 바닥 지지 규칙으로
   GT에 넣었다. 닫힌 문, 미스캔 공간은 GT에서도 통과 불가다.
5. **SGBM은 낙관적 근사.** 좌우 영상이 같은 스캔 텍스처라 노출 차·스페큘러·모션블러가 없다.
6. **장면 1개.** 진입 판정 결과는 한 층·한 구간(9.15 복도)에서만 나왔다.
7. **틈 규칙 의존.** SGBM 도달 면적의 63 %(60.7 m² 중 38.4 m²)는 관측 안 된 바닥을 0.5 m 이내로 건너
   도달한 칸이다(`RESULTS_traversability.md` §9). SGBM 지도가 바닥을 드물게 관측하기 때문이다.

## 10. 재현 (필요 시; 결과 파일은 이미 커밋되어 있음)

```bash
PY=/src/gs25058/miniconda3/envs/covor/bin/python
# 장면 보정과 GT
$PY scripts/scan/repair_floor.py --obj "meshes/2026-09-15/2026. 9. 15.obj" \
    --mesh-config results/synth_corridor915/mesh_config.json --out "meshes/2026-09-15/2026. 9. 15.floorfix.obj"
$PY scripts/synth/build_gt_voxel.py --obj "meshes/2026-09-15/2026. 9. 15.floorfix.obj" \
    --name corridor915f --seed-xyz -2.05 0 1.2 --floor-z 0.05
$PY scripts/entry/gt_floor_supported.py --name corridor915f
# 합성 비행과 지도
$PY scripts/synth/make_dataset.py --name corridor915f --seed 0 --zone-axis y --duration 120
for r in ifo001 ifo002 ifo003; do $PY scripts/entry/render_depth_cache.py --name corridor915f --robot $r; done
$PY scripts/entry/dump_fused_map.py --name corridor915f --cond C_3drone --coverage ii_all_cams \
    --arm depth_only --stride 4 --mode sgbm --depth-cache          # SGBM
$PY scripts/entry/dump_fused_map.py --name corridor915f --cond C_3drone --coverage ii_all_cams \
    --arm depth_only --stride 4                                    # ideal (+ --corrupt jitter0.2 / --render-truth)
# 판정
D=results/synth_corridor915f/entry
$PY scripts/entry/run_traverse.py --gt results/synth_corridor915f/gt_voxel_fs.npz \
    --map $D/map3d_C_3drone_ii_all_cams_depth_only_clean_sgbm_s4.npz:SGBM \
    --map $D/map3d_C_3drone_ii_all_cams_depth_only_clean_s4.npz:ideal \
    --entry=-2.35,-19.55 --out $D/traverse_results_gtfs.csv --fig $D/traverse_corridor915f_gtfs.png
# 스캔 사진
$PY scripts/scan/render_scan_views.py --mesh-config results/synth_corridor915/mesh_config.json --out paper/figures
```

메시(`meshes/`)는 용량 때문에 git에 없다. 출처와 md5는 `meshes/PROVENANCE.json`.

## 11. 출처 문서 지도

| 주제 | 문서 |
|---|---|
| 연구 목표·방법 전체 | `reference files/연구계획서_불확실성인지_다중드론_점유지도.md` (gitignored, 로컬) |
| 융합·점유지도·협업 인과 (MILUV) | `reference files/RESULTS_SUMMARY.md` |
| 합성 규약·스캔 GT | `RESULTS_synth.md` |
| 밴드 모델 진입지도 (방 909) | `RESULTS_entry.md`, `reference files/DESIGN_entry_map.md` |
| 복도 장면·SGBM·바닥 보정 | `RESULTS_sgbm.md` (§14가 보정 장면) |
| 사람형 판정 설계·결과 | `DESIGN_traversability.md`, `RESULTS_traversability.md` |
| 사전등록 | `PREREG_*.md` |
