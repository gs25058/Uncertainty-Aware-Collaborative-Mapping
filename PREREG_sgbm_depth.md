# 사전등록 — SGBM 깊이에서도 진입지도 결론이 유지되는가 (Part 2)

작성 2026-09-23, **실행 전**. 목적은 "ideal 깊이에서 세운 진입지도 결론이 현실적
스테레오 깊이에서도 유지되는가"의 검증이지, SGBM 숫자를 좋게 만드는 것이 아니다.

## 1. 가설

ideal depth를 SGBM depth로 바꾸면
(i) 텍스처 없는 영역에서 unknown 컬럼이 늘고,
(ii) 경계 번짐으로 문·통로의 clearance가 계통적으로 줄어 **false-passable은 감소하되
passable recall이 떨어진다.**
판정은 결론 유지 여부이지 성능 경쟁이 아니다.

## 2. 비교쌍 — 다른 축은 전부 동결

| | ideal (Part 1 기준선, `RESULTS_sgbm.md` §7) | SGBM |
|---|---|---|
| 장면 | corridor915, res 0.10, 바닥 평탄화 | 동일 |
| 데이터셋 | `synth_corridor915` seed 0, 120 s, zone_axis y | **동일 파일** (재생성 없음) |
| 융합 | C_3drone, anchor-free | **동일 포즈** (재융합 없음 — 같은 npz 포즈를 씀) |
| 지도 | coverage ii, arm depth-only, stride 4 | 동일, `mode="sgbm"` |
| 진입지도 | w 0.70, k_sigma 0, 공유 문 (−2.35, −19.55) 스냅 1.5 m | 동일 |

두 arm의 차이는 `DepthRenderer.frame(mode=...)` 한 인자뿐이다.

## 3. SGBM 파라미터 — 1세트, 이후 변경 금지

`covor.occupancy.DepthCfg` 기본값 그대로: `num_disp 96, min_disp 0, blockSize 7,
uniqueness 10, speckle 100/2, disp_sigma_px 1.0`, 베이스라인 0.050 m(MILUV infra1/2),
848×480. **근거**: MILUV 실데이터 파이프라인(`scripts/build_occupancy.py`)이 쓰는 값과
동일하다. 합성 arm이 실데이터 arm과 같은 매처를 쓰는 것이 이 실험의 요점이다.
`RESULTS_synth.md` §2.4가 방 909에서 같은 설정으로 MAE 0.122 m를 기록했다.

## 4. 구현 전제 — 판정 코드 무수정, 배선만

저장소에 sgbm 배선이 없다(Part 0 보고): `MG.raycasting_scene`은 지오메트리를 하나만
추가해 `geometry_ids`가 전부 0이고, `MG.load`는 `force="mesh"`로 재질을 합치며,
`run_experiment --mode sgbm`은 `textures=`를 넘기지 않아 즉시 죽는다.
`scripts/entry/`에 다중 지오메트리 scene + `load_textures` 배선을 추가하고
`dump_fused_map.py`에 `--mode sgbm`을 연다. `covor/synth/render.py`·`covor/occupancy.py`
무수정.

**배선 값 검증(P0 전에)**: 다중 지오메트리 scene의 `ideal` 깊이가 단일 메시 scene의
`ideal` 깊이와 **비트 단위로 동일**해야 한다(같은 표면을 다르게 조립했을 뿐이므로).
아니면 배선이 표면을 바꾼 것이고 이후 전부 무효.

## 5. 판정 (전부 실행 전 고정, 튜닝 금지)

**P0 (구현 유효성).** 실제 실행의 전 프레임에서, ideal·SGBM 모두 유효한 화소의
|Z_sgbm − Z_ideal| **중앙값 < 0.10 m (1 voxel)**. Part 0의 7프레임 프로브는 중앙값
+0.015…+0.061 m였다. 실패 시 이후 판정 전체 무효.

**S1.** walk 밴드 **unknown 컬럼 수**: SGBM > ideal.
> **미리 적어두는 주의**: Part 0 측정에서 SGBM이 잃는 화소의 **91.4 %가 좌측 96열
> disparity 탐색 여백**이다 — 텍스처 실패가 아니라 실제 스테레오 카메라도 갖는 고정
> 기하다. 따라서 S1은 거의 확실히 통과하며 **기전 확인이지 증거가 아니다.** 부호가
> 장면에 따라 뒤집힌다는 것도 안다(방 909의 열린 쪽에서는 SGBM 유효 화소가 더 많았다).

**S2.** walk 밴드 **false-passable rate**: SGBM ≤ ideal.
> **미리 적어두는 주의**: 이 지표는 세 장면·다섯 실행에서 손상에 한 번도 증가하지 않았고
> 게이트에서 제외됐다(`RESULTS_sgbm.md` §8). S2 통과는 약한 정보다. 브리프대로 유지한다.

**S3.** walk 밴드 **passable recall 감소율 ≤ 30 %**: recall_sgbm ≥ 0.7 × recall_ideal.
기준선 recall 0.2378 → 문턱 **0.1665**. 이것이 "결론이 유지되는가"의 실질 판정이다.

**전부 충족 → 성립. 하나라도 위반 → 불성립. 부분 성립 없음.** 두 번째 파라미터 세트,
두 번째 문턱 없음.

## 6. 한계 — 결과 해석에 이 문장을 그대로 싣는다

> 좌우 뷰가 **같은 baked 텍스처**를 렌더하므로 노출 차·스페큘러·모션블러가 없어 실제보다
> 매칭이 유리하다. 따라서 이 실험의 SGBM은 현실의 하한이 아니라 **낙관적 근사**이며,
> "SGBM에서도 유지된다"는 "이상적 조명·정적 장면의 SGBM에서도 유지된다"로 읽어야 한다.

## 7. 보조 (판정 불참)

- unknown 맵 차분 그림 (SGBM − ideal), walk 밴드.
- 복도 4 m 슬랩별 최대 clearance 표 (ideal / SGBM / GT).
- 3D 지표(precision·recall·false-free) ideal vs SGBM, 관측성 도메인.
- 재정의 게이트(§8)를 SGBM arm에도 적용할지는 **범위 밖** — 여기서는 손상 없이 두 arm을
  비교한다.

## 8. 범위 밖 (`RESULTS_sgbm.md` "답 못한 것"에 그대로)

0.05 해상도, 3D 뷰어 uint16, 방 909 SGBM(텍스처는 있으나 별도 실행), 실사 조명 효과.
