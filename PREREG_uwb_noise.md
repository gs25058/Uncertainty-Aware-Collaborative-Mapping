# 사전등록 — UWB 노이즈 주입 스윕

**실행 전에 기록한다.** `DIAGNOSIS.md`의 정성 명제("VO 오차 > UWB 유효정확도일 때만
융합이 이득")를 정량 곡선으로 바꾼다. 시퀀스가 1개뿐인 한계를 합성 노이즈로 부분 우회한다.

작성 2026-09-03 · `default_3_zigzag_0` · 근거 `reference files/RESULTS_SUMMARY.md`

---

## 1. 관측 생성 모델

`r_sim = gt_range + b(kind) + N(0, σ) + outlier`

`gt_range`는 안테나-대-안테나 참거리(sub-mm 검증 완료)이므로 `use_moment_arm=True`와
정합한다. **타임스탬프와 연관(association)은 실측 그대로 두고 값만 바꾼다** — ORB 시대
실패 원인이 UWB 연관 89.9% 손실이었으므로(부록 A-2, §3.3) 연관 구조를 건드리면 교란된다.

### 실측에서 가져온 상수

| kind | bias | 총 std | **코어 std** | P(\|e\|>0.5) |
|---|---|---|---|---|
| anchor | +0.0871 | 0.2457 | **0.1288** | 0.0861 |
| inter | +0.0035 | 0.1660 | **0.1278** | 0.0217 |
| 전체 | +0.0537 | 0.2211 | **0.1296** | 0.0603 |

이상치는 `+U(0.5, 1.8)` m — **단측(양수)** 이다. 실측 `|e|>0.5` 꼬리의 **99.8%가 양수**이고
크기가 0.50~1.78 m이기 때문이다.

> ⚠ **σ 축은 가우시안 코어의 σ이지 총 std가 아니다.** 실측 총 std 0.2211에는 이상치가
> 이미 포함돼 있으므로, σ=0.22를 넣고 이상치를 또 얹으면 실측보다 커진다(검증: anchor
> std 0.41 vs 실측 0.25). 따라서 **곡선 위 "실측 UWB" 마커는 σ ≈ 0.13(코어)** 에 찍는다.
> σ 격자는 그대로 둔다.

## 2. 스윕 축 (960 run)

- **σ** ∈ {0, 0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.50} (8)
- **bias** ∈ {off, 실측 per-kind} (2)
- **outlier** ∈ {off, 실측률} (2)
- **seed** ∈ 5개 → mean ± std
- **조건** ∈ {C (anchor-free, inter 3쌍), D (inter 3쌍 + anchor)} (2) — `Cfg.inter_pairs` /
  `Cfg.anchor_robots` 로 지정(`--drones`는 출력 필터일 뿐, 부록 A-13)
- **range_sigma_floor** ∈ {matched, 0.05, 0.3} (3)
  - `matched` = `max(σ_inject, 0.01)` — 하한 0.01은 σ=0에서 가중치 발산을 막는 수치 하한
  - `0.05` = 현행 실험값(부록 B 정정본)
  - `0.3` = ORB 시대 값, "노이즈를 과대 추정할 때" 사례

`8×2×2×5×2×3 = 960`. 단일 스레드 × **20 병렬** → 약 3 h.

## 3. 무수정 (부록 B 정정본)

`sigma_odo_trans/rot`, `sigma_tilt`, `use_gravity_prior`, `gauge_mode="component"`,
`init_yaw_only`, `vins_stride=2`, `range_tol=0.07`, `range_motion_sigma=True`,
`use_moment_arm=True`, `bias_mode="off"`, `robust=True`, `huber_k=1.0` — 전부 그대로.
스윕 축(σ, bias, outlier, floor, 조건)만 움직인다.

## 4. 사전 예측

- **P1** bias=0, outlier=off일 때 융합 ATE는 σ에 **단조 증가**하고, VINS 단독선을
  **ifo001(0.171)이 가장 먼저, ifo003(0.242)이 가장 늦게** 넘는다.
- **P2** bias를 켜면 σ→0에서도 융합 ATE가 0으로 가지 않고 **bias 수준(~0.087 m 부근)에서
  바닥**을 친다. 바닥을 정하는 것은 RMSE가 아니라 계통 bias다(§3.3의 결론과 일치).
- **P3** outlier를 켜도 Huber(k=1.0)가 대부분 흡수해 곡선이 크게 변하지 않는다.
  **크게 변하면 Huber 설정을 의심할 근거**가 된다.

## 5. 판정·보고 규칙

- 교차점 σ\*(융합 ATE = VINS 단독 ATE)를 로봇별로 보고. 교차하지 않으면 "교차 없음"으로 보고.
- 예측과 달라도 **파라미터를 조정해 다시 돌리지 않는다.** 그대로 보고한다.
- 실측 range를 섞어 쓰지 않는다 — 전부 `gt_range` 기반 합성이다.
- seed 5개의 mean ± std를 함께 보고한다(단일 seed 값으로 결론내지 않는다).

## 6. 산출물

1. `results/results_uwb_noise_sweep.csv` — `condition, sigma, bias_mode, outlier,
   sigma_floor, seed, robot, ATE, tr_sigma_med, align_gap, n_range_factors` (append + flush)
2. `results/fig_noise_ate.png` — x=σ, y=ATE(로봇 3선, ±std 밴드) + VINS 단독 3 수평선,
   교차점 σ\* 표시, 실측 코어 σ 마커
3. `results/fig_noise_bias.png` — bias on/off 대비(σ→0 바닥 차이가 보이게)
4. 요약 md — σ\* 3개, bias 바닥값, 실측 UWB의 곡선상 위치
