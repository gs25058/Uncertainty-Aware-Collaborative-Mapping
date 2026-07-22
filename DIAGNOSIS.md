# UWB–VO 융합 악화 진단 결론 (default_3_zigzag_0)

## 질문
mono-VO에 UWB 레인지를 팩터 그래프로 융합하면 ATE가 VO보다 나빠진다
(VO 0.091 / 0.129 / 0.284 m → 융합 0.4~0.6 m). 그래프 에러(목적함수)는 줄지만
mocap 대비 정확도는 악화. **원인이 무엇인가?**

## 두 가설 판별 + 3요인 분해

핵심 비교(prior_every=8 게이지 고정, UWB-only, 세 로봇 모두 앵커링된 상태):

| 실행 | ATE [ifo001 / 002 / 003] | mean | 비고 |
|---|---|---|---|
| **VO 베이스라인** (Sim3) | 0.091 / 0.129 / 0.284 | 0.168 | 목표 |
| 실측 range + 바디중심 + const bias | 0.387 / 0.352 / 0.397 | 0.379 | 기존 악화 |
| gt_range + 바디중심 | 0.297 / 0.266 / 0.486 | — | 완벽 측정 |
| gt_range + moment-arm | 0.251 / 0.318 / 0.240 | 0.270 | 완벽 측정 + 정확 모델 |
| **gt_range + moment-arm + height** | **0.147 / 0.202 / 0.130** | **0.160** | ≈ VO, robot3 대폭↑ |
| 실측 + moment-arm + height | 0.290 / 0.229 / 0.257 | 0.259 | 실무 최선 |

### H1 (평가 프로토콜) — 기각
원래 `run_covor.py`는 VO를 Sim(3), fused를 SE(3)로 평가했다(비대칭). 그러나
**동일 `evaluate.align`으로 양쪽에 같은 정렬을 적용**해도(Test 1) fused는 Sim(3)
정렬에서조차 VO보다 나쁘다(0.38 vs 0.09). fused는 메트릭이라 SE3(0.387)≈Sim3(0.376).
→ 악화는 평가 산물이 아니라 실재한다.

### H2 (관측 모델/모멘트암) — 부차적
- `gt_range - 바디중심거리`의 태그별 mean ≈ 0, std ≈ 0.15 m: **레버암은 상수 편향이
  아니라 자세 의존 산포**. 6개 앵커 모두 mean≈0으로 균일 → **앵커 좌표 오류 없음**.
- 지금까지 쓰던 `range_bias=0.14`는 robot1-only 버그 서브셋에 맞춰진 **과보정**
  (실제 계통 편향 ~0.05 m).
- 모멘트암 팩터(`ma_*_range_factor`, 안테나 `q=p+R·l`, 야코비안
  `∂r/∂ξ=[−uᵀR[l]ₓ, uᵀR]`, 단위테스트 통과)는 ATE를 ~0.025–0.05 m만 개선. 주원인 아님.

### 진짜 주원인 (2가지)
1. **수직 기하(VDOP)**: 앵커 6개가 모두 ~1.7 m 높이 → **완벽한 gt_range로도 수직
   위치를 못 정함**. gt+MA 0.270 → **+height 0.160**(수직 zRMSE 0.30→0.05)으로 회복.
   → 하향 레이저 **height 팩터가 해결책**.
2. **실측 UWB 노이즈**: `range−gt_range` mean +0.05, **std 0.22 m, 아웃라이어
   0.9 m**. 이 노이즈가 robots 1,2의 VO 오차(0.09, 0.13)보다 **크다** → 융합하면
   반드시 악화. robot3(VO 0.284 > 0.22)만 이득.

## 결정적 증거
**gt_range(완벽) + moment-arm(정확) + height = mean 0.160 ≈ VO(0.168)**, 그리고
**robot3는 0.284 → 0.130으로 대폭 개선**. → 파이프라인은 근본적으로 정상이며,
남는 격차는 전적으로 **실측 UWB 품질**이다.

## 일반 원리
**로봇의 VO 오차 > UWB 유효정확도(~0.22 m)일 때만 융합이 이득.**
- 이 시퀀스: robot3만 해당(이득), robots 1,2는 VO가 이미 더 정확(손해).
- 이 시퀀스는 융합 검증에 **너무 쉽다**. 세 로봇 모두 VO 드리프트 > 0.22 m인
  고속/장궤적 시퀀스가 필요(→ 후속 작업, 현재 다운로드 보류).

## 권장 구성 (이 시퀀스에서 손해를 최소화하는 최선)
```python
Cfg(use_height=True, use_moment_arm=True, bias_mode="off",
    range_sigma_floor=0.3, huber_k=1.0, prior_every=8)
```
- `use_height`: 수직 VDOP 보완(필수). `use_moment_arm`: 레버암 std 제거.
- `bias_mode="off"`: 계통 편향이 사실상 없으므로 const 0.14 제거.
- 그래도 이 시퀀스에선 fused(mean 0.26) < VO(0.168)를 못 넘음 — 데이터 한계.

## 재현
```bash
conda activate covor
python scripts/test_jacobians.py        # 모든 팩터 야코비안 단위테스트
python scripts/test1_alignment.py       # H1: 정렬 2x2
python scripts/test2_aux.py             # H2: 레버암/앵커 데이터 진단
python scripts/test2_gtrange.py         # H2/geometry: gt_range·moment-arm 조합
# 모든 수치는 diagnosis_results.csv 에 append(즉시 flush)
```
