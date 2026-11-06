# 모의 연기·단일 재관측 실험 (smoke_20261103)

기준: `paper/conference-2026`, `dffbb8e`. 기존 데이터·결과·핵심 SLAM/지도/판정 코드는 보존했다.
**[전체 결과·재실행 명령·선행연구와 변경점](reports/smoke_20261103/RESULTS.md)**.

실제 실행한 범위:

- MILUV `default_3_zigzag_0`: 3대 × 48쌍, 고정 융합 포즈, 무처리 + 균일/비균일 감쇠 각각 3강도.
  중간 비균일 감쇠는 추가 시드 1·2에서도 반복했다.
- `corridor915f`: 3대 × 100쌍. 참 포즈에서 스캔 텍스처 관측을 생성하고 추정 포즈로 적분했다.
  같은 24개 후보에서 nearest / unknown / route가 각각 스테레오 한 쌍을 추가했다.
- VIO: ifo001 첫 20초의 무처리·강한 균일 감쇠, 각 299쌍과 기존 IMU.
  원래 입력·출력 경로를 쓰지 않고 실행했으며 역순 반복도 수행했다.

| 관찰 | 측정 결과 |
|---|---|
| MILUV 강한 비균일 감쇠 | 유효 깊이 69.84% → 66.24%, 공동 유효 픽셀 평균 깊이 차이 0.227 m |
| 같은 조건의 점유지도 | 무처리 지도와 occupied exact IoU 0.258, free exact IoU 0.760 |
| 복도 중간 균일 감쇠 + 경로 중심 한 쌍 | 통과 확인 목표 3 → 4개, nearest·unknown은 3개 유지 |
| 복도 중간 비균일 감쇠 + 경로 중심 한 쌍 | 도달률 9.07% → 10.43%, 통과 확인 목표는 4개 유지 |
| 전체 7조건의 경로 중심 통과 확인 수 | 증가 1조건, 감소 1조건, 나머지 변화 없음 |

목표 단위 오확인은 추가되지 않았다. 다만 22개 목표 중 GT 도달 16개, 관측된 고형물로
막힌 목표 1개라 안전성 검증의 표본은 작다. 면적 증가와 목표 통과 확인은 구분한다.
관측된 발판으로 이어지는 경로가 없거나 지정 입구를 확인할 수 없으면 **미확인**이다.
기존 판정기의 0.5 m 미관측 바닥 건넘 도달 지표도 별도로 보존했다.
강한 비균일 조건의 입구 스냅 0.224 m가 최종 대조에서 발견돼, 확인 판정의 입구 스냅을
금지하는 보수적 후처리를 추가했다. 기존 수치와 보수적 수치를 모두 남겼다.

이것은 적외선 영상의 **거리 의존 감쇠·대비 저하 근사 실험**이며 실제 화재 검증이 아니다.
beta는 설정값으로만 기록한다. 깨끗한 SGBM 깊이는 합성에만 사용했고, 모든 연기 지도는
연기 영상에서 다시 계산한 SGBM 깊이만 적분했다. 깨끗한 MILUV 지도는 비교 기준이다.
깊이 무효 영역의 보완 흔적과 물리적 포즈 오차가 결과에 남는다.

검토용 산출물은 `reports/smoke_20261103/`, 큰 원시 캐시·포즈·3D 격자는
`results/smoke_20261103/`에 있다. 새 구현은 `covor/{smoke,reobserve}.py`와 `scripts/smoke/`.
작은 무처리 실행의 기존 복셀 log-odds 일치, 새 테스트 10개와 기존 테스트 20개,
저장 지도 28개의 지표 재계산·입력 해시 대조를 수행했다.

주요 파일:

- [MILUV CSV](reports/smoke_20261103/miluv/summary.csv)
- [복도·재관측 보수적 CSV](reports/smoke_20261103/corridor/summary_conservative.csv)
- [목표별 보수적 판정](reports/smoke_20261103/corridor/goals_conservative.csv)
- [원래 스냅 규약 CSV](reports/smoke_20261103/corridor/summary.csv)
- [VIO 반복 결과](reports/smoke_20261103/vio/all_runs.csv)
- [영상·깊이 비교](reports/smoke_20261103/miluv/examples/heterogeneous_strong.png)
- [3D 점유지도 비교](reports/smoke_20261103/miluv/occupancy_3d_comparison.png)
- [진입지도 비교](reports/smoke_20261103/corridor/entry_comparison.png)
