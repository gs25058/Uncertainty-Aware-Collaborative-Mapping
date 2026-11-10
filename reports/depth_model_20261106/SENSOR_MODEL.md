# 낙관적 깊이 관측 모델

모델 명칭: **RealSense D455급의 공개 사양을 참고한 낙관적 깊이 관측 모델과 통제된 관측 저하 조건**. 실제 D455의 모든 오차·실제 화재 연기·SGBM 영상 처리를 재현하지 않는다. 목적은 명시한 관측 모델에서 관측 배정의 효과를 검증하는 것이다.

## 공식 자료와 실험 가정

| 항목 | 사용값 | 출처/해석 |
|---|---|---|
| 양안 간격 | 0.095 m | [450 시리즈 공식 안내](https://www.realsenseai.com/product-family/450-series/)의 95 mm |
| depth FOV | 약 87°×58° | 같은 공식 안내의 명목 시야각; 왜곡 없는 pinhole로 근사 |
| 유효 Z-depth | 0.6–6.0 m | 공식 안내의 ideal range를 실험의 hard Z 범위로 채택 |
| 명목 고해상도 모드 | 1280×720, 30 FPS | [D400 datasheet Table 4-3](https://realsenseai.com/wp-content/uploads/2025/08/Intel-RealSense-D400-Series-Datasheet-August-2025.pdf)의 Z16 모드 |
| 고속 모드 예 | 848×480, 최대 90 FPS | 같은 표. 최대 해상도와 최대 FPS를 동시 지원한다고 해석하지 않음 |
| 시뮬레이션 깊이 격자 | 320×180 | 좋은 센서의 깊이 출력에서 성긴 광선을 샘플링하는 계산 근사. 저해상도 양안 영상으로 stereo를 계산하는 모형이 아님 |
| 지도 적분 속도 | 4 Hz; 별도 1 Hz | 실험의 실제 생성·적분 속도. 하드웨어 30 FPS의 중간 프레임은 생성하지 않음 |
| normal sigma | 0.002+0.001 Z² m | 사용자 지정 낙관적 가정. 제조사 보장 확률분포가 아님 |
| bias/drop/outlier/상관/연기 | 아래 설정 | 모두 실험 가정이며 제조사 인증값이 아님 |

공식 [D400 accuracy 안내](https://support.realsenseai.com/hc/en-us/articles/360059129453-Depth-accuracy-for-Intel-RealSense-D400-Series-Cameras)는 D455 4 m 이내·중앙 80% ROI·HD·적절한 통합/보정 조건의 정확도와 fill rate를 기술한다. 이 조건부 사양을 모든 픽셀·거리·표면의 분포로 환산하지 않았다. [튜닝 안내](https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/)의 높은 해상도에서 얻은 depth를 받은 뒤 decimation하는 구분을 참고했다. 이 안내의 예시는 D415/D435이며 우리 가상 센서의 오차 상수를 제공하는 자료가 아니다.

## 기하·프레임·관측 가능성

새 백엔드는 `covor/explore/depth_observation.py`다. `sensor.backend`는 `legacy_sgbm`(기본값 보존), `ideal_depth`, `depth_error_model` 중 하나다. 새 두 백엔드는 SGBM 객체와 좌우 영상을 만들지 않는다. 기존 `--ideal-depth`는 기존 backend에서 옛 진단 동작을 유지한다. 새 backend에서도 명시적 ideal override가 가능하지만 주 실행은 설정의 backend 선택을 사용한다.

K는 fx=W/[2tan(87°/2)], fy=H/[2tan(58°/2)], cx=W/2, cy=H/2다. 모든 센서 ray·역투영·후보 FOV·unknown/entry 가시성이 같은 K와 0.095 m baseline을 쓴다. world←left-camera T는 기록한 실제 body/yaw/짐벌에서 계산한다. optical Z와 광선 길이는 다르다. 범위 판정은 0.6≤Z≤6.0 m, 역투영은 optical Z, 연기는 실제 광선 길이로 계산한다. 지도·표면 적분의 Euclidean ray cap은 FOV 대각선의 최대 광선 길이로 도출한다(약 8.9 m). 따라서 Z≤6인 주변부를 옛 4.5/5 m 상수로 버리지 않는다.

왼쪽에서 첫 표면만 raycast한다. 오른쪽 카메라에서 같은 표면 좌표로 ray를 쏘아, 오른쪽 FOV 안이고 첫 교차가 같은 표면인지 0.2 mm 수치 허용으로 확인한다. 다른 드론의 반경 0.14 m 구 가림은 양쪽에서 적용한다. SGBM 시차 탐색 폭에 의한 고정 64픽셀 손실은 없다. 기존 0.35 m 동료 제외와 관측 지도·표면 적분을 유지한다.

기하 가시성은 오염 전 첫 표면의 관측 가능성이다. 오염 후 양의 오차가 표면 뒤 좌표를 만들더라도 그것은 오류 측정이며 뒤 구조를 관측한 것이 아니다. 이 오류를 참 형상으로 다시 잘라 교정하지 않는다. 오염 Z가 범위를 벗어나면 NaN/invalid이고 free-ray를 만들지 않는다.

## 오차와 상관성

실행 전에 `configs/sensors/d455_inspired_optimistic.json`에 수치를 고정했다. 거리 z는 optical Z[m]다.

- 정상 σz=0.002+0.001 z²: 1/2/4 m에서 3/6/18 mm.
- 드론별 상수 bias ~N(0,1 mm²). 실행 동안 유지한다. 이는 조건부 랜덤 표준편차 σz와 별도이며 RMS에 함께 기여한다.
- 정상 drop=0.005, outlier=0.0005. 기하학적으로 양안 관측 가능한 픽셀을 분모로 한다.
- outlier 성분은 부호 양/음과 크기 0.05–0.20 m다. 정상 오차를 outlier 성분으로 대체하고 작은 고정 bias를 더한다.
- edge: 인접 depth 차이 >max(0.10 m, 0.05×작은 Z), 유한/비유한 경계도 포함. 한 픽셀 팽창한다. 이미지 바깥을 이유로 탐색 손실을 만들지 않는다.
- 입사각 영향은 75°부터 88°까지 선형으로 0→1, 그 밖은 clip한다. 법선은 생성기 내부에서만 쓴다.
- severity=max(edge,입사각 영향), σclean=σz(1+0.5 severity), pdrop_clean=0.005+0.02 severity. 최대 σ 1.5배, 최대 drop 2.5%다. outlier 기본 확률은 이 단계에서 그대로다.

공간 Gaussian lattice 간격 0.15 m, 시간 anchor 간격 0.75초다. 8개 공간×2개 시간 node를 보간하고 가중치 제곱합으로 나눠 한 점의 Gaussian 분산을 1로 유지한다. 난수 node는 seed·channel·world lattice·시간 anchor의 stateless hash다. 오차는 공통 환경/드론별/독립 성분의 분산 비율 0.25/0.55/0.20, 누락/outlier patch는 0.35/0.45/0.20으로 결합한다. 독립 성분 키는 scene/run seed·드론·실제 시각·정확한 T이며 정책명과 호출 횟수는 쓰지 않는다.

patch Gaussian의 CDF를 uniform으로 바꿔 하위 꼬리를 drop, 상위 꼬리를 outlier로 쓴다. 두 이벤트는 겹치지 않으며 각각의 주변 확률은 지정한 p다(최대 합이 1보다 작음). patch는 약 0.5–1초와 인접 표면에서 이어진다. outlier 부호·크기도 별도 기존 Gaussian field로 이어진다. 모든 드론이 같거나 완전히 독립인 모형이 아니다. 격자를 바꿔도 σz 식은 그대로다.

## 모의 연기로 인한 깊이 성능 저하

기존 `covor.smoke.line_mean_density`의 공통 이동 3D field를 재사용한다. tau=0.4/m×광선 길이×평균 밀도, T=exp(-tau). 양안 연기 영상과 SGBM은 실행하지 않는다. T와 참 법선·실제 오차는 simulator 전용이다.

| 조건 | k | a | q |
|---|---:|---:|---:|
| clean | 0 | 0 | 0 |
| weak | 0.5 | 0.02 | 0.0005 |
| medium | 1.5 | 0.08 | 0.002 |
| strong | 3.0 | 0.20 | 0.005 |

기하 severity를 먼저 적용한 뒤 σ=σclean[1+k(1−T)], pdrop=pdrop_clean+(1−pdrop_clean)a(1−T), poutlier=0.0005+q(1−T)를 계산하고 확률을 [0,1]로 제한한다. 정상 기하의 strong 최대 추가 drop도 20% 수준이다. 이는 실제 화재 농도나 센서 성능 실측이 아닌 통제된 저하 시나리오다.

## 공개 품질과 정보 분리

지도에 주는 σ는 관측된 Z와 알려진 프로필로 계산한 보수적 값이다. base_sigma(observed Z)×최대 기하 계수×(1+k)에 알려진 outlier 혼합 2차 모멘트와 bias 분산을 합친다. 참 T·참 incidence·실제 오차·outlier flag를 읽지 않는다. ideal_depth는 확률 오차와 bias가 없고 σ=0이다. 옛 disp_sigma_px=1 또는 fB disparity 식은 새 backend에서 사용하지 않는다.

주 프레임 NPZ에는 Z/sigma/valid/T/time/drone_id/teammates만 있다. 참 깊이·오류·outlier mask·T 투과율은 넣지 않는다. 별도 simulator_audit.csv는 평가용 분모와 오차 집계이며 정책이 읽지 않는다. 검증 fixture의 참 깊이는 평가·그림 전용 디렉터리에만 저장한다. 정책에는 현재 관측 지도와 기존 공통 사전 정보만 전달한다.

발판 2프레임·증거 1.2·높이 일관성, 몸 부피·자세·0.18 m 걸음·차단 증명·경로 검증은 유지한다. 4Hz는 기존 0.25초 궤적 실행 시각에서 새 관측을 생성한다. 동일 depth를 복제하지 않는다. 지도가 필요한 배정/평가 주기와 각 촬영의 map_version도 기록한다.

## 검증 분모와 유효 거리 경계

정면·비경계·거리 한계에서 떨어진 core는 기하학적으로 관측 가능, incidence<75°, non-edge, 0.8<Z<5.8 m로 정의했다. 이 영역에서 약 99% 이상의 valid를 검사한다. 전체 영상의 valid 비율에는 범위·벽/양안 가림 손실도 들어가므로 같은 수치가 아니다.

1/2/4/6 m fixture 전체는 빠짐없이 보고한다. 정확히 Z=6 m에서는 양의 Gaussian 오차가 범위 밖으로 나가므로 약 절반이 invalid가 된다. 이를 6 m로 강제 보정하거나 core 통계에 섞어 숨기지 않는다. 범위 censoring 이전의 정상 오차 σ 검증과, valid 측정만 남긴 후의 오차 분포를 별도로 기록한다. 하드 범위 경계의 유효율 감소는 선언한 모델의 한계다.

기존 실험 설정에서 계승한 최상위 `smoke`/`airlight` 항목은 legacy 호환 정보다. 새 backend는 `sensor.error_model.smoke_beta_per_m`와 `degradation` 표만 사용하며 airlight나 영상 혼합을 계산하지 않는다. 유틸리티의 세 함수(candidates/score_pool/marginal)는 기존 커밋과 AST가 같고 사람·이동·경로 평가 모듈의 해시도 유지했다. 바뀐 policy 보조 코드는 오직 새 optical-Z frustum의 가시성·거리 전달이다.
