# 복층 화재·붕괴 건물 10개

2026-11-07. 기존 단층 실험 장면과 결과를 보존하고 별도 라이브러리를 생성했다. 아래는 드론이 복원한 지도가 아니라 **가상 장면의 완전한 형상**이다.

**[3D viewer](../../web/fire_buildings_3d_small_fire.html)** · **[10개 미리보기 갤러리](index.html)** · **[장면 목록 CSV](catalog.csv)** · **[모델·viewer ZIP](../../results/fire_buildings_20261107/fire_buildings_10.zip)**

## 장면

모든 장면에 복층, 실제 계단, 창문/창틀/남은 유리 조각, 열린 문과 잠긴 닫힌 문, 파손된 내·외벽, 바닥 구멍, 떨어진 천장 패널, 잔해, 낮게 처진 보, 가구, 실내 화재 위치가 있다. 벽 구멍과 창은 메시 자체의 개구부다. 바닥 구멍 위를 투명 색으로만 표시한 것이 아니며 윗층 슬래브도 실제 제거했다. 층별 슬래브와 천장 두께 0.22 m, 층고 3.3 m를 사용한다.

| ID | 유형 | 층 | 크기[m] | 실내 평가 구역 | 창문 | 잠긴 문 | 계단 연결 |
|---|---|---:|---|---:|---:|---:|---:|
| 01_apartment | 공동주택 | 2 | 22×16 | 16 | 27 | 3 | 2 |
| 02_school | 학교 | 3 | 26×18 | 24 | 41 | 4 | 4 |
| 03_office | 사무동 | 3 | 24×18 | 30 | 48 | 4 | 4 |
| 04_hotel | 소형 호텔 | 3 | 23×17 | 30 | 48 | 4 | 4 |
| 05_clinic | 의원 | 2 | 24×17 | 16 | 27 | 3 | 2 |
| 06_retail | 상가 | 2 | 26×19 | 16 | 27 | 3 | 2 |
| 07_workshop | 작업장 | 2 | 25×18 | 16 | 27 | 3 | 2 |
| 08_dormitory | 기숙사 | 3 | 25×17 | 30 | 48 | 4 | 4 |
| 09_duplex | 복층 주거동 | 2 | 20×16 | 16 | 27 | 3 | 2 |
| 10_library | 도서관 별관 | 3 | 26×19 | 24 | 41 | 4 | 4 |

seed는 71100–71109로 고정했다. 공통 모듈식 건물 규칙에서 크기·층·칸막이·손상·가구를 달리했다. 주거는 거실/주방/침실, 학교는 책상/칠판, 사무실은 책상/모니터, 의원은 진료 침대, 상가/도서관은 진열장/서가를 사용한다. 상가·작업장·도서관 일부 칸막이는 제거해 연결된 큰 구역을 만들었다. 호텔·작업장·복층 주거동은 보조 계단의 두 발판도 파손했다. 주 계단은 남겼다. 실제 건축물 스캔이나 독립적으로 설계한 실건물 도면 10개라고 해석하지 않는다.

## viewer 사용

HTML을 직접 열 수 있다. 외부 JS/CDN, 인터넷, 별도 서버가 필요 없다. 건물 선택 후 `전체 / 1층 / 2층 / 3층`을 전환한다. `벽 단면`은 높은 벽을 잘라 내부를 보고, `층 간격 벌리기`는 전체 보기의 표시 위치만 바꾼다. 메시 원본과 충돌 좌표는 바뀌지 않는다. 지붕·가구/창틀/문·화염·연기·이름을 별도로 켤 수 있다. 주요 구간을 누르면 그 위치로 확대한다. 드래그 회전, 우클릭 드래그 이동, 휠 확대/축소, 숫자 0–3 층 선택을 지원한다.

창문은 깨진 상태로 모델링했다. 남아 있는 유리 조각은 고체 메시이고 광학적으로는 불투명한 표면으로 취급한다. 나머지 창 개구부에는 보이지 않는 전체 유리판을 두지 않았다. 잠긴 문은 닫힌 문짝·프레임·손잡이·잠금장치와 `locked_closed` 상태를 갖는다. 실행 중 문을 강제로 열거나 붕괴시키는 동작은 없다.

## 저장 파일

원자료는 `results/fire_buildings_20261107/<ID>/`이다.

- `scene.glb`: 색상·부품 이름과 정적인 화염 표시를 포함하는 3D 보기용 자산. 화염 부품은 `fire_visual`이다.
- `scene.ply`: **고체 충돌 메시**. 화염·연기는 포함하지 않는다.
- `gt_voxel.npz`: 15 cm 완전 형상 복셀, `labels`(1=air, 2=solid), `ijk`, `res`. convex half-space 기반 보수적 래스터화이므로 얇은 부품은 두꺼워질 수 있고, 형상 검증은 메시를 우선한다.
- `scene.json`: m·Z-up 좌표, 층·실내 구역·출입구·계단·문/손상 특징·화재 위치·seed·공통 사전 정보.
- `collision_geometry.json`: 부품별 꼭짓점/삼각형과 `solid` 플래그. `solid=false` 화염은 센서·충돌·복셀에서 제외한다.
- `validation.json`: 실제 메시 raycast 결과, 계단·잠긴 문·창 개구부·바닥 구멍 검사와 기존 가상 깊이 backend 연결 결과.

루트의 catalog/manifest/release에는 파일 경로·seed·소스/모델 묶음 SHA256을 저장한다. `source_snapshot/`에 생성기·뷰어 템플릿·검증·포장 스크립트 사본도 있다. 초기 화면 점검용 draft/pilot은 별도 `_pilot`, `_draft` 경로에 남겼고, 최종 viewer/ZIP은 위 최종 경로만 사용한다.

## 검증과 적용 범위

10개 모두 메시 생성과 **총 240회 실제 메시 raycast 검사**를 통과했다. 각 층의 복도 바닥, 양쪽 계단 발판, 실제 창 개구부, 잠긴 문짝, 윗층 바닥 구멍을 검사했다. 계단은 두 flight의 총 20단, 한 단 높이 **0.165 m**, 디딤면 **0.29 m**, 폭 **1.25 m**다. 첫 표면 raycasting에 기존 `RayDepthSensor`를 연결해 장면마다 실제 유효 깊이를 생성했다. 창이나 구멍을 가로질러 가짜 표면을 추가하지 않았다.

Chrome/WebGL2에서 10개 장면과 모든 층, 단면/층 간격/지붕/가구/화염/연기/라벨 전환을 실제 실행했다. 결과는 [browser_checks.json](browser_checks.json), 실제 브라우저 캡처는 `previews/`에 있다. 저장된 화면 예시는 [공동주택 1층](previews/01_apartment_L1.png)이다.

화염·연기는 **시각화와 소스 주석**이다. CFD, 열·가스·실측 연기 농도·시간에 따른 구조 붕괴를 계산하지 않는다. 손상·잔해는 정적이다. 기존의 공통 깊이 저하 모델과 새 화재 위치가 물리적으로 연동됐다고 주장하지 않는다.

새 복층 장면은 기존 direct-depth 관측 생성에 사용할 수 있지만, 기존 runner의 단층 `Building`을 자동 대체하지 않는다. **새 장면의 사람 목표 도달 정답과 unknown/entry 비교는 아직 실행·인증하지 않았다.** 특히 기존 단층 reference의 높이 제한과 목표 분모를 그대로 옮겨 복층 정답이라고 부르면 안 된다. 여기서 완료한 범위는 10개 장면·고체/시각 자산·복셀·관측 생성 연결·viewer다.

## 재생성

기존 출력은 덮어쓰지 않는다. 새 경로를 지정한다.

```bash
cd /src/gs25058/cr_RNE/covor_slam
export COVOR_PY=/src/gs25058/miniconda3/envs/covor/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
$COVOR_PY scripts/explore/build_fire_suite.py --out results/fire_buildings_rebuild --viewer web/fire_buildings_rebuild.html
```

`--indices 0 2 5`처럼 일부 장면만 생성하거나 `--resolution 0.10`으로 복셀 격자를 바꿀 수 있다. 원본 메시 좌표는 바뀌지 않는다.

선택적인 실제 브라우저 검사에는 Chrome과 Python `websocket-client`가 필요하다. 장면 생성이나 viewer 열기에는 이 패키지가 필요 없다. 이번 서버에서는 임시 경로 `/tmp/covor_browser_tools_20261107`에만 설치했다.

```bash
PYTHONPATH=/tmp/covor_browser_tools_20261107 $COVOR_PY scripts/explore/check_fire_viewer.py --viewer web/fire_buildings_rebuild.html --out reports/fire_buildings_rebuild
$COVOR_PY scripts/explore/package_fire_suite.py --root results/fire_buildings_rebuild --viewer web/fire_buildings_rebuild.html --report reports/fire_buildings_rebuild
```

구현은 `covor/explore/fire_buildings.py`, `scripts/explore/build_fire_suite.py`, `web/fire_buildings_3d.template.html`에 있다. 기존 단층 코드·센서·정책·실험 결과를 수정하지 않았다.
