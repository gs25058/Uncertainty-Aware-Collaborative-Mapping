# 화재 규모 확대 — 건물·기하 경로 조건은 고정

기존 10개 복층 건물에서 **화재 규모만 확대**했다. `fire_scale=3`이며, 기존 국소 가구 화재를 층당 3개 인접 실내 구역의 화재로 넓혔다.

**[확대 화재 viewer](../../web/fire_buildings_large_fire_3d.html)** · **[미리보기](index.html)** · **[모델·viewer ZIP](../../results/fire_buildings_large_fire_20261107/fire_buildings_10.zip)** · **[변경 전후 CSV](fire_scale_comparison.csv)**

- 총 화재 소스: **25 → 75곳**, 장면당 2/3곳에서 **6/9곳**으로 증가했다.
- 선언한 반경의 원을 실내 경계로 자른 평면 합집합 면적: **38.70 → 311.65 m²**(10개 장면·층 합계, 5cm 샘플). 약 **8.05배**다. 이는 가상 소스 반경의 투영 면적이며 실제 연소 면적·화재 열방출률·농도가 아니다.
- 불꽃 군집을 소스당 4개에서 11개로 늘리고 폭을 키웠다. 높이는 기존 약 1.8–2.0m에서 가구/바닥 높이에 따라 약 2.2–2.8m까지 커지며, 각 층 천장 아래에서 제한한다. 소스와 군집은 기존 가구·바닥 표면에 붙여 배치했다.
- 인접한 방의 가구·서가 등에 화재가 분산되어 손상된 내·외벽과 천장 주변에도 큰 화재가 보인다. 연기 입자 수·폭·불투명도를 늘리고 소속 실내 범위와 층 높이로 제한했다. 표시용 조명은 최대 12개 소스를 처리해 3층 장면의 9개 화재를 빠뜨리지 않는다.

## 연구 목적에 맞춰 유지한 조건

화재 확대는 건물 생성이 끝난 후 별도 seed의 시각 요소로 추가한다. **벽·창·문·계단·붕괴·잔해·가구 형상과 색, 입구·목표·층 조건을 바꾸지 않았다.** `solid=false`인 화염을 벽처럼 점유지도에 넣지 않았다.

기존 파일과 비교해 10개 모두 아래를 실제로 확인했다.

- 고체 부품의 꼭짓점·삼각형·재질 목록 동일.
- `scene.ply` 파일 바이트 동일(기존 재질 색 포함).
- `gt_voxel.npz`의 labels/ijk/res 배열 동일.
- 입구·목표·계단·바닥 구멍 동일.
- 화염이 고체 충돌 형상에서 제외되고 천장 높이를 넘지 않음.

검증 결과는 [invariants.json](invariants.json), 장면별 결과는 [fire_scale_comparison.csv](fire_scale_comparison.csv)에 있다. 240회 기존 형상 raycast 검사를 새 장면에도 실행했다. 전체 10개 장면·25개 층과 각 표시 기능의 실제 브라우저 검사는 `browser_checks.json`, 화면은 `previews/`에 저장한다.

이번 범위는 **화재 시각화 및 소스 설정의 확대**다. 기존 깊이 오차·관측 저하 프로필, 사람 기하 판정, unknown/entry 정책은 수정하지 않았다. 새 소스의 위치/반경/높이/상대 세기를 저장했지만 **현재 센서의 연기장과는 아직 연동하지 않았다.** 따라서 불이 크게 보인다는 이유로 알고리즘 성능이나 경로 차단 정답이 달라졌다고 주장하지 않는다. 열·가스 위험으로 인한 진입 불가도 추가하지 않았다.

이 구분은 실험에 필요하다. 다음 관측 저하 실험에서 두 정책이 동일한 소스·시드·센서 저하 조건을 공유하도록 사용할 수 있고, 고체 지도는 동일하게 둔 채 관측 조건의 효과를 분리할 수 있다. 현재 화재 규모가 실제 구조 붕괴를 물리적으로 일으켰다고 검증한 결과는 아니다. 구조 손상은 기존의 고정 형상이다.

## 파일과 재생성

새 자산: `results/fire_buildings_large_fire_20261107/<장면ID>/`의 GLB, 고체 PLY, 기준 복셀, 소스 설정, 충돌 부품, 형상 검증. 기존 작은 화재 자산·ZIP은 `results/fire_buildings_20261107/`에 보존했다. 이전 viewer도 `web/fire_buildings_3d_small_fire.html`에 보존하고, 기본 `web/fire_buildings_3d.html`은 확대본으로 갱신한다.

```bash
cd /src/gs25058/cr_RNE/covor_slam
export COVOR_PY=/src/gs25058/miniconda3/envs/covor/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
$COVOR_PY scripts/explore/build_fire_suite.py --fire-scale 3 --out results/fire_large_rebuild --viewer web/fire_large_rebuild.html
$COVOR_PY scripts/explore/verify_fire_scale.py --base results/fire_buildings_20261107 --large results/fire_large_rebuild --out reports/fire_large_rebuild
PYTHONPATH=/tmp/covor_browser_tools_20261107 $COVOR_PY scripts/explore/check_fire_viewer.py --viewer web/fire_large_rebuild.html --out reports/fire_large_rebuild
$COVOR_PY scripts/explore/package_fire_suite.py --root results/fire_large_rebuild --viewer web/fire_large_rebuild.html --report reports/fire_large_rebuild
```

`--fire-scale 1`은 기존 국소 화재 생성 방식이다. 새 출력 경로를 사용하면 기존 파일을 덮어쓰지 않는다. 수정한 구현은 `FireBuilding._expand_fire`, 생성 CLI의 `--fire-scale`, viewer의 화염·연기·조명 처리이며, 실제 비교 검증은 `verify_fire_scale.py`다.
