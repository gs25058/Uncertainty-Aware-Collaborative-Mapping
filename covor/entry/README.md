# `covor/entry/` — 진입지도 (Phase 4)

융합 3D 점유격자를 구조대원이 읽는 2D 진입지도로 바꾼다.
설계는 `reference files/DESIGN_entry_map.md`, 결과는 `RESULTS_entry.md`.

## 원칙 — 판정은 격자, 그림은 벡터

> **안전 판정은 전부 `entry_grid.npz`에서 끝난다. `render.py`는 이미 결정된 것을
> 그릴 뿐이고, 라벨을 만들거나 바꾸거나 문턱을 적용하지 않는다.**

그림이 격자와 다르게 보이는 곳이 있으면 그것은 `render.py`의 버그다. 벡터로 그려지는 것은 두 가지다.

- **벽 윤곽**: occupied 마스크의 marching squares → Douglas-Peucker(ε = 0.5 voxel).
  마스크의 계단은 **실제 정보**다(분류가 바뀌는 자리) — 그래서 문턱을 넘겨 지우지 않고,
  계단을 없애는 답은 더 고운 격자지 더 센 필터가 아니다.
- **폭 등급 경계**: clearance는 연속장이고 walk/narrow는 그 **레벨셋**이다. 곡선이
  매끄러운 이유가 필터가 아니라 양 자체가 매끄럽기 때문이며 sub-voxel 정확하다.

영역은 **중첩 순서**로 뒤에서 앞으로 칠한다(unknown ⊃ 비장애물 ⊃ clearance≥narrow ⊃
clearance≥walk). 각 영역이 직전 것 안에 완전히 들어가므로(장애물 셀의 clearance는 0)
두 경계가 서로 맞을 필요가 없고 이음매도 열리지 않는다. 클래스별로 따로 벡터화하면
독립적으로 단순화된 경계 사이에 실금이 생기는데, 그것을 피하는 구조다.

`tests/test_entry_render.py`가 둘 다 묶는다 — 윤곽 안쪽 셀 집합 == occupied 마스크,
그리고 "walk" 폴리곤 안의 셀은 clearance ≥ clear_walk(허용오차 반 voxel, find_contours가
곡선을 옮길 수 있는 최대치이고 그 이상은 불허).
`fills="raster"`(1셀 = 1픽셀)도 남아 있고 `plot_entry_grid.py`가 그걸 쓴다 — 그쪽 일은
격자를 손대지 않고 보여주는 것이다.

`covor/fusion.py`·`covor/occupancy.py`·`covor/synth/`는 **읽기만** 한다. 수정하지 않는다.

## 모듈

| 파일 | 역할 |
|---|---|
| `config.py` | 모든 파라미터 한 곳 + 라벨 상수. 값마다 근거가 주석으로 붙어 있다 |
| `clean3d.py` | 바닥·천장 추정, 부유큐브 → unknown, occupied 방향 closing |
| `project.py` | 밴드 투영. occupied = 밴드 안 OR, free = 밴드 전체 AND |
| `inflate.py` | EDT clearance, 폭 등급, `r = w/2 + k·σ_xy`, passable |
| `reach.py` | 진입점을 포함하는 passable 연결성분 |
| `sigma_map.py` | 컬럼별 σ_xy와 관측 프레임 수 |
| `route.py` | A*. 비용 = 길이 × (1 + λ/clearance) + unknown 인접 페널티 |
| `render.py` | 채움(격자) + 벽 윤곽(벡터) + 경로·범례·축척·생성조건 |
| *(3D)* | `web/entry_map_3d.template.html` + `scripts/entry/export_entry_3d.py` |
| `metrics.py` | 설계 §6 지표 전부를 **하나의 레코드**로 |

## 파라미터 (`EntryCfg`)

`res 0.10`, `w 0.70`, `clear_walk 0.45`, `clear_narrow 0.30`, `z_min 0.10`,
`H_walk 1.90`, `H_crawl 0.90`, `k_sigma 1.0`(현재 실행은 전부 0),
`close_iter 1` / `close_structure "3d"`, `cube_min_voxels 1`(꺼짐),
`lambda_route 0.5`, `unknown_penalty_steps 1.0`, `n_min_obs 3`.

세 값(`close_*`, `z_min`, `H_crawl`)은 **측정으로 비용이 확인됐지만 설계값을 유지**한다.
첫 지표를 내기 전에 확정했고 사후 변경은 문턱 선택이 된다(RESULTS_SUMMARY §9-4).
비용은 `config.py` 각 필드 주석과 `RESULTS_entry.md` §1에 있다.

**clearance 규약**: `(EDT − 0.5) · res`. 셀 중심→장애물 **표면** 거리다. 반 voxel을
빼지 않으면 설계 §6의 0.6 m 문이 blocked가 아니라 narrow로 나온다. `config.py` 하단 참조.

## 실행

```bash
PY=/src/gs25058/miniconda3/envs/covor/bin/python

# 1) 융합 3D 지도 덤프 (run_experiment는 스칼라만 남기므로 격자를 따로 만든다)
$PY scripts/entry/dump_fused_map.py --cond C_3drone --coverage ii_all_cams \
    --arm depth_only --stride 4                 # 약 5분, 전면 호출 하나

#    음성 대조군: --corrupt jitter0.05 | shift0.05 | yaw3 | nobodycam

# 2) 진입지도 격자
$PY scripts/entry/build_entry_map.py --gt results/synth_room909/gt_voxel.npz \
    --out results/synth_room909/entry/entry_grid_gt.npz
$PY scripts/entry/build_entry_map.py --map <dump>.npz --out <grid>.npz --w 0.7

# 3) 지표 (GT 대비)
$PY scripts/entry/run_entry_metrics.py --gt results/synth_room909/gt_voxel.npz \
    --map <dump>.npz:tag --w 0.5,0.7,0.9 --k-sigma 0

# 4) 그림 (SVG + PNG) + routes.json
$PY scripts/entry/make_entry_figure.py --gt results/synth_room909/gt_voxel.npz \
    --map <dump>.npz:"C - 3 drones" --out <stem> --w 0.70 --k-sigma 0
#    --sigma-panel  σ_xy 필드를 한 줄 더 (측정값, 판정 불참)
#    --bands walk   한 밴드만

# 격자를 있는 그대로(렌더 층 없이) 보고 싶을 때
$PY scripts/entry/plot_entry_grid.py --in <grid>.npz --out <png>

# 5) 3D 뷰어 (자립형 HTML 하나, three.js는 CDN)
$PY scripts/entry/export_entry_3d.py --gt results/synth_room909/gt_voxel.npz \
    --map <dump>.npz:key:라벨:설명  --out web/entry_map_3d_room909.html
```

## 2D와 3D는 같은 격자를 다르게 본다

평면도는 **폭 등급**으로 칠하고(walk/narrow/blocked), 3D 뷰어는 **제시 여부**로 칠한다
(설 수 있는 곳 / 몸이 닿는 바닥 / 권고 / 도달 불가). 둘 다 `entry_grid.npz`만 읽고
라벨을 바꾸지 않으며, 각각 회귀 테스트가 그것을 셀 단위로 단언한다.

**c-space vs workspace** (uacm에서 이식): `passable`·`reachable`은 몸 *중심*이 놓일 수
있는 집합(configuration space)이고, 사람이 "걸을 수 있는 바닥"이라고 부르는 것은 몸이
쓸고 지나가는 집합(workspace)이다. 후자는 전자를 몸 반경만큼 되부풀린 것이고, 각 지도의
free 셀로 잘라낸다. **판정은 c-space에서만** 하고(보수적인 쪽), workspace는 보고·표시
전용이다. room909 C 조건에서 11.3 m² → 18.6 m², 도달 IoU 0.622 → 0.672.

## 테스트

```bash
$PY tests/test_entry_map.py      # 12 — 설계 §6 회귀 + 바닥/천장 + 진입점
$PY tests/test_entry_dump.py     #  2 — 중복된 두 규칙을 원본과 값으로 대조
$PY tests/test_entry_render.py   #  4 — 벡터 윤곽 == occupied 마스크, 비변경, A* 비용
```

전부 **값 비교**다. 개수만 보는 테스트는 쓰지 않는다(RESULTS_SUMMARY §9-2 / 부록 A-4).

## 한계 (읽기 전에)

- **false-passable rate는 이 데이터에서 탐지력이 없다.** 음성 대조군 2종이 모두
  불통과했고 사전등록한 규칙대로 그렇게 판정했다. 단독 인용 금지. `RESULTS_entry.md` §4.
- **σ 기하 여유(k_sigma)는 검증되지 않았다.** 모든 실행이 k_sigma = 0이다.
- 다층은 층별 반복까지만(계단 연결 없음), 동적 장애물과 진입점 자동 검출은 범위 밖.
