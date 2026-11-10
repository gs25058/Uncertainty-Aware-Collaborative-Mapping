#!/usr/bin/env python3
"""Show saved virtual-building experiments with the existing offline WebGL viewer."""
import argparse
import base64
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


def payload(lab, ijk, res, trajectories):
    occ = np.argwhere(lab == 2)
    free = np.argwhere(lab == 1)
    if len(free) > 60000:
        free = free[np.linspace(0, len(free) - 1, 60000).astype(int)]
    z = (occ[:, 2] + ijk[2] + .5) * res
    stops = np.array([[68, 1, 84], [62, 74, 137], [38, 130, 142],
                      [53, 183, 121], [180, 222, 44], [253, 231, 37]])
    t = np.clip((z + .9) / 3.8, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    col = np.round(stops[i] * (1 - (t-i)[:, None]) + stops[i+1] * (t-i)[:, None]).astype('u1')
    padded = np.pad(lab == 2, 1)
    mask = np.zeros(len(occ), 'u1')
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for bit, offset in enumerate(neighbors):
        q = occ + 1 + offset
        mask |= padded[tuple(q.T)].astype('u1') << bit
    return dict(res=res, occ_idx_b64=b64((occ+ijk).astype('<i2')),
                occ_col_b64=b64(col), occ_mask_b64=b64(mask),
                free_idx_b64=b64((free+ijk).astype('<i2')),
                zmin=-.9, zmax=2.9, czmin=-.9, czmax=2.9,
                center=[6, 3.5, .8], radius=8, trajs=trajectories,
                color_label='높이 (m) · 보라: 낮음 / 노랑: 높음')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', type=Path, default=ROOT/'results/depth_model_20261106')
    ap.add_argument('--out', type=Path, default=ROOT/'web/fire_scene_3d.html')
    args = ap.parse_args()
    # Use the scene code archived with this experiment, including its precise RNG order.
    scene_file = args.experiment/'source_snapshot/covor/explore/scene.py'
    spec = importlib.util.spec_from_file_location('recorded_fire_scene', scene_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    datasets = {}
    colors = [[.96, .65, .14], [.18, .83, .75], [.88, .36, .78], [.4, .6, 1.]]
    for seed in (11, 29):
        scene = module.Building(seed)
        lab, ijk = scene.labels(.1)
        datasets[f'{seed}_gt'] = payload(lab, ijk, .1, [])
        for condition in ('clean', 'strong'):
            run = args.experiment/'main'/f'scene{seed}_seed0_{condition}_entry'
            with np.load(run/'maps/final.npz') as q:
                lab, ijk, res = q['labels'], q['ijk'], float(q['res'])
            trajectories = [[] for _ in colors]
            with (run/'trajectories.csv').open() as f:
                for row in csv.DictReader(f):
                    trajectories[int(row['drone'])].append([float(row[f'estimated_{axis}']) for axis in 'xyz'])
            traj = [dict(name=f'드론 {i+1}', color=c, pts=pts)
                    for i, (c, pts) in enumerate(zip(colors, trajectories))]
            datasets[f'{seed}_{condition}'] = payload(lab, ijk, res, traj)
    html = (ROOT/'web/occupancy_viewer.template.html').read_text()
    html = '<!doctype html>\n<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n' + html + '\n</html>\n'
    html = html.replace('__TITLE__', '가상 화재 현장 · 3D 지도')
    html = html.replace('<h1>Occupancy Map</h1><span class="tag">CoVOR · UWB</span>', '<h1>가상 화재 현장</h1><span class="tag">3D MAP</span>')
    html = html.replace('__SUBTITLE__', '''
      <label>장면 <select id="sceneSelect"><option value="11">건물 11</option><option value="29">건물 29</option></select></label><br>
      <label>지도 <select id="mapSelect"><option value="gt">가상 건물 원형 (GT)</option><option value="clean">드론 복원 · 무연기</option><option value="strong">드론 복원 · 강한 관측 저하</option></select></label><br>
      <span id="sourceNote"></span><br>
      12 × 9 m 건물 · 잔해, 낮은 보, 좁은 문, 바닥 함몰.<br>
      천장을 잘라 내부를 표시합니다. 높이 상한을 올리면 천장이 나타납니다.<br>
      색은 높이이며 화염·온도 표시가 아닙니다. 연기는 깊이 관측 저하로 반영됩니다.
    ''')
    library = json.dumps(datasets, separators=(',', ':'), ensure_ascii=False)
    init = '''
      const params = new URLSearchParams(location.search);
      const seed = ['11','29'].includes(params.get('scene')) ? params.get('scene') : '11';
      const mode = ['gt','clean','strong'].includes(params.get('map')) ? params.get('map') : 'gt';
      document.getElementById('sceneSelect').value=seed;
      document.getElementById('mapSelect').value=mode;
      document.getElementById('sourceNote').textContent=mode==='gt'
        ? '실험에 보관된 건물 형상 · 10 cm 복셀'
        : '2026-11-06 저장 지도 · 4대 / 48초 · entry 정책 / seed 0';
      for (const id of ['sceneSelect','mapSelect']) document.getElementById(id).onchange=()=>{
        const q=new URLSearchParams({scene:document.getElementById('sceneSelect').value,map:document.getElementById('mapSelect').value});
        location.search=q.toString();
      };
      window.__OCC__ = window.__FIRE__[seed+'_'+mode];
    '''
    html = html.replace('window.__OCC__ = /*__OCC_DATA__*/null;', 'window.__FIRE__ = '+library+';\n'+init)
    translations = {'Layers':'표시 항목', 'Occupied voxels':'벽·바닥·장애물', 'Free space':'관측된 빈 공간 (최대 6만 점)',
                    'Drone trajectories':'드론 이동 궤적', 'Floor grid':'기준 격자', 'Auto-orbit':'자동 회전',
                    'Pretty render':'음영 효과', 'Height slice':'표시 높이 범위', 'Reset view':'시점 초기화',
                    'drag to orbit · scroll to zoom · right-drag to pan':'드래그: 회전 · 휠: 확대/축소 · 우클릭 드래그: 이동'}
    for old, new in translations.items():
        html = html.replace(old, new)
    html = html.replace('value="100" aria-label="slice ceiling"', 'value="70" aria-label="slice ceiling"')
    html = html.replace('sliceHi.value=100;', 'sliceHi.value=70;')
    html = html.replace('</style>', 'select{max-width:100%;margin:5px 0;padding:6px;background:#172335;color:#e0ebf8;border:1px solid #40536a;border-radius:5px}#controls{max-height:calc(100vh - 36px);overflow:auto}</style>')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f'{args.out}: {len(datasets)} maps, {args.out.stat().st_size/1e6:.2f} MB')


if __name__ == '__main__':
    main()
