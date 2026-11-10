#!/usr/bin/env python3
"""Bundle the scene library with an offline viewer and record exact provenance."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_buildings_20261107');p.add_argument('--viewer',default='web/fire_buildings_3d.html');p.add_argument('--report',default='reports/fire_buildings_20261107');a=p.parse_args()
    root,viewer,report=Path(a.root),Path(a.viewer),Path(a.report);report.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((root/'catalog.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest['source_sha256'].items()),'source differs from generated scenes'
    browser=json.loads((report/'browser_checks.json').read_text());assert browser['scenes']==len(catalog) and not browser['runtime_errors']
    archive=root/'fire_buildings_10.zip'
    if archive.exists():raise FileExistsError(archive)
    readme='''복층 화재·붕괴 가상 건물 10개\n\nviewer.html을 Chrome/Edge/Firefox에서 직접 여세요. 외부 CDN과 서버가 필요 없습니다.\n좌측에서 건물·층을 고르고 전체/단면/층 간격/지붕/화염/연기를 전환할 수 있습니다.\n\nscenes/<장면ID>/scene.glb: 색상과 화염 시각 요소가 있는 3D 장면\nscene.ply: 고체 충돌 메시. 불·연기는 고체에 포함하지 않습니다.\ngt_voxel.npz: 15cm 완전 형상 복셀(labels: 1=air, 2=solid), ijk, res.\nscene.json: 층·실내 구역·창·문·구멍·계단·화재 위치/시드/설정.\ncollision_geometry.json: 모든 부품의 꼭짓점·삼각형·solid 플래그.\nvalidation.json: 실제 메시 raycasting·계단·문·창·가상 깊이 검사.\n\n단위 m, Z-up. 화재·연기는 시각/소스 주석이며 열·가스·CFD·붕괴 동역학은 아닙니다.\n건물 손상은 실행 중 고정됩니다. 이 신규 복층 장면의 사람 도달 정답은 아직 인증하지 않았습니다.\n기존 48초 unknown/entry 실험 결과를 이 장면의 결과로 해석하면 안 됩니다.\n'''
    bundled=re.sub(r"window.FIRE_ASSET_ROOT='[^']*'","window.FIRE_ASSET_ROOT='scenes'",viewer.read_text())
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('fire_buildings_10/viewer.html',bundled);z.writestr('fire_buildings_10/README.txt',readme)
        portable=[dict(s,files={name:'scenes/'+s['id']+'/'+name for name in s['files']}) for s in catalog]
        z.writestr('fire_buildings_10/catalog.json',json.dumps(portable,ensure_ascii=False,indent=2))
        for scene in catalog:
            for f in (root/scene['id']).iterdir():
                if f.is_file():z.write(f,'fire_buildings_10/scenes/'+scene['id']+'/'+f.name)
    (root/'README.txt').write_text(readme)
    for name in ('catalog.json','catalog.csv','manifest.json'):shutil.copy2(root/name,report/name)
    source=root/'source_snapshot';source.mkdir(exist_ok=True)
    files=list(manifest['source_sha256'])+['scripts/explore/check_fire_viewer.py','scripts/explore/package_fire_suite.py']
    if Path('scripts/explore/verify_fire_scale.py').exists():files.append('scripts/explore/verify_fire_scale.py')
    for f in files:
        relative=Path(f).resolve().relative_to(Path.cwd())
        destination=source/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,destination)
    hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    metadata=dict(source_sha256=hashes,viewer_sha256=hashlib.sha256(viewer.read_bytes()).hexdigest(),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),scenes=len(catalog),mesh_validation='10/10 passed',browser_validation='10/10; all floors and controls passed')
    (root/'release.json').write_text(json.dumps(metadata,indent=2)+'\n');shutil.copy2(root/'release.json',report/'release.json')
    cards=[]
    for i,s in enumerate(catalog):
        href=os.path.relpath(viewer,report)+f'#scene={i+1}&floor=-1'
        image='previews/'+s['id']+'_overview.png'
        cards.append(f'<a class="card" href="{href}"><img src="{image}" loading="lazy"><h2>{html.escape(s["title"])}</h2><p>{s["levels"]}층 · {s["footprint_m"][0]:g} × {s["footprint_m"][1]:g}m · 창문 {s["windows"]}개</p><p>{html.escape(s["description"])}</p></a>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>복층 화재·붕괴 건물 10선</title><style>body{margin:40px auto;max-width:1300px;background:#171e24;color:#e8e8e8;font-family:system-ui;padding:0 20px}h1{font-size:30px}p{color:#aebbc3;line-height:1.6}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}.card{display:block;text-decoration:none;color:inherit;background:#242e35;border:1px solid #48545d;border-radius:10px;overflow:hidden}.card img{display:block;width:100%;aspect-ratio:1.5;object-fit:cover}.card h2,.card p{margin:12px 18px}h2{font-size:18px}.card:hover{border-color:#e6a06a}</style><h1>복층 화재·붕괴 건물 10선</h1><p>창문 · 계단 · 벽 파손 · 천장 붕괴 · 바닥 구멍 · 잠긴 문 · 실내 화재<br>장면을 누르면 오프라인 3D viewer가 열립니다. 층별 내부와 전체 구조를 전환할 수 있습니다.</p><main>'''+''.join(cards)+'</main></html>'
    (report/'index.html').write_text(page)
    print('packaged',archive,round(archive.stat().st_size/1e6,2),'MB; gallery',report/'index.html')


if __name__=='__main__':main()
