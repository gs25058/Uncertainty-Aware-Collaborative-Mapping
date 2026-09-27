#!/usr/bin/env python3
"""Bake a Polycam scan into the standalone 3D viewer (web/scan_3d.template.html).

    python scripts/scan/export_scan_3d.py \
        --obj "meshes/2026-09-15/2026. 9. 15.obj" --name 2026-09-15 \
        --out web/scan_3d_2026-09-15.html

WHY THIS EXISTS. The entry map is a voxel drawing of a space; on its own there
is no way to see whether the thing it draws is the space. Polycam's own mesh is
that reference -- it is what the GT voxels are cut from -- but outside the app
it renders black (see mesh_payload's material note) and it lives in the scan's
own arbitrary frame. This puts it in THIS project's world frame, with the same
camera and the same theme as web/entry_map_3d_*.html, so the two can be read
side by side without registering anything.

THE WORLD TRANSFORM IS NOT INVENTED HERE. It comes from covor.synth.mesh_gt's
`mesh_to_world`, the same call scripts/synth/build_gt_voxel.py makes with its
default flags, computed from the same `inspect` dump. Run Part A on this OBJ
later and the map lands on top of this mesh, because it is literally the same
4x4. With --mesh-config the frozen one is read from disk instead and the two
are asserted equal.
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/scan")
from covor.synth import mesh_gt as MG
import mesh_payload as MP

FOV = 42.0


def floor_drift(mesh, up=1, bins=14):
    """(drift, low, high) of the dominant floor plane along the longest axis.

    A scan of a corridor is locally metric and globally not: Polycam's floor
    wanders slowly over a long walk. One rigid transform cannot take that out,
    so it is measured and reported rather than silently absorbed.
    """
    N, A, C = mesh.face_normals, mesh.area_faces, mesh.triangles_center
    long_axis = int(np.argmax(mesh.extents))
    hor = (np.abs(N[:, up]) > 0.9) & (C[:, up] < np.median(mesh.vertices[:, up]))
    if hor.sum() < 200:
        return None
    edges = np.linspace(mesh.bounds[0, long_axis], mesh.bounds[1, long_axis], bins + 1)
    modes = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = hor & (C[:, long_axis] >= a) & (C[:, long_axis] < b)
        if m.sum() < 80:
            continue
        h, e = np.histogram(C[m, up], bins=120, weights=A[m])
        modes.append(float(0.5 * (e[:-1] + e[1:])[np.argmax(h)]))
    if len(modes) < 3:
        return None
    return dict(drift=float(max(modes) - min(modes)), lo=float(min(modes)),
                hi=float(max(modes)), n_seg=len(modes))


def shard_stats(mesh):
    """How the scan is cut up. Polycam exports chunk-wise, so 'keep the largest
    connected component' -- the reflex cleanup for a scan -- would here throw
    away almost everything. Reported so nobody reaches for it."""
    A = mesh.area_faces
    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1,
                                            nodes=np.arange(len(mesh.faces)))
    ar = np.array(sorted((A[c].sum() for c in cc), reverse=True))
    return dict(n=len(ar), largest_frac=float(ar[0] / ar.sum()),
                small_n=int((ar < 0.5).sum()),
                small_frac=float(ar[ar < 0.5].sum() / ar.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="web/scan_3d.template.html")
    ap.add_argument("--mesh-config", help="read the frozen T_mesh_world from a "
                                          "Part A mesh_config.json instead")
    ap.add_argument("--atlas-px", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--blurb", default="")
    args = ap.parse_args()

    mesh = MG.load(args.obj)
    info = MG.inspect(mesh)
    T = MG.mesh_to_world(info)
    if args.mesh_config:
        Tf = np.array(json.load(open(args.mesh_config))["T_mesh_world"], float)
        if not np.allclose(T, Tf, atol=1e-9):
            raise SystemExit("the frozen T_mesh_world and the one this OBJ "
                             "implies differ -- refusing to guess which is the "
                             "map's frame:\nfrozen\n%s\nhere\n%s" % (Tf, T))
        T = Tf
    mw = MG.apply_transform(mesh, T)
    lo, hi = mw.bounds
    ext = mw.extents

    drift = floor_drift(mesh)
    shards = shard_stats(mesh)
    tri_edge = np.sqrt(4.0 * (info["area"] / len(mesh.faces)) / np.sqrt(3.0))

    payload = MP.build(args.obj, T, px=args.atlas_px, quality=args.quality)
    n_atlas = sum(1 for c in payload["chunks"] if c["tex"] is not None)
    texel_m = np.sqrt(info["area"] / (n_atlas * float(args.atlas_px) ** 2))

    stats = [
        ("정점 / 삼각형", "{:,} / {:,}".format(payload["stats"]["n_vert"],
                                              payload["stats"]["n_face"])),
        ("스캔 크기 (x·y·z)", "%.1f × %.1f × %.1f m" % tuple(ext)),
        ("표면적", "%.0f m²" % info["area"]),
        ("천장고", "%.2f m" % info["room_height"]),
        ("삼각형 평균 변", "%.1f cm" % (tri_edge * 100)),
        ("아틀라스", "%d장 × %d px" % (n_atlas, args.atlas_px)),
        ("텍셀 간격", "%.1f mm" % (texel_m * 1000)),
        ("텍스처 / 기하 배율", "%.0f×" % (tri_edge / texel_m)),
        ("조각 수", "{:,}개".format(shards["n"])),
        ("최대 조각", "전체 면적의 %.1f%%" % (100 * shards["largest_frac"])),
        ("바닥 z 범위", "%.2f … %.2f m" % (lo[2], hi[2])),
    ]
    if drift:
        stats.append(("바닥 표류", "%.0f cm / %.0f m" % (drift["drift"] * 100,
                                                        ext[int(np.argmax(ext))])))

    caveats = [
        "<b>기하는 손대지 않았다.</b> 데시메이션·구멍 메움·부유물 제거 없음. "
        "이 메시가 GT 복셀의 원본이므로, 지도와 비교하려면 GT가 보는 그대로여야 한다.",
        "<b>조각 %s개, 최대 조각이 면적의 %.1f%%.</b> Polycam이 청크 단위로 내보내서 "
        "'가장 큰 연결 성분만 남기기'식 정리는 여기서 스캔을 거의 다 지운다. "
        "정리가 필요하면 위상이 아니라 기하(ROI·면적 임계)로 해야 한다."
        % ("{:,}".format(shards["n"]), 100 * shards["largest_frac"]),
    ]
    if drift:
        caveats.append(
            "<b>바닥이 %.0f cm 표류한다</b>(%.0f m 구간, 천장고는 어디서나 %.2f m로 일정). "
            "국소는 정확하고 전역이 휜다는 뜻이라, 강체 변환 하나로는 먼 쪽 끝이 "
            "바닥에서 그만큼 떠 있다 — 0.05 m 격자로 %.0f복셀."
            % (drift["drift"] * 100, ext[int(np.argmax(ext))], info["room_height"],
               round(drift["drift"] / 0.05)))
    caveats.append(
        "<b>월드 프레임</b>: 바닥 z = 0, 바닥면 x·y 중심이 원점. "
        "covor.synth.mesh_gt.mesh_to_world가 정한 것으로, build_gt_voxel.py를 "
        "기본 인자로 돌리면 같은 4×4가 나온다.")

    fit_top = float(max(ext[0], ext[1]) / 2 / np.tan(np.radians(FOV / 2)) * 1.15)
    meta = dict(
        name=args.name, stats=stats, caveats=caveats,
        z_range=[float(lo[2]), float(hi[2])],
        grid_span=float(np.ceil(max(ext[0], ext[1])) + 2),
        fit_top=round(fit_top, 1), fit_iso=round(fit_top * 0.62, 1),
        T_mesh_world=T.tolist(), obj=os.path.abspath(args.obj))

    blurb = args.blurb or (
        "%s 스캔을 이 프로젝트의 월드 프레임에 놓고 Polycam과 같은 방식으로 그린 것. "
        "진입지도가 그리는 공간이 실제로 어떤 공간인지에 대한 기준선이다." % args.name)

    tpl = open(args.template, encoding="utf-8").read()
    if "/*__DATA__*/null" not in tpl:
        raise SystemExit("template has no /*__DATA__*/null placeholder")
    html = (tpl.replace("/*__DATA__*/null",
                        json.dumps(dict(mesh=payload, meta=meta), ensure_ascii=False))
               .replace("__NAME__", args.name)
               .replace("__BLURB__", blurb))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    cfgp = os.path.splitext(args.out)[0] + ".config.json"
    with open(cfgp, "w") as f:
        json.dump(dict(obj=meta["obj"], name=args.name, T_mesh_world=T.tolist(),
                       inspect=info, drift=drift, shards=shards,
                       atlas_px=args.atlas_px, world_bounds=[lo.tolist(), hi.tolist()]),
                  f, indent=2)

    print("  %-22s %s" % ("viewer", args.out))
    print("  %-22s %.2f MB" % ("size", os.path.getsize(args.out) / 1e6))
    print("  %-22s %s" % ("frame config", cfgp))
    for k, v in stats:
        print("  %-22s %s" % (k, v))


if __name__ == "__main__":
    main()
