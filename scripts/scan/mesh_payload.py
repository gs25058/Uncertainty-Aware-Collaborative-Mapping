#!/usr/bin/env python3
"""Polycam OBJ -> a compact, quantised payload the compare viewer can inline.

Why not glTF/GLB: measured on the 9.9 scan, trimesh's GLB is 16 MB even with
1024px atlases, because float32 positions/normals/uvs + uint32 indices dominate.
The project's viewers are single-file HTML the user opens with a double click
(no http server, so `fetch` of a sibling .glb is blocked by file:// CORS), and a
23 MB base64 blob is not that. Quantising positions to uint16 over the mesh bbox
gives 0.6 mm steps over a 42 m scan -- 80x finer than the 0.05 m grid the map is
compared against -- and splitting chunks below 65536 vertices buys uint16
indices too, together a third of the bytes.

THE GEOMETRY IS NOT TOUCHED. No decimation, no hole filling, no floater
removal: this mesh is the one gt_voxel.npz gets voxelised from, so the viewer
has to show it as the GT sees it, warts included. The only things changed are
(1) the frozen T_mesh_world, so the scan lands in the same world frame as the
map, and (2) atlas resolution.

Material note: Polycam writes `Pm 1.00` (metalness = 1) and no Kd. In any PBR
renderer that makes the base colour specular-only, so with no environment map
the scan renders BLACK -- which is why the export looks nothing like the app.
The payload therefore carries the albedo atlas alone and the viewer draws it
unlit (MeshBasicMaterial), which is what Polycam's own viewer does.
"""
import base64
import io
import json

import numpy as np
import trimesh
from PIL import Image


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _atlas(image, px, quality):
    """Downscale an atlas. The 9.15 scan's six 4096^2 atlases carry 2.7 mm/texel
    against 12 cm triangles; at 1024 px that is 10.7 mm/texel, still an order of
    magnitude finer than the geometry, so the look is unaffected."""
    w, h = image.size
    s = min(1.0, px / float(max(w, h)))
    im = image.convert("RGB")
    if s < 1.0:
        im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    return im.size, buf.getvalue()


def split_chunk(F, limit=65536):
    """Partition faces so each part references < `limit` distinct vertices.

    Only so the viewer can use uint16 indices, which halves the single biggest
    item in the payload. Greedy over faces in file order, which for a Polycam
    export is already chunk-coherent, so the parts stay spatially compact.
    """
    parts, cur, seen = [], [], set()
    for f in F:
        new = [v for v in f if v not in seen]
        if len(seen) + len(new) >= limit and cur:
            parts.append(np.array(cur, np.int64)); cur, seen = [], set()
            new = list(f)
        seen.update(new); cur.append(f)
    if cur:
        parts.append(np.array(cur, np.int64))
    return parts


def build(obj_path, T_mesh_world, px=1024, quality=85):
    """-> dict: one entry per material group, positions/uvs quantised to uint16."""
    scene = trimesh.load(obj_path, process=False)
    geoms = (list(scene.geometry.values()) if hasattr(scene, "geometry")
             else [scene])
    T = np.asarray(T_mesh_world, float)

    for g in geoms:
        g.apply_transform(T)
    lo = np.min([g.vertices.min(axis=0) for g in geoms], axis=0)
    hi = np.max([g.vertices.max(axis=0) for g in geoms], axis=0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)

    chunks, n_v, n_f, n_tex = [], 0, 0, 0
    for g in geoms:
        V, F, uv = np.asarray(g.vertices), np.asarray(g.faces), np.asarray(g.visual.uv)
        size, jpg = _atlas(g.visual.material.image, px, quality)
        tex = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")
        n_tex += len(jpg)
        for part, Fp in enumerate(split_chunk(F)):
            keep, Fl = np.unique(Fp), None
            remap = np.zeros(len(V), np.int64); remap[keep] = np.arange(len(keep))
            Fl, Vp, uvp = remap[Fp], V[keep], uv[keep]
            q = np.clip(np.rint((Vp - lo) / span * 65535.0), 0, 65535).astype(np.uint16)
            # UVs get their own box: an atlas is padded, so the used range is a
            # sub-box of [0,1] and spending the full uint16 on it costs nothing.
            # v is NOT flipped -- OBJ's origin is bottom-left, which is what
            # three.js gives with the default texture.flipY = true.
            ulo, uhi = uvp.min(axis=0), uvp.max(axis=0)
            uspan = np.where(uhi - ulo > 1e-9, uhi - ulo, 1.0)
            qu = np.clip(np.rint((uvp - ulo) / uspan * 65535.0), 0, 65535).astype(np.uint16)
            assert len(Vp) < 65536, "chunk split failed: %d verts" % len(Vp)
            chunks.append(dict(
                pos=b64(q), uv=b64(qu), idx=b64(Fl.astype(np.uint16)),
                n_vert=int(len(Vp)), n_face=int(len(Fl)),
                uv_lo=[float(v) for v in ulo], uv_span=[float(v) for v in uspan],
                tex=tex if part == 0 else None, tex_ref=len(chunks) - part,
                tex_px=list(size)))
            n_v += len(Vp); n_f += len(Fl)

    return dict(lo=[float(v) for v in lo], span=[float(v) for v in span],
                chunks=chunks,
                stats=dict(n_vert=n_v, n_face=n_f, n_chunk=len(chunks),
                           tex_bytes=n_tex, atlas_px=px))


def load_config(path):
    cfg = json.load(open(path))
    return np.array(cfg["T_mesh_world"], float), cfg
