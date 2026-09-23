"""The textured, multi-geometry raycasting scene the sgbm depth mode needs.

PREREG_sgbm_depth.md §4. Nothing in the repository could render the sgbm mode:

  - ``covor.synth.mesh_gt.raycasting_scene`` adds ONE geometry, so every hit
    has geometry_id 0 and ``render._shade`` cannot tell the six texture atlases
    apart;
  - ``covor.synth.mesh_gt.load`` uses ``force="mesh"``, which concatenates the
    OBJ's materials into one mesh and drops the per-material UV/image pairing;
  - nothing passes ``textures=`` to ``DepthRenderer``.

This module is the wiring only. ``covor.synth.render.load_textures`` (unmodified)
does the per-material split; here the submeshes are added to one
RaycastingScene in the SAME order, so open3d's geometry_ids index the texture
list. The placement is the sequence's own ``mesh_config.json`` T_mesh_world --
the matrix the GT grid and the ideal scene were built with -- never re-derived.

VALUE CHECK (§4, before P0). The multi-geometry scene is the same surface
assembled differently, so its IDEAL depth must equal the single-mesh scene's
bit for bit. ``ideal_bitcheck`` measures that; render_depth_cache.py runs it on
every frame it renders.
"""
import json

import numpy as np


def textured_scene(mesh_config_path):
    """-> (open3d RaycastingScene with one geometry per material,
           [(uv (F,3,2), gray image)] -- what DepthRenderer(textures=...) and
           render._shade index by geometry_id)."""
    import open3d as o3d
    import trimesh
    from covor.synth.render import load_textures

    mc = json.load(open(mesh_config_path))
    T = np.array(mc["T_mesh_world"], float)
    # a Scene, NOT force="mesh": the materials must stay separate geometries
    scn = trimesh.load(mc["obj"], process=False)
    subs, tex = load_textures(scn, T)
    rs = o3d.t.geometry.RaycastingScene()
    for k, g in enumerate(subs):
        tm = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(g.vertices, float)),
            o3d.utility.Vector3iVector(np.asarray(g.faces, np.int32)))
        gid = rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tm))
        if int(gid) != k:
            raise RuntimeError("geometry id %d added at position %d: the texture "
                               "list would be indexed wrongly" % (gid, k))
    return rs, tex


def ideal_bitcheck(rend_single, rend_multi, T_wc):
    """Number of pixels whose ideal depth differs between the two scenes
    (NaN == NaN counts as equal). 0 is the only acceptable value."""
    Za, _, va = rend_single.depth_ideal(T_wc)
    Zb, _, vb = rend_multi.depth_ideal(T_wc)
    same = (va == vb) & ((Za == Zb) | (np.isnan(Za) & np.isnan(Zb)))
    return int((~same).sum())
