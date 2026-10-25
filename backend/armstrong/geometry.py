"""Compiled MuJoCo visual geometry for the browser renderer (Workcell Studio format).

Exports what the simulation actually uses: every visual geom of the robot, the table and the camera rig from
the compiled model (mesh vertices are taken after compilation, so MuJoCo's mesh re-centring is already folded
into geom_pos/geom_quat). The browser places each geom in its body frame and moves the bodies with the poses
the server streams. Objects on the table are drawn by the browser from the scene.

Binary layout: for every mesh, float32 positions (3 per vertex), int8 normals (3 per vertex, scaled by 127)
and uint16 or uint32 triangle indices, each block padded to 4 bytes. The manifest gives byte offsets.
"""

from __future__ import annotations

import mujoco
import numpy as np

GEOM_TYPES = {int(mujoco.mjtGeom.mjGEOM_PLANE): "plane", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
              int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
              int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder", int(mujoco.mjtGeom.mjGEOM_BOX): "box",
              int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}
DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0])


def _pad(buf: bytearray) -> None:
    buf.extend(b"\0" * (-len(buf) % 4))


def export(m: mujoco.MjModel, bodies: list[str], max_group: dict[str, int] | None = None) -> tuple[dict, bytes]:
    """Visual geoms (groups 0-2, or up to max_group[body]) of the named bodies."""
    max_group = max_group or {}
    wanted = {m.body(b).id: b for b in bodies}
    blob = bytearray()
    meshes: dict[int, int] = {}
    mesh_list, geoms = [], []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in wanted or m.geom_group[g] > max_group.get(wanted[b], 2) or m.geom_rgba[g][3] == 0:
            continue
        gtype = int(m.geom_type[g])
        mat = int(m.geom_matid[g])
        rgba = np.array(m.geom_rgba[g], float)
        if mat >= 0 and np.allclose(rgba, DEFAULT_RGBA):
            rgba = np.array(m.mat_rgba[mat], float)
        item = {"body": wanted[b], "name": m.geom(g).name, "type": GEOM_TYPES.get(gtype, "unknown"),
                "size": np.round(m.geom_size[g], 6).tolist(), "pos": np.round(m.geom_pos[g], 6).tolist(),
                "quat": np.round(m.geom_quat[g], 7).tolist(), "rgba": np.round(rgba, 4).tolist(),
                "material": None if mat < 0 else m.material(mat).name,
                "specular": None if mat < 0 else float(m.mat_specular[mat]),
                "shininess": None if mat < 0 else float(m.mat_shininess[mat])}
        if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
            k = int(m.geom_dataid[g])
            if k not in meshes:
                meshes[k] = len(mesh_list)
                mesh_list.append(_mesh_block(m, k, blob))
            item["mesh"] = meshes[k]
        geoms.append(item)
    manifest = {"bodies": bodies, "geoms": geoms, "meshes": mesh_list, "byte_length": len(blob),
                "conventions": {"up": "z", "units": "m", "quat": "wxyz"}}
    return manifest, bytes(blob)


def _mesh_block(m: mujoco.MjModel, k: int, blob: bytearray) -> dict:
    va, vn = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
    fa, fn = int(m.mesh_faceadr[k]), int(m.mesh_facenum[k])
    verts = np.asarray(m.mesh_vert[va:va + vn], np.float32)
    faces = np.asarray(m.mesh_face[fa:fa + fn], np.int64)
    fnorm = np.asarray(m.mesh_facenormal[fa:fa + fn], np.int64)
    na, nn = int(m.mesh_normaladr[k]), int(m.mesh_normalnum[k])
    normals = np.asarray(m.mesh_normal[na:na + nn], np.float32)
    if nn == vn and np.array_equal(faces, fnorm):
        vnorm = normals
    else:  # normals indexed separately: split corners into (vertex, normal) pairs
        pairs, inverse = np.unique(np.stack([faces.ravel(), fnorm.ravel()], 1), axis=0, return_inverse=True)
        verts, vnorm = verts[pairs[:, 0]], normals[pairs[:, 1]]
        faces = inverse.reshape(-1, 3)
        vn = len(verts)
    lengths = np.linalg.norm(vnorm, axis=1, keepdims=True)
    vnorm = vnorm / np.maximum(lengths, 1e-12)
    out = {"name": m.mesh(k).name, "vertex_count": vn, "index_count": int(faces.size)}
    out["positions"] = len(blob)
    blob.extend(verts.astype("<f4").tobytes())
    _pad(blob)
    out["normals"] = len(blob)
    blob.extend(np.clip(np.round(vnorm * 127.0), -127, 127).astype(np.int8).tobytes())
    _pad(blob)
    out["indices"] = len(blob)
    if vn < 65536:
        out["index_type"] = "uint16"
        blob.extend(faces.astype("<u2").tobytes())
    else:
        out["index_type"] = "uint32"
        blob.extend(faces.astype("<u4").tobytes())
    _pad(blob)
    return out
