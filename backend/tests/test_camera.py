"""Camera calibration conventions and image-only localization of every assembly part."""
import mujoco
import numpy as np

from armstrong import scene as sc
from armstrong.camera import backproject, project
from armstrong.perception import estimate_object

from conftest import names


def catalog(o):
    return {"id": o.id, "name": o.name, "color": o.color, "size_m": list(o.size), "support_z_m": sc.TABLE_TOP_Z,
            "prior_xy_m": list(o.pos[:2]), "prior_yaw_rad": o.yaw}


def test_projection_convention_matches_rendered_pixels(worker, store):
    obs = worker.observe()
    o = store.objects[names(store)["Base block"]]
    top = np.array([o.pos[0], o.pos[1], sc.TABLE_TOP_Z + o.size[2]])
    uv, z = project(obs.T_world_camera, obs.K, top)
    u, v = int(round(uv[0])), int(round(uv[1]))
    assert 0 <= u < obs.rgb.shape[1] and 0 <= v < obs.rgb.shape[0]
    r, g, b = obs.rgb[v, u].astype(int)
    assert r > 150 and r - b > 80, "the known top of the orange block must project onto orange pixels"
    assert abs(float(obs.depth_m[v, u]) - z) < 0.004, "depth must be optical-axis Z in meters"
    assert u < obs.rgb.shape[1] / 2, "world -y (the block is at y < 0) appears in the left half of the image"
    p = backproject(obs.depth_m, obs.K, obs.T_world_camera, np.array([u]), np.array([v]))[0]
    assert np.linalg.norm(p - top) < 0.006


def test_optical_frame_is_right_handed_and_looks_at_table(worker):
    obs = worker.observe()
    R = obs.T_world_camera[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    tgt = np.asarray(worker.cfg.camera.target) - np.asarray(worker.cfg.camera.position)
    assert np.dot(R[:, 2], tgt / np.linalg.norm(tgt)) > 0.999
    assert R[2, 1] < 0, "image-down (+Y optical) must point downward in the world"


def test_every_part_is_localized_from_the_image(worker, store):
    obs = worker.observe()
    for o in store.objects.values():
        est = estimate_object(obs, catalog(o), worker.scene_revision)
        assert est["valid"], (o.name, est["reasons"])
        truth = worker.data.xpos[worker.ids.objects[o.id].body]
        assert np.linalg.norm(np.asarray(est["position_m"][:2]) - truth[:2]) < 0.006, o.name
        assert abs(est["position_m"][2] - truth[2]) < 0.006, o.name


def test_estimate_follows_the_object_not_the_layout(worker, store):
    """The scene layout only sets the search window: an object pushed 2 cm is found where it really is."""
    oid = names(store)["Beam"]
    oi = worker.ids.objects[oid]
    worker.data.qpos[oi.qadr:oi.qadr + 2] += [0.02, -0.015]
    mujoco.mj_forward(worker.model, worker.data)
    est = estimate_object(worker.observe(), catalog(store.objects[oid]), worker.scene_revision)
    assert est["valid"], est["reasons"]
    assert np.linalg.norm(np.asarray(est["position_m"][:2]) - worker.data.xpos[oi.body][:2]) < 0.006


def test_rotated_rectangle_yaw(worker, store):
    oid = names(store)["Beam"]
    store.update_object(oid, {"yaw": 0.6})
    worker._sync_scene()
    est = estimate_object(worker.observe(), catalog(store.objects[oid]), worker.scene_revision)
    assert est["valid"], est["reasons"]
    assert abs(est["yaw_rad"] - 0.6) < np.radians(4)


def test_stale_and_masked_frames_are_invalid(worker, store):
    o = next(iter(store.objects.values()))
    obs = worker.observe()
    assert not estimate_object(obs, catalog(o), worker.scene_revision + 1)["valid"]
    worker.faults["camera_masked"] = True
    est = estimate_object(worker.observe(), catalog(o), worker.scene_revision)
    assert not est["valid"] and "unavailable" in est["reasons"][0]
