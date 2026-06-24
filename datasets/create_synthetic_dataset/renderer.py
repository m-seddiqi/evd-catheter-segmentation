from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
from PIL import Image

# NumPy 2.0 compatibility for older pyrender versions.
if not hasattr(np, "infty"):
    np.infty = np.inf

# Prefer software headless OpenGL when no X display is available.
if "DISPLAY" not in os.environ and "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
if "MESA_SHADER_CACHE_DIR" not in os.environ:
    os.environ["MESA_SHADER_CACHE_DIR"] = "/tmp/mesa_shader_cache"

import pyrender

from .config import CameraConfig, SceneSynthConfig
from .sampler import SceneSpec


def _rotation_matrix_xyz_deg(rotation_deg: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.deg2rad(rotation_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array(
        [[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    Ry = np.array(
        [[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    Rz = np.array(
        [[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    return Rz @ Ry @ Rx


def _transform_matrix(rotation_deg: np.ndarray, translation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    S = np.diag([scale[0], scale[1], scale[2], 1.0])
    R = _rotation_matrix_xyz_deg(rotation_deg)
    T = np.eye(4, dtype=float)
    T[:3, 3] = translation
    return T @ R @ S


def _rotation_pose(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=float)
    Ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=float)
    Rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def _make_camera_pose(camera_cfg: CameraConfig) -> np.ndarray:
    eye = np.asarray(camera_cfg.position, dtype=float)
    target = np.asarray(camera_cfg.target, dtype=float)
    up = np.asarray(camera_cfg.up, dtype=float)

    forward = target - eye
    forward /= max(np.linalg.norm(forward), 1e-9)

    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-9)

    true_up = np.cross(right, forward)
    true_up /= max(np.linalg.norm(true_up), 1e-9)

    pose = np.eye(4, dtype=float)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _scene_to_trimesh_list(scene_or_mesh) -> List[trimesh.Trimesh]:
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return [scene_or_mesh.copy()]
    if isinstance(scene_or_mesh, trimesh.Scene):
        # Avoid graph internals: API differs across trimesh versions.
        dumped = scene_or_mesh.dump(concatenate=False)
        if isinstance(dumped, list):
            meshes = [m.copy() for m in dumped if isinstance(m, trimesh.Trimesh)]
            if meshes:
                return meshes

        # Fallback for newer trimesh APIs.
        geom = scene_or_mesh.to_geometry()
        if isinstance(geom, trimesh.Trimesh):
            return [geom.copy()]
        if isinstance(geom, list):
            meshes = [m.copy() for m in geom if isinstance(m, trimesh.Trimesh)]
            if meshes:
                return meshes
        raise ValueError("Unable to extract meshes from trimesh.Scene.")
    raise TypeError(f"Unsupported type: {type(scene_or_mesh)}")


def _ensure_mesh_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    if mesh.faces.shape[0] == 0:
        return mesh
    mesh.fix_normals(multibody=True)
    _ = mesh.vertex_normals
    return mesh


def _load_textured_mesh(glb_path: Path, smooth_shading: bool) -> pyrender.Mesh:
    loaded = trimesh.load(glb_path, force="scene")
    mesh_list = [_ensure_mesh_normals(m) for m in _scene_to_trimesh_list(loaded)]
    return pyrender.Mesh.from_trimesh(mesh_list, smooth=smooth_shading)


def _prepare_background(path: Path, size: Tuple[int, int], crop_ratio: float) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # Center-crop before resize (e.g. 0.80 keeps 80% of width/height).
    cr = float(max(0.05, min(1.0, crop_ratio)))
    cw = max(1, int(round(w * cr)))
    ch = max(1, int(round(h * cr)))
    left = (w - cw) // 2
    top = (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    img = img.resize(size, resample=Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _composite_rgba_on_background(fg_rgba: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    fg = fg_rgba[:, :, :3].astype(np.float32)
    alpha = (fg_rgba[:, :, 3:4].astype(np.float32) / 255.0)
    bg = bg_rgb.astype(np.float32)
    out = fg * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_flags(enable_shadows: bool) -> int:
    flags = int(pyrender.RenderFlags.RGBA)
    if enable_shadows and hasattr(pyrender.RenderFlags, "SHADOWS_DIRECTIONAL"):
        flags |= int(pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    if hasattr(pyrender.RenderFlags, "ALL_SOLID"):
        flags |= int(pyrender.RenderFlags.ALL_SOLID)
    return flags


def _override_material(color_rgb: Tuple[float, float, float]) -> pyrender.MetallicRoughnessMaterial:
    r, g, b = color_rgb
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[float(r), float(g), float(b), 1.0],
        metallicFactor=0.85,
        roughnessFactor=0.28,
    )


def _build_scene(cfg: SceneSynthConfig, objects) -> pyrender.Scene:
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
    for obj in objects:
        glb_path = cfg.paths.model_root / obj.rel_path
        if obj.color_override is None:
            mesh = _load_textured_mesh(glb_path, smooth_shading=cfg.smooth_shading)
        else:
            loaded = trimesh.load(glb_path, force="scene")
            mesh_list = [_ensure_mesh_normals(m) for m in _scene_to_trimesh_list(loaded)]
            color_rgb = cfg.catheter_color_palette[obj.color_override]
            mesh = pyrender.Mesh.from_trimesh(
                mesh_list,
                smooth=cfg.smooth_shading,
                material=_override_material(color_rgb),
            )
        pose = _transform_matrix(obj.rotation_deg, obj.translation, obj.scale)
        scene.add(mesh, pose=pose)

    cam = pyrender.PerspectiveCamera(
        yfov=np.deg2rad(cfg.camera.fov_y_deg),
        znear=cfg.camera.near,
        zfar=cfg.camera.far,
    )
    cam_pose = _make_camera_pose(cfg.camera)
    scene.add(cam, pose=cam_pose)

    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.6)
    rim_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.2)
    scene.add(key_light, pose=cam_pose @ _rotation_pose(-20.0, -25.0, 0.0))
    scene.add(fill_light, pose=cam_pose @ _rotation_pose(-10.0, 35.0, 0.0))
    scene.add(rim_light, pose=cam_pose @ _rotation_pose(15.0, 170.0, 0.0))
    return scene


def _render_scene_rgba(cfg: SceneSynthConfig, scene: pyrender.Scene) -> np.ndarray:
    ss = max(1, int(cfg.render_supersample))
    render_w = int(cfg.camera.width * ss)
    render_h = int(cfg.camera.height * ss)
    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=render_w, viewport_height=render_h)
    except Exception as e:
        platform = os.environ.get("PYOPENGL_PLATFORM", "<unset>")
        display = os.environ.get("DISPLAY", "<unset>")
        raise RuntimeError(
            "Failed to initialize offscreen OpenGL context. "
            f"PYOPENGL_PLATFORM={platform}, DISPLAY={display}. "
            "If running headless, try EGL with GPU access or run under Xvfb. "
            "Example: PYOPENGL_PLATFORM=egl python ...  OR  xvfb-run -s '-screen 0 1920x1080x24' python ..."
        ) from e
    try:
        color, depth = renderer.render(scene, flags=_render_flags(cfg.enable_directional_shadows))
    finally:
        renderer.delete()
    if ss > 1:
        color = np.array(
            Image.fromarray(color).resize((cfg.camera.width, cfg.camera.height), resample=Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
        depth = np.array(
            Image.fromarray(depth).resize((cfg.camera.width, cfg.camera.height), resample=Image.Resampling.NEAREST),
            dtype=np.float32,
        )
    # Some backends (notably OSMesa) can return RGB without alpha.
    # Build alpha from depth so compositing and masks still work.
    if color.ndim == 3 and color.shape[2] == 3:
        alpha = np.where(np.isfinite(depth) & (depth > 0), 255, 0).astype(np.uint8)
        color = np.dstack([color, alpha])
    return color


def render_scene_arrays(cfg: SceneSynthConfig, spec: SceneSpec) -> Tuple[np.ndarray, np.ndarray]:
    bg = _prepare_background(
        spec.background_path,
        size=(cfg.camera.width, cfg.camera.height),
        crop_ratio=cfg.background_crop_ratio,
    )

    catheter_objects = [obj for obj in spec.objects if obj.category == "catheter"]
    non_catheter_objects = [obj for obj in spec.objects if obj.category != "catheter"]
    if not catheter_objects:
        raise ValueError("SceneSpec must contain at least one catheter object.")

    composed_base = bg
    if non_catheter_objects:
        tool_scene = _build_scene(cfg, non_catheter_objects)
        tool_rgba = _render_scene_rgba(cfg, tool_scene)
        composed_base = _composite_rgba_on_background(tool_rgba, bg)

    catheter_scene = _build_scene(cfg, catheter_objects)
    catheter_rgba = _render_scene_rgba(cfg, catheter_scene)
    composed = _composite_rgba_on_background(catheter_rgba, composed_base)
    catheter_mask = catheter_rgba[:, :, 3]
    return composed, catheter_mask


def render_scene(cfg: SceneSynthConfig, spec: SceneSpec, output_path: Path) -> None:
    composed, _ = render_scene_arrays(cfg, spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composed).save(output_path)
