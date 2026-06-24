"""Optional GLB diagnostic for mesh size, texture presence, and coarse scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh


GLB_PATTERNS = ("*.glb", "*.GLB")


def _flatten_meshes(scene_or_mesh) -> List[trimesh.Trimesh]:
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return [scene_or_mesh]
    if isinstance(scene_or_mesh, trimesh.Scene):
        dumped = scene_or_mesh.dump(concatenate=False)
        if isinstance(dumped, list):
            return [m for m in dumped if isinstance(m, trimesh.Trimesh)]
        geom = scene_or_mesh.to_geometry()
        if isinstance(geom, trimesh.Trimesh):
            return [geom]
        if isinstance(geom, list):
            return [m for m in geom if isinstance(m, trimesh.Trimesh)]
    return []


def _image_size(obj) -> Tuple[int, int] | None:
    if obj is None:
        return None
    if hasattr(obj, "size"):
        try:
            w, h = obj.size
            return int(w), int(h)
        except Exception:
            return None
    return None


def _texture_sizes(mesh: trimesh.Trimesh) -> List[Tuple[int, int]]:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return []
    material = getattr(visual, "material", None)
    if material is None:
        return []

    sizes: List[Tuple[int, int]] = []

    # Generic/simple material texture.
    maybe = _image_size(getattr(material, "image", None))
    if maybe:
        sizes.append(maybe)

    # GLTF PBR texture slots.
    for attr in (
        "baseColorTexture",
        "normalTexture",
        "emissiveTexture",
        "metallicRoughnessTexture",
        "occlusionTexture",
    ):
        maybe = _image_size(getattr(material, attr, None))
        if maybe:
            sizes.append(maybe)

    return sizes


def inspect_glb(path: Path) -> Dict:
    loaded = trimesh.load(path, force="scene")
    meshes = _flatten_meshes(loaded)
    if not meshes:
        return {"rel_path": str(path), "error": "No mesh found"}

    verts = int(sum(m.vertices.shape[0] for m in meshes))
    faces = int(sum(m.faces.shape[0] for m in meshes))
    verts_all = np.concatenate([m.vertices for m in meshes], axis=0)
    diag = float(np.linalg.norm(np.ptp(verts_all, axis=0)))
    tex_sizes: List[Tuple[int, int]] = []
    for m in meshes:
        tex_sizes.extend(_texture_sizes(m))
    max_tex = max((max(w, h) for w, h in tex_sizes), default=0)
    min_tex = min((min(w, h) for w, h in tex_sizes), default=0)

    issues: List[str] = []
    if faces < 2_000:
        issues.append("low_poly_faces<2000")
    if max_tex == 0:
        issues.append("no_texture_image_detected")
    elif max_tex < 512:
        issues.append("low_texture_max<512")
    if diag < 0.03:
        issues.append("very_small_bbox")

    return {
        "rel_path": str(path),
        "mesh_count": len(meshes),
        "vertices": verts,
        "faces": faces,
        "bbox_diag": round(diag, 6),
        "texture_sizes": tex_sizes,
        "max_texture": max_tex,
        "min_texture": min_tex,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect available GLB mesh/texture quality.")
    parser.add_argument(
        "--generator-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing this generator and its 3d_tools/ assets.",
    )
    parser.add_argument("--repo-root", dest="generator_root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: datasets/synthetic_dataset/e1/glb_quality_report.json)",
    )
    args = parser.parse_args()

    generator_root = args.generator_root
    model_root = generator_root / "3d_tools"
    categories = ("catheter", "clippers")

    all_files: List[Path] = []
    for c in categories:
        for pattern in GLB_PATTERNS:
            all_files.extend((model_root / c).glob(pattern))
    all_files = sorted(set(all_files))

    report = []
    for path in all_files:
        entry = inspect_glb(path)
        entry["rel_path"] = str(path.relative_to(model_root))
        report.append(entry)

    out = args.output
    if out is None:
        out = generator_root.parent / "synthetic_dataset" / "e1" / "glb_quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    flagged = [r for r in report if r.get("issues")]
    print(f"Analyzed {len(report)} glb files")
    print(f"Flagged {len(flagged)} files with potential quality issues")
    print(f"Saved report: {out}")
    for r in flagged[:10]:
        print(f"- {r['rel_path']}: {', '.join(r['issues'])}")


if __name__ == "__main__":
    main()
