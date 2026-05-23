"""Recolour tool USDs for the pegboard scene.

Creates colour variants of the 4 pick objects so they are visually distinct
without relying on the original copyrighted textures/colours. Geometry and
physics are unchanged — only the material diffuse colour is overwritten.

Run with:
    /path/to/IsaacLab/isaaclab.sh -p /path/to/ur_pick/scripts/recolor_assets.py

Outputs new USDs alongside the originals, suffixed with the colour name
(e.g. scissors_ring_red.usd). The originals are not modified.
"""

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

from pathlib import Path
import shutil

from pxr import Gf, Sdf, Usd, UsdShade
import omni.usd

ASSETS = Path(__file__).resolve().parents[1] / "exported_assets" / "object"

# Object file → (output suffix, RGB colour 0-1)
RECOLOUR_PLAN = [
    ("tooth_brush.usd",         "green",   (0.05, 0.80, 0.15)),
    ("scissors_ring.usd",       "red",     (0.90, 0.08, 0.08)),
    ("silicone_tube_ring.usd",  "blue",    (0.10, 0.35, 0.95)),
    ("pliers_ring.usd",         "orange",  (0.95, 0.55, 0.05)),
]


def set_diffuse_colour(stage: Usd.Stage, rgb: tuple) -> int:
    """Try to set diffuse colour on every shader prim in the stage.

    Handles both OmniPBR (MDL) and UsdPreviewSurface shaders.
    Returns the number of shaders successfully updated.
    """
    colour = Gf.Vec3f(*rgb)
    updated = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)

        # OmniPBR / OmniGlass MDL shaders
        src_asset = prim.GetAttribute("info:mdl:sourceAsset")
        if src_asset and src_asset.IsValid():
            # diffuse_color_constant is the flat colour when no texture is bound
            attr = prim.GetAttribute("inputs:diffuse_color_constant")
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute(
                    "inputs:diffuse_color_constant",
                    Sdf.ValueTypeNames.Color3f,
                )
            attr.Set(colour)
            # Also clear any diffuse texture so the flat colour shows
            tex_attr = prim.GetAttribute("inputs:diffuse_texture")
            if tex_attr and tex_attr.IsValid():
                tex_attr.Set("")
            updated += 1
            continue

        # UsdPreviewSurface
        shader_id = shader.GetIdAttr()
        if shader_id and shader_id.Get() == "UsdPreviewSurface":
            inp = shader.GetInput("diffuseColor")
            if not inp:
                inp = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            inp.Set(colour)
            updated += 1

    return updated


for src_fname, colour_name, rgb in RECOLOUR_PLAN:
    src_path = ASSETS / src_fname
    if not src_path.exists():
        print(f"[SKIP] {src_fname} not found")
        continue

    stem = src_path.stem
    dst_path = ASSETS / f"{stem}_{colour_name}.usd"

    shutil.copy2(src_path, dst_path)
    print(f"[COPY] {src_path.name} → {dst_path.name}")

    ctx = omni.usd.get_context()
    ctx.open_stage(str(dst_path))
    stage = ctx.get_stage()

    n = set_diffuse_colour(stage, rgb)
    stage.Save()
    ctx.close_stage()

    print(f"  → updated {n} shader(s) to {colour_name} {rgb}")

print("\nDone. New USDs written to:", ASSETS)
app.close()
