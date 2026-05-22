"""In-scene camera display screens.

Creates two self-illuminated quad panels in the simulation scene showing live
feeds from the wrist and table cameras. Panels appear in the VR headset (via
WiVRn) and in the desktop Isaac Sim viewport.

Reads camera data directly from IsaacLab sensor buffers — no Replicator render
products are created, keeping VRAM usage minimal.

These are operator visual aids only — not part of the HDF5 dataset.

Usage (after env.reset()):
    from vr_camera_screens import VRCameraScreens
    screens = VRCameraScreens.try_create(env)

Usage (in sim loop):
    if screens:
        screens.update(env)
"""

from __future__ import annotations

import numpy as np


class VRCameraScreens:
    # Operator sits at (-0.8, 0, 0.3) looking toward +X.
    # Screens float between operator and robot, side by side at eye level.
    _WRIST_CENTER = (-0.3, -0.35, 1.0)  # operator's right
    _TABLE_CENTER = (-0.3,  0.35, 1.0)  # operator's left
    _SIZE = 0.35   # metres, square
    _RES  = (84, 84)   # match recording resolution — avoids extra render overhead

    def __init__(self):
        import omni.ui as ui
        import omni.usd

        self._stage = omni.usd.get_context().get_stage()

        self._tex_wrist = ui.DynamicTextureProvider("vr_wrist_cam")
        self._tex_table = ui.DynamicTextureProvider("vr_table_cam")

        self._make_screen("/World/VRScreen_Wrist", "vr_wrist_cam", self._WRIST_CENTER)
        self._make_screen("/World/VRScreen_Table", "vr_table_cam", self._TABLE_CENTER)

        print("[VRScreens] Wrist and table camera screens created")

    @classmethod
    def try_create(cls, env) -> "VRCameraScreens | None":
        """Create screens if camera prims exist in the current stage. Returns None silently if not."""
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        # Prim paths depend on the env namespace — single env is always env_0.
        wrist = "/World/envs/env_0/Robot/panda_hand/wrist_cam"
        table = "/World/envs/env_0/table_cam"

        if not stage.GetPrimAtPath(wrist).IsValid():
            print(f"[VRScreens] wrist_cam prim not found — screens skipped (non-visuomotor task?)")
            return None

        try:
            return cls()
        except Exception as e:
            print(f"[VRScreens] Setup failed: {e}")
            return None

    def update(self, env=None) -> None:
        """Refresh both screen textures from IsaacLab sensor data. Call once per simulation step."""
        if env is None:
            return  # no data source without env
        try:
            wrist = env.scene["wrist_cam"].data.output["rgb"]  # (N, H, W, 3) uint8 tensor
            table = env.scene["table_cam"].data.output["rgb"]
            if wrist is not None and wrist.shape[0] > 0:
                self._push_array(wrist[0].cpu().numpy(), self._tex_wrist)
            if table is not None and table.shape[0] > 0:
                self._push_array(table[0].cpu().numpy(), self._tex_table)
        except Exception:
            pass  # sensor data not ready yet (e.g. before first step)

    def _push_array(self, data: np.ndarray, provider) -> None:
        """Push an (H, W, 3) uint8 numpy array to a DynamicTextureProvider."""
        if data is None or data.size == 0:
            return
        data = data.astype(np.uint8)
        h, w = data.shape[:2]
        rgba = np.concatenate(
            [data[..., :3], np.full((h, w, 1), 255, dtype=np.uint8)], axis=2
        )
        provider.set_data_array(rgba, [h, w])

    def _make_screen(self, prim_path: str, tex_name: str, center: tuple) -> None:
        from pxr import UsdGeom, UsdShade, Sdf, Gf

        cx, cy, cz = center
        s = self._SIZE / 2

        mesh = UsdGeom.Mesh.Define(self._stage, prim_path)
        mesh.CreateDoubleSidedAttr(True)

        # Quad in the YZ plane at x=cx, facing toward the operator (-X direction).
        mesh.CreatePointsAttr([
            Gf.Vec3f(cx, cy - s, cz - s),  # 0 bottom-left
            Gf.Vec3f(cx, cy + s, cz - s),  # 1 bottom-right
            Gf.Vec3f(cx, cy + s, cz + s),  # 2 top-right
            Gf.Vec3f(cx, cy - s, cz + s),  # 3 top-left
        ])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(-1, 0, 0)] * 4)

        # Flip V: annotator origin is top-left; USD UV origin is bottom-left.
        pvars = UsdGeom.PrimvarsAPI(mesh)
        st = pvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([(0, 1), (1, 1), (1, 0), (0, 0)])

        # Self-illuminated OmniPBR material so the screen glows regardless of scene lighting.
        mat = UsdShade.Material.Define(self._stage, prim_path + "_Mat")
        sh  = UsdShade.Shader.Define(self._stage, prim_path + "_Mat/Shader")
        sh.CreateIdAttr("OmniPBR")
        sh.CreateInput("diffuse_texture",        Sdf.ValueTypeNames.Asset).Set(f"dynamic://{tex_name}")
        sh.CreateInput("enable_emission",        Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("emissive_color",         Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
        sh.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(f"dynamic://{tex_name}")
        sh.CreateInput("emissive_intensity",     Sdf.ValueTypeNames.Float).Set(2.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh).Bind(mat)


class VRHandMarker:
    """Emissive sphere at the operator's real wrist position.

    Green = recording active, red = recording paused.
    Gives the operator a visible reference for hand position in the sim world,
    and a clear recording-state indicator without needing a separate panel.
    """

    _RADIUS = 0.025  # metres

    def __init__(self, prim_path: str = "/World/VRHandMarker"):
        from pxr import UsdGeom, UsdShade, Sdf, Gf
        import omni.usd

        self._stage = omni.usd.get_context().get_stage()
        self._prim_path = prim_path

        sphere = UsdGeom.Sphere.Define(self._stage, prim_path)
        sphere.CreateRadiusAttr(self._RADIUS)

        self._translate_op = UsdGeom.Xformable(sphere).AddTranslateOp()
        self._translate_op.Set(Gf.Vec3d(0, 0, 0))

        # UsdPreviewSurface — works without MDL, emissive colour visible in viewport.
        mat = UsdShade.Material.Define(self._stage, prim_path + "_Mat")
        sh  = UsdShade.Shader.Define(self._stage, prim_path + "_Mat/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness",  Sdf.ValueTypeNames.Float).Set(1.0)
        sh.CreateInput("metallic",   Sdf.ValueTypeNames.Float).Set(0.0)
        self._emissive_input = sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
        self._emissive_input.Set(Gf.Vec3f(0.0, 1.0, 0.0))  # default green
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(sphere).Bind(mat)

        print("[VRHandMarker] Created at", prim_path)

    def update(self, wrist_pos: tuple, is_recording: bool) -> None:
        """Call once per sim step with the current wrist world position and recording state."""
        from pxr import Gf
        self._translate_op.Set(Gf.Vec3d(float(wrist_pos[0]), float(wrist_pos[1]), float(wrist_pos[2])))
        color = (0.0, 1.0, 0.0) if is_recording else (1.0, 0.0, 0.0)
        self._emissive_input.Set(Gf.Vec3f(*color))
