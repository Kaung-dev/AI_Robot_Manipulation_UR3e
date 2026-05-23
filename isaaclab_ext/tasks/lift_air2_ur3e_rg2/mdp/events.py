"""Reset-time event: randomize which 4 of 8 hooks each object occupies.
Startup-time event: tag basket and table prims with dedicated semantic classes
so the segmentation CNN can distinguish them from generic 'environment'."""
import torch


_semantics_done = False


def tag_basket_and_table(env, env_ids=None):
    """Override the broad 'environment' tag on specific prims (basket, table).

    Runs once. Walks the loaded stage looking for SM_BoxPortableD (the basket)
    and the SeattleLabTable mesh — applies a Semantics API tag that the
    colorized segmentation annotator will pick up.
    """
    global _semantics_done
    if _semantics_done:
        return
    try:
        from pxr import Semantics  # type: ignore
    except ImportError:
        print("[WARN semantics] pxr.Semantics unavailable; skipping basket/table tag", flush=True)
        _semantics_done = True
        return

    stage = env.sim.stage
    tagged = {"basket": [], "table": []}
    table_candidates = []  # any prim with 'table' in its name, at any depth

    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if "/World/envs/env_" not in path_str:
            continue
        name_lower = prim.GetName().lower()

        # Basket: SM_BoxPortableD prim and its instance child
        if "sm_boxportabled" in name_lower or "boxportable" in name_lower:
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics_basket")
            sem.CreateSemanticTypeAttr().Set("class")
            sem.CreateSemanticDataAttr().Set("basket")
            tagged["basket"].append(path_str)
            continue

        # Table: anything with 'table' or 'seattlelab' in the name, anywhere
        if "table" in name_lower or "seattlelab" in name_lower:
            table_candidates.append(path_str)
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics_table")
            sem.CreateSemanticTypeAttr().Set("class")
            sem.CreateSemanticDataAttr().Set("table")
            tagged["table"].append(path_str)

    print(f"[DEBUG semantics] Tagged {len(tagged['basket'])} basket prims, {len(tagged['table'])} table prims", flush=True)
    for cls, paths in tagged.items():
        for p in paths[:5]:
            print(f"  {cls}: {p}", flush=True)

    _semantics_done = True

# Hook world positions as reported by inspect_air2_hooks.py.
HOOK_POSITIONS = [
    (-3.7500, -5.9, 1.55000),   # hook_01
    (-3.8850, -5.9, 1.65000),   # hook_02
    (-4.0600, -5.9, 1.59000),   # hook_03
    (-4.3656, -5.9, 1.34000),   # hook_04
    (-4.2620, -5.9, 1.49600),   # hook_05
    (-4.0910, -5.9, 1.34000),   # hook_06
    (-3.9220, -5.9, 1.29200),   # hook_07
    (-4.4330, -5.9, 1.54300),   # hook_08
]

_OBJECT_NAMES = ["object", "tool_pliers", "tool_scissors", "tool_silicone"]


_debug_printed = False


def reset_objects_on_hooks(env, env_ids: torch.Tensor):
    """Assign each of the 4 objects to a unique randomly chosen hook per env."""
    global _debug_printed
    hooks = torch.tensor(HOOK_POSITIONS, device=env.device)  # [8, 3]
    n = len(env_ids)

    # argsort of uniform noise gives a unique random permutation of 0..7 per env.
    perms = torch.argsort(torch.rand(n, 8, device=env.device), dim=1)  # [n, 8]

    # Try each in turn until objects look correct on the hooks:
    #   no rotation:   [1.0, 0.0, 0.0, 0.0]
    #   90° around X:  [0.7071, 0.7071, 0.0,   0.0  ]  (tilts forward/back)
    #   90° around Y:  [0.7071, 0.0,   0.7071, 0.0  ]  (spins left/right)
    #   90° around Z:  [0.7071, 0.0,   0.0,   0.7071]  (rolls sideways)
    object_quat = torch.tensor([0.7071, 0.0,   0.0,   0.7071], device=env.device).expand(n, -1)

    if not _debug_printed:
        print(f"[DEBUG hooks] env_origins[env_ids[0]] = {env.scene.env_origins[env_ids[0]].tolist()}")
        for i, name in enumerate(_OBJECT_NAMES):
            chosen = hooks[perms[0, i]].tolist()
            world = (env.scene.env_origins[env_ids[0]] + hooks[perms[0, i]]).tolist()
            print(f"[DEBUG hooks] {name}: hook_local={chosen}  world={world}")

        # Read actual simulation stage transforms for hooks and board
        try:
            from pxr import UsdGeom, Usd
            stage = env.sim.stage
            print("[DEBUG stage] Scanning stage for hook prims...", flush=True)
            for prim in stage.Traverse():
                name_lower = prim.GetName().lower()
                if "hook" in name_lower or "pegboard" in name_lower:
                    xf = UsdGeom.Xformable(prim)
                    if xf:
                        t = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                        print(f"[DEBUG stage] {prim.GetPath()}  sim=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})", flush=True)
        except Exception as e:
            print(f"[DEBUG stage] Error: {e}", flush=True)

        _debug_printed = True

    for i, name in enumerate(_OBJECT_NAMES):
        asset = env.scene[name]
        positions = env.scene.env_origins[env_ids] + hooks[perms[:, i]]
        asset.write_root_pose_to_sim(
            torch.cat([positions, object_quat], dim=-1), env_ids=env_ids
        )
        asset.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=env.device), env_ids=env_ids
        )
