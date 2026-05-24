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
    tagged = {"basket": 0, "table": 0, "skipped": 0}

    def _set_class(prim, class_name: str):
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set(class_name)

    # Walk the per-env Environment subtree (which holds the AIR2 payload —
    # walls, floor, pegboard, hooks, basket). Tag the basket as 'basket' and
    # everything else as 'table'. Skip prims that already have a more-specific
    # semantic from spawn-time tags (robot, tools).
    already_tagged_paths = []
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if "/World/envs/env_" not in path_str:
            continue
        if "/Environment" not in path_str:
            continue
        # Skip prims that aren't part of the broad scene payload — robot lives
        # under .../env_N/Robot, tools under .../env_N/Object etc., not under
        # /Environment. So this loop only touches the AIR2 scene contents.
        name_lower = prim.GetName().lower()
        if "sm_boxportabled" in name_lower or "boxportable" in name_lower:
            _set_class(prim, "basket")
            tagged["basket"] += 1
        else:
            # Default: tag as 'table'. This includes pegboard, walls, floor,
            # hooks — anything that's not the basket. The CNN learns to
            # distinguish basket from this broader 'table' background.
            _set_class(prim, "table")
            tagged["table"] += 1

    print(f"[DEBUG semantics] Tagged {tagged['basket']} basket prims + {tagged['table']} table prims (clean-slate post-load)", flush=True)

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
