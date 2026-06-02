"""Run AIR2 segmentation inference on a saved image or live Isaac camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CNN_DIR = REPO_ROOT / "isaaclab_ext/tasks/air2_franka/cnn"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CNN_DIR))

from model import build_model, build_resnet_model
from postprocess import extract_detections


def _load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    backbone = checkpoint.get("backbone", "unet")
    num_classes = int(checkpoint.get("num_classes", 7))
    class_map = {int(key): value for key, value in checkpoint.get("class_map", {}).items()}
    if backbone == "resnet18":
        model = build_resnet_model(num_classes=num_classes, pretrained=False).to(device)
    else:
        base_channels = int(checkpoint.get("base_channels", 32))
        model = build_model(num_classes=num_classes, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_map


def _image_to_tensor(path: Path, image_size: int | None, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)


def _save_prediction_mask(logits: torch.Tensor, path: Path) -> None:
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(pred).save(path)


def run_image_mode(args) -> None:
    device = torch.device(args.device)
    model, class_map = _load_checkpoint(Path(args.checkpoint), device)
    image = _image_to_tensor(Path(args.image), args.image_size, device)
    depth = np.load(args.depth) if args.depth else None
    intrinsics = np.load(args.intrinsics) if args.intrinsics else None

    with torch.no_grad():
        logits = model(image)
    detections = extract_detections(
        logits,
        depth=depth,
        intrinsic_matrix=intrinsics,
        class_map=class_map,
        min_area=args.min_area,
        min_confidence=args.min_confidence,
    )
    if args.mask_output:
        _save_prediction_mask(logits, Path(args.mask_output))
    print(json.dumps(detections, indent=2))


def run_live_mode(args) -> None:
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    import isaaclab_ext.tasks.air2_franka  # noqa: F401
    import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    device = torch.device(args.device)
    model, class_map = _load_checkpoint(Path(args.checkpoint), device)
    env_cfg = parse_env_cfg(args.task, device=launcher_args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    action_dim = env.action_manager.total_action_dim
    dummy_action = torch.zeros(1, action_dim, dtype=torch.float32, device=env.device)
    for _ in range(args.settle_steps):
        env.step(dummy_action)

    camera = env.scene[args.camera]
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
    image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    depth = camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    intrinsics = camera.data.intrinsic_matrices[0].detach().cpu().numpy()
    pos_w = camera.data.pos_w[0].detach().cpu().numpy()
    rot_w_quat = camera.data.quat_w_ros[0].detach().cpu().numpy()

    with torch.no_grad():
        logits = model(image)
    detections = extract_detections(
        logits,
        depth=depth,
        intrinsic_matrix=intrinsics,
        pos_w=pos_w,
        rot_w_quat=rot_w_quat,
        class_map=class_map,
        min_area=args.min_area,
        min_confidence=args.min_confidence,
    )
    if args.mask_output:
        _save_prediction_mask(logits, Path(args.mask_output))
    print(json.dumps(detections, indent=2))
    env.close()
    simulation_app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AIR2 CNN segmentation inference.")
    parser.add_argument("--checkpoint", default="checkpoints/air2_segmentation_unet.pth")
    parser.add_argument("--image", help="Saved RGB image path. Omit for live Isaac mode.")
    parser.add_argument("--depth", help="Optional saved depth .npy for image mode.")
    parser.add_argument("--intrinsics", help="Optional camera intrinsics .npy for image mode.")
    parser.add_argument("--task", default="Isaac-AIR2-Robotis-Franka-Segmentation-v0")
    parser.add_argument("--camera", default="main_camera")
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min-area", type=int, default=32)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--mask-output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = parser.parse_known_args()

    if args.image:
        run_image_mode(args)
    else:
        run_live_mode(args)


if __name__ == "__main__":
    main()
