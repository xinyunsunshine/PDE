"""Pre-compute SigLIP image embeddings for cached SFT training.

For each frame in the dataset, runs the frozen SigLIP vision tower (with train=False
preprocessing) and saves the resulting embeddings to disk as bf16 safetensors.

Usage:
    cd /PROJECT_ROOT/repo
    source .venv-openpi/bin/activate
    python real/scripts/precompute_siglip_cache.py \
        --config-name pi05_fr3vla_pde_2cam \
        --output-dir data/siglip_cache/jmarangola/pde_real_sft \
        --batch-size 32

Output structure:
    {output_dir}/episode_{idx:06d}.safetensors  — per-episode embeddings
    {output_dir}/zeros_emb.safetensors          — embedding for masked (zeros) images
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, required=True,
                        help="OpenPI config name (e.g., pi05_fr3vla_pde_2cam)")
    parser.add_argument("--model-path", type=str, default="checkpoints/torch/pi05_base_hf")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--end-episode", type=int, default=-1)
    return parser.parse_args()


def load_model(model_path, config_name, device):
    """Load the pi0.5 model (only need SigLIP + projector)."""
    import glob

    import safetensors.torch as st

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    config = get_openpi_config(config_name, model_path=model_path, batch_size=1)
    actor_model_config = OpenPi0Config(
        **config.model.__dict__,
        train_expert_only=True,
        num_images_in_input=2,
    )

    model = OpenPi0ForRLActionPrediction(actor_model_config)
    checkpoint_dir = model_path
    weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not weight_paths:
        weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
    for wp in weight_paths:
        st.load_model(model, wp, strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    model = model.to(device)
    model.eval()
    return model, config


def load_dataset(config_name, model_path):
    """Load the raw LeRobot dataset (before openpi transforms)."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config = get_openpi_config(config_name, model_path=model_path, batch_size=1)
    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id = data_config.repo_id

    # Load from local path — pass root to avoid HF Hub lookups
    lerobot_home = os.environ.get("HF_LEROBOT_HOME", "data")
    local_root = os.path.join(lerobot_home, repo_id)
    dataset = LeRobotDataset(repo_id, root=local_root, tolerance_s=1e6, video_backend="pyav")
    return dataset, data_config


def get_episode_frames(dataset, episode_idx):
    """Get all frame indices for a given episode."""
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    to_idx = min(to_idx, len(dataset))
    return list(range(from_idx, to_idx))


def preprocess_and_embed(model, images_batch, device):
    """Run preprocessing (train=False) and SigLIP embed_image on a batch of images.

    images_batch: list of torch.Tensor [C, H, W] float32 (0-1) or np.ndarray [H, W, C] uint8
    Returns: tensor [B, 256, 2048] bf16
    """
    from openpi.shared import image_tools

    batch = []
    for img in images_batch:
        if isinstance(img, torch.Tensor):
            # [C, H, W] float32 0-1 from LeRobotDataset
            if img.dim() == 3 and img.shape[0] == 3:
                # Normalize 0-1 → -1 to 1
                img = img * 2.0 - 1.0
                # resize_with_pad expects [1, H, W, C], returns [H, W, C]
                if img.shape[1:3] != (224, 224):
                    img_hwc = img.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]
                    img_hwc = image_tools.resize_with_pad_torch(img_hwc, 224, 224)  # [224, 224, C]
                    img_t = img_hwc.permute(2, 0, 1).unsqueeze(0)  # [1, C, 224, 224]
                else:
                    img_t = img.unsqueeze(0)  # [1, C, H, W]
            else:
                raise ValueError(f"Unexpected tensor shape: {img.shape}")
        else:
            img = np.asarray(img)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            img_t = torch.from_numpy(img).unsqueeze(0)  # [1, H, W, C]
            if img_t.shape[1:3] != (224, 224):
                img_t = image_tools.resize_with_pad_torch(img_t, 224, 224)
            img_t = img_t.permute(0, 3, 1, 2)  # [1, C, H, W]
        batch.append(img_t)

    batch_tensor = torch.cat(batch, dim=0).to(device)  # [B, C, H, W]

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        emb = model.paligemma_with_expert.embed_image(batch_tensor)  # [B, 256, 2048]

    return emb.to(torch.bfloat16).cpu()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    model, config = load_model(args.model_path, args.config_name, device)

    print("Loading dataset...")
    dataset, data_config = load_dataset(args.config_name, args.model_path)
    num_episodes = dataset.num_episodes
    print(f"  Episodes: {num_episodes}, Total frames: {len(dataset)}")

    end_episode = args.end_episode if args.end_episode > 0 else num_episodes

    # Compute zeros-image embedding (for masked camera slots)
    print("Computing zeros-image embedding...")
    zeros_img = np.zeros((224, 224, 3), dtype=np.float32) * 2.0 - 1.0  # [-1, -1, -1]
    zeros_emb = preprocess_and_embed(model, [zeros_img], device)  # [1, 256, 2048]
    safetensors.torch.save_file(
        {"zeros_emb": zeros_emb.squeeze(0)},
        os.path.join(args.output_dir, "zeros_emb.safetensors"),
    )
    print(f"  Saved zeros_emb shape: {zeros_emb.shape[1:]}")

    # Determine image keys from dataset
    image_keys = []
    sample = dataset[0]
    if "observation.images.global" in sample:
        image_keys.append(("observation.images.global", "base_0_rgb"))
    elif "observation/image" in sample:
        image_keys.append(("observation/image", "base_0_rgb"))
    if "observation.images.wrist" in sample:
        image_keys.append(("observation.images.wrist", "left_wrist_0_rgb"))
    elif "observation/wrist_image" in sample:
        image_keys.append(("observation/wrist_image", "left_wrist_0_rgb"))
    print(f"  Image keys: {image_keys}")

    # Process episodes
    print(f"\nProcessing episodes {args.start_episode} to {end_episode}...")
    for ep_idx in tqdm(range(args.start_episode, end_episode), desc="Episodes"):
        output_path = os.path.join(args.output_dir, f"episode_{ep_idx:06d}.safetensors")
        if os.path.exists(output_path):
            continue

        frame_indices = get_episode_frames(dataset, ep_idx)
        num_frames = len(frame_indices)

        # Process each image key
        episode_embs = {}
        for dataset_key, emb_key in image_keys:
            all_embs = []

            for batch_start in range(0, num_frames, args.batch_size):
                batch_end = min(batch_start + args.batch_size, num_frames)
                batch_imgs = []
                for fi in frame_indices[batch_start:batch_end]:
                    frame = dataset[fi]
                    img = frame[dataset_key]
                    if not isinstance(img, torch.Tensor):
                        img = torch.as_tensor(np.asarray(img))
                    batch_imgs.append(img)

                embs = preprocess_and_embed(model, batch_imgs, device)
                all_embs.append(embs)

            episode_embs[emb_key] = torch.cat(all_embs, dim=0)  # [num_frames, 256, 2048]

        safetensors.torch.save_file(episode_embs, output_path)

    print(f"\nDone. Cache saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
