"""
OpenVLA 14-DoF LoRA Fine-Tuning Engine for Bimanual ALOHA
=========================================================
Trains OpenVLA-7B with Parameter-Efficient Fine-Tuning (LoRA Rank 32)
on the 30 Training Episodes of aloha_static_screw_driver (12,000 frames).

Outputs 14 action tokens per step (7 Left Arm + 7 Right Arm).
Saves the trained LoRA adapter to: ./openvla_aloha_output/model_a_raw_lora
"""

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image

# --- OpenVLA Compatibility Layer for Modern Transformers ---
import transformers
try:
    import transformers.tokenization_utils
    import transformers.tokenization_utils_base
    for attr in ["PaddingStrategy", "PreTokenizedInput", "TextInput", "TruncationStrategy"]:
        if not hasattr(transformers.tokenization_utils, attr) and hasattr(transformers.tokenization_utils_base, attr):
            setattr(transformers.tokenization_utils, attr, getattr(transformers.tokenization_utils_base, attr))
    import transformers.modeling_utils
    transformers.modeling_utils.PreTrainedModel._supports_sdpa = True
except Exception:
    pass

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_JSON = os.path.join(SCRIPT_DIR, "dataset_splits.json")

# Normalization bounds for ALOHA 14-DoF action space
ACTION_MIN = np.array([
    -0.0276, -0.9986,  0.9311, -0.4126, -1.1950, -0.2286, -0.0092,
    -0.2807, -0.9710, -0.0107, -0.1488, -0.9403, -0.3007, -0.0834
], dtype=np.float32)

ACTION_MAX = np.array([
     0.5354,  0.0046,  1.3530,  0.1227, -0.3037,  0.4034,  0.7713,
     0.2255,  0.5216,  1.1796,  0.1089,  0.7194,  0.3252,  0.7697
], dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA 14-DoF LoRA Training")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank dimension (default: 32)")
    parser.add_argument("--camera_mode", choices=["4cam", "cam_high"], default="4cam",
                        help="Camera input mode: '4cam' (2x2 mosaic) or 'cam_high' (overhead only)")
    parser.add_argument("--output_dir", type=str, default="./openvla_aloha_output/model_a_raw_lora", help="Output adapter dir")
    parser.add_argument("--cache_dir", type=str, default="./dataset_cache_4cam", help="Pre-extracted image cache directory")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token")
    return parser.parse_args()

def download_dataset(hf_token=None):
    from huggingface_hub import snapshot_download
    print("📥 Loading ALOHA Screwdriver dataset (Parquets + 4-Camera Videos)...")
    token = hf_token or os.environ.get("HF_TOKEN", None)
    repo_dir = snapshot_download(
        repo_id="lerobot/aloha_static_screw_driver",
        repo_type="dataset",
        allow_patterns=["*.parquet", "*.json", "meta/**", "videos/**"],
        token=token
    )
    parquet_files = glob.glob(os.path.join(repo_dir, "**", "*.parquet"), recursive=True)
    dfs = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Dataset ready: {len(df):,} total frames across {df['episode_index'].nunique()} episodes.")
    return df, repo_dir

def prepare_image_cache(repo_dir, train_episodes, cache_dir="./dataset_cache_4cam", mode="4cam"):
    import cv2
    os.makedirs(cache_dir, exist_ok=True)
    expected_frames = len(train_episodes) * 400

    existing = len(glob.glob(os.path.join(cache_dir, "*.jpg")))
    if existing >= expected_frames:
        print(f"⚡ Found {existing:,} pre-cached images in {cache_dir}. Skipping extraction.")
        return cache_dir

    print(f"🎬 Pre-decoding {expected_frames:,} {mode} frames into {cache_dir}...")
    def _find_vid(name):
        res = glob.glob(os.path.join(repo_dir, "**", f"*{name}*", "**", "*.mp4"), recursive=True)
        if not res:
            raise FileNotFoundError(f"Video stream '{name}' not found in {repo_dir}/videos")
        return res[0]

    cap_h = cv2.VideoCapture(_find_vid("cam_high"))
    cap_l = cv2.VideoCapture(_find_vid("cam_low")) if mode == "4cam" else None
    cap_lw = cv2.VideoCapture(_find_vid("cam_left_wrist")) if mode == "4cam" else None
    cap_rw = cv2.VideoCapture(_find_vid("cam_right_wrist")) if mode == "4cam" else None

    for ep_id in tqdm(train_episodes, desc=f"  Extracting {mode} Episodes"):
        start_frame = ep_id * 400
        cap_h.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if mode == "4cam":
            cap_l.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            cap_lw.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            cap_rw.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for f_offset in range(400):
            abs_idx = start_frame + f_offset
            save_path = os.path.join(cache_dir, f"frame_{abs_idx:06d}.jpg")
            if os.path.exists(save_path):
                continue

            _, fh = cap_h.read()
            if fh is None:
                raise RuntimeError(f"Corrupted cam_high frame at absolute index {abs_idx}")

            if mode == "cam_high":
                cv2.imwrite(save_path, fh, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                _, fl = cap_l.read()
                _, flw = cap_lw.read()
                _, frw = cap_rw.read()
                if fl is None or flw is None or frw is None:
                    raise RuntimeError(f"Corrupted wrist/low camera frame at index {abs_idx}")
                top = np.hstack([cv2.resize(fh, (320, 240)), cv2.resize(fl, (320, 240))])
                bot = np.hstack([cv2.resize(flw, (320, 240)), cv2.resize(frw, (320, 240))])
                grid = np.vstack([top, bot])
                cv2.imwrite(save_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])

    cap_h.release()
    if cap_l: cap_l.release()
    if cap_lw: cap_lw.release()
    if cap_rw: cap_rw.release()
    print(f"✅ Successfully cached {expected_frames:,} images in {cache_dir}!")
    return cache_dir

def actions_to_tokens(action_14, vocab_size=32000, num_bins=256):
    """Normalizes 14 continuous joint values to 256 discrete bins and maps to action token IDs."""
    norm = np.clip((action_14 - ACTION_MIN) / (ACTION_MAX - ACTION_MIN + 1e-6), 0.0, 1.0)
    bins = np.clip(np.floor(norm * num_bins).astype(int), 0, num_bins - 1)
    # OpenVLA maps action bins to the final 256 tokens in vocabulary
    action_token_ids = (vocab_size - num_bins) + bins
    return action_token_ids.tolist()

class AlohaDataset(Dataset):
    def __init__(self, df, train_episodes, cache_dir, processor):
        self.samples = []
        self.cache_dir = cache_dir
        train_df = df[df["episode_index"].isin(train_episodes)].sort_values(["episode_index", "frame_index"])
        
        for _, row in train_df.iterrows():
            act = row["action"]
            if act is not None and hasattr(act, "__len__") and len(act) == 14:
                abs_idx = int(row["index"])
                img_path = os.path.join(cache_dir, f"frame_{abs_idx:06d}.jpg")
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Missing image for training at {img_path}")
                self.samples.append({
                    "action": np.array(act, dtype=np.float32),
                    "img_path": img_path
                })
        
        self.processor = processor
        self.prompt = "In: What action should the robot take to pick up the yellow screwdriver and place it into the blue cup?\nOut:"
        print(f"✅ Prepared {len(self.samples):,} verified training samples across {len(train_episodes)} episodes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["img_path"]).convert("RGB")
        action_tokens = actions_to_tokens(item["action"])
        return image, self.prompt, action_tokens

def collate_fn(batch, processor):
    images, prompts, action_tokens_list = zip(*batch)
    
    # Process text and image inputs
    inputs = processor(text=list(prompts), images=list(images), return_tensors="pt", padding=True)
    
    # Target label masking: mask prompt tokens with -100, calculate loss only on the 14 action tokens
    input_ids = inputs["input_ids"]
    action_tensor = torch.tensor(action_tokens_list, dtype=torch.long)
    full_input_ids = torch.cat([input_ids, action_tensor], dim=1)
    
    # Keep attention mask consistent with expanded input_ids
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        action_mask = torch.ones_like(action_tensor)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], action_mask], dim=1)
    
    # Mask prompt positions
    labels_masked = torch.full_like(input_ids, -100)
    full_labels = torch.cat([labels_masked, action_tensor], dim=1)
    
    inputs["input_ids"] = full_input_ids
    inputs["labels"] = full_labels
    return inputs

def main():
    args = parse_args()
    print("=" * 80)
    print("🎯 OPENVLA 14-DoF BIMANUAL ALOHA LORA FINE-TUNING PIPELINE")
    print("=" * 80)

    # 1. Load Splits
    with open(SPLITS_JSON, "r") as f:
        splits = json.load(f)["splits"]
    train_eps = splits["train"]["episodes"]
    print(f"📂 Loaded 30 Training Episodes ({train_eps[0]} to {train_eps[-1]}).")

    # 2. Download Dataset
    df, repo_dir = download_dataset(args.hf_token)

    # 3. Cache and verify real camera vision frames
    cache_dir = prepare_image_cache(repo_dir, train_eps, cache_dir=args.cache_dir, mode=args.camera_mode)

    # 4. Initialize OpenVLA Processor & 4-bit Quantized Base Model
    print("🤖 Initializing OpenVLA Foundation Model in 4-bit...")
    from transformers import AutoProcessor, BitsAndBytesConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    MODEL_ID = "openvla/openvla-7b"
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # OpenVLA dynamic model class
    ModelClass = get_class_from_dynamic_module("modeling_prismatic.OpenVLAForActionPrediction", MODEL_ID)
    ModelClass._supports_sdpa = True
    ModelClass._supports_flash_attn_2 = False

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    base_model = ModelClass.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # 4. Attach LoRA Adapter for 14-DoF Bimanual Joint Prediction
    print(f"\n⚙️ Attaching LoRA Adapter (Rank={args.lora_rank}, Alpha={args.lora_rank*2})...")
    try:
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
    except Exception as e:
        print(f"Note on kbit preparation: {e}")

    # Ensure vision backbone and projector modules are strictly bfloat16 to match input tensors
    for mod_name in ["vision_backbone", "projector"]:
        m = getattr(base_model, mod_name, None)
        if m is not None:
            m.to(dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 5. Dataset & DataLoader
    dataset = AlohaDataset(df, train_eps, cache_dir, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=2
    )

    # 6. Optimizer & Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    print(f"\n🚀 Commencing 14-DoF Training across {args.epochs} Epochs on L4 GPU...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if torch.is_floating_point(v):
                        device_batch[k] = v.to("cuda", dtype=torch.bfloat16)
                    else:
                        device_batch[k] = v.to("cuda")
                else:
                    device_batch[k] = v
            batch = device_batch
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{epoch_loss / (step + 1):.4f}"})

        print(f"✅ Epoch {epoch} completed! Mean Loss: {epoch_loss / len(dataloader):.4f}")

    # 7. Save Model A Checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    print("\n" + "=" * 80)
    print(f"🎉 MODEL A (RAW 30-EPISODE LORA) SAVED SUCCESSFULLY!")
    print(f"💾 Checkpoint path: {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

