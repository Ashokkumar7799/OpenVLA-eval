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
    parser.add_argument("--output_dir", type=str, default="./openvla_aloha_output/model_a_raw_lora", help="Output adapter dir")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token")
    return parser.parse_args()

def download_dataset(hf_token=None):
    from huggingface_hub import snapshot_download
    print("📥 Loading ALOHA Screwdriver dataset...")
    token = hf_token or os.environ.get("HF_TOKEN", None)
    repo_dir = snapshot_download(
        repo_id="lerobot/aloha_static_screw_driver",
        repo_type="dataset",
        allow_patterns=["*.parquet", "*.json", "meta/**"],
        token=token
    )
    parquet_files = glob.glob(os.path.join(repo_dir, "**", "*.parquet"), recursive=True)
    dfs = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Dataset ready: {len(df):,} total frames across {df['episode_index'].nunique()} episodes.")
    return df

def actions_to_tokens(action_14, vocab_size=32000, num_bins=256):
    """Normalizes 14 continuous joint values to 256 discrete bins and maps to action token IDs."""
    norm = np.clip((action_14 - ACTION_MIN) / (ACTION_MAX - ACTION_MIN + 1e-6), 0.0, 1.0)
    bins = np.clip(np.floor(norm * num_bins).astype(int), 0, num_bins - 1)
    # OpenVLA maps action bins to the final 256 tokens in vocabulary
    action_token_ids = (vocab_size - num_bins) + bins
    return action_token_ids.tolist()

class AlohaDataset(Dataset):
    def __init__(self, df, train_episodes, processor):
        self.samples = []
        train_df = df[df["episode_index"].isin(train_episodes)].sort_values(["episode_index", "frame_index"])
        
        for _, row in train_df.iterrows():
            act = row["action"]
            if act is not None and hasattr(act, "__len__") and len(act) == 14:
                self.samples.append({
                    "action": np.array(act, dtype=np.float32),
                    "image": row.get("observation.images.cam_high", None)
                })
        
        self.processor = processor
        self.prompt = "In: What action should the robot take to pick up the yellow screwdriver and place it into the blue cup?\nOut:"
        print(f"✅ Prepared {len(self.samples):,} training steps across {len(train_episodes)} episodes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_img = item["image"]
        
        if isinstance(raw_img, dict) and "bytes" in raw_img:
            import io
            image = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
        elif isinstance(raw_img, np.ndarray):
            image = Image.fromarray(raw_img).convert("RGB")
        elif hasattr(raw_img, "convert"):
            image = raw_img.convert("RGB")
        else:
            image = Image.new("RGB", (480, 640), (0, 0, 0))

        action_tokens = actions_to_tokens(item["action"])
        return image, self.prompt, action_tokens

def collate_fn(batch, processor):
    images, prompts, action_tokens_list = zip(*batch)
    
    # Process text and image inputs
    inputs = processor(text=list(prompts), images=list(images), return_tensors="pt", padding=True)
    
    # Target label masking: mask prompt tokens with -100, calculate loss only on the 14 action tokens
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    
    # Append the 14 action token targets
    action_tensor = torch.tensor(action_tokens_list, dtype=torch.long)
    full_input_ids = torch.cat([input_ids, action_tensor], dim=1)
    
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
    df = download_dataset(args.hf_token)

    # 3. Initialize OpenVLA Processor & 4-bit Quantized Base Model
    print("🤖 Initializing OpenVLA Foundation Model in 4-bit...")
    from transformers import AutoProcessor, BitsAndBytesConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from peft import LoraConfig, get_peft_model

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
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # 4. Attach LoRA Adapter for 14-DoF Bimanual Joint Prediction
    print(f"\n⚙️ Attaching LoRA Adapter (Rank={args.lora_rank}, Alpha={args.lora_rank*2})...")
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
    dataset = AlohaDataset(df, train_eps, processor)
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
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
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

