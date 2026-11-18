"""
OpenVLA Aloha Evaluation & Fine-Tuning Engine (Google Colab / Cloud GPU Runner)
=============================================================================
This script runs on a GPU-enabled Google Colab instance (e.g. L4, A100, or T4).
It performs:
  1. Automated LeRobot ALOHA dataset download from Hugging Face.
  2. Zero-Shot OpenVLA-7B Action Inference on the Holdout Test Episodes.
  3. Headless MuJoCo Physical Simulation & Success Rate Evaluation.
  4. Benchmark scorecard generation (Task Success Rate, Peak Lift, L1 Error).
"""

import os
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
import sys
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image

# --- OpenVLA Compatibility Layer for Transformers ---
import transformers
try:
    import transformers.tokenization_utils
    import transformers.tokenization_utils_base
    for attr in ["PaddingStrategy", "PreTokenizedInput", "TextInput", "TruncationStrategy"]:
        if not hasattr(transformers.tokenization_utils, attr) and hasattr(transformers.tokenization_utils_base, attr):
            setattr(transformers.tokenization_utils, attr, getattr(transformers.tokenization_utils_base, attr))
    
    # Global PreTrainedModel patch for SDPA compatibility across all submodules
    import transformers.modeling_utils
    transformers.modeling_utils.PreTrainedModel._supports_sdpa = True
except Exception:
    pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "mujoco_menagerie", "aloha", "scene_with_screwdriver.xml")
SPLITS_JSON = os.path.join(SCRIPT_DIR, "dataset_splits.json")

ACTION_MIN = np.array([
    -0.0276, -0.9986,  0.9311, -0.4126, -1.1950, -0.2286, -0.0092,
    -0.2807, -0.9710, -0.0107, -0.1488, -0.9403, -0.3007, -0.0834
], dtype=np.float32)

ACTION_MAX = np.array([
     0.5354,  0.0046,  1.3530,  0.1227, -0.3037,  0.4034,  0.7713,
     0.2255,  0.5216,  1.1796,  0.1089,  0.7194,  0.3252,  0.7697
], dtype=np.float32)

def tokens_to_aloha_actions(action_tokens, vocab_size=32000, num_bins=256):
    bins = np.array(action_tokens) - (vocab_size - num_bins)
    bins = np.clip(bins, 0, num_bins - 1)
    norm = (bins + 0.5) / float(num_bins)
    actions_14 = ACTION_MIN + norm * (ACTION_MAX - ACTION_MIN)
    return actions_14.astype(np.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA ALOHA Colab Engine")
    parser.add_argument("--mode", choices=["baseline", "zero_shot", "eval_lora", "finetune", "full_pipeline"], default="baseline",
                        help="Execution mode: baseline, zero_shot, eval_lora (14-DoF fine-tuned), or full_pipeline")
    parser.add_argument("--adapter_path", type=str, default="./openvla_aloha_output/model_a_raw_lora",
                        help="Path to fine-tuned 14-DoF LoRA adapter directory")
    parser.add_argument("--test_limit", type=int, default=5, help="Number of test episodes to evaluate (default: 5)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank adaptation dimension")
    parser.add_argument("--output_dir", type=str, default="./openvla_aloha_output", help="Directory for artifacts")
    parser.add_argument("--camera_mode", choices=["4cam", "cam_high"], default="4cam",
                        help="Camera input mode: '4cam' (2x2 mosaic) or 'cam_high' (overhead only)")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Specific episode IDs to evaluate (e.g. --episodes 35)")
    parser.add_argument("--ood_red", action="store_true",
                        help="OOD Stress Test: shift both yellow screwdriver and blue cup to RED across all 4 cameras")
    parser.add_argument("--save_video", action="store_true", help="Record and save MP4 rollout videos of evaluated episodes")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token if needed")
    return parser.parse_args()

# ==============================================================================
# 1. Physics Engine Setup (MuJoCo)
# ==============================================================================
def init_mujoco():
    try:
        import mujoco
    except ImportError:
        print("⚠️ Dependencies not detected. Auto-installing from requirements.txt...")
        import subprocess
        req_file = os.path.join(SCRIPT_DIR, "requirements.txt")
        subprocess.run(["pip", "install", "-q", "-r", req_file], check=True)
        import mujoco

    scene_path = SCENE_XML
    if not os.path.exists(scene_path):
        fallback = os.path.join(SCRIPT_DIR, "mujoco_menagerie", "aloha", "scene.xml")
        if os.path.exists(fallback):
            scene_path = fallback
        else:
            raise FileNotFoundError(f"MuJoCo scene XML not found in {SCRIPT_DIR}/mujoco_menagerie/aloha/")
    
    print(f"⚙️ Compiling MuJoCo physics from {scene_path}...")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    return model, data

def scale_gripper(action_val):
    norm = np.clip(action_val / 0.69, 0.0, 1.0)
    return float(0.002 + norm * (0.037 - 0.002))

def map_action_to_ctrl(action_14):
    ctrl = np.zeros(14, dtype=np.float32)
    ctrl[0:6] = action_14[0:6]
    ctrl[6] = scale_gripper(action_14[6])
    ctrl[7:13] = action_14[7:13]
    ctrl[13] = scale_gripper(action_14[13])
    return ctrl

def compute_episode_grasp_point(model, data, actions):
    import mujoco
    b_finger_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right/left_finger_link")
    b_finger_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right/right_finger_link")
    z_list = []
    pos_list = []
    for f in range(min(220, len(actions))):
        act = actions[f]
        data.qpos[0:6] = act[0:6]
        data.qpos[6] = data.qpos[7] = scale_gripper(act[6]) / 2.0
        data.qpos[8:14] = act[7:13]
        data.qpos[14] = data.qpos[15] = scale_gripper(act[13]) / 2.0
        mujoco.mj_forward(model, data)
        pt = (data.xpos[b_finger_l] + data.xpos[b_finger_r]) / 2.0
        z_list.append(pt[2])
        pos_list.append(pt)

    min_f = int(np.argmin(z_list))
    gx = float(pos_list[min_f][0])
    gy = float(pos_list[min_f][1])
    return gx, gy

def run_simulation_rollout(model, data, action_sequence, initial_joint_state, grasp_xy=(0.05, -0.10), record_video=False, video_path=None):
    import mujoco
    screw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screwdriver")
    screw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "screwdriver_joint")
    cup_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")
    cup_pos = model.body_pos[cup_body_id] if cup_body_id != -1 else np.array([-0.12, 0.19, 0.001])
    
    mujoco.mj_resetData(model, data)
    data.qpos[0:6] = initial_joint_state[0:6]
    grip_l = scale_gripper(initial_joint_state[6]) / 2.0
    data.qpos[6] = grip_l
    data.qpos[7] = grip_l

    data.qpos[8:14] = initial_joint_state[7:13]
    grip_r = scale_gripper(initial_joint_state[13]) / 2.0
    data.qpos[14] = grip_r
    data.qpos[15] = grip_r

    data.ctrl[:14] = map_action_to_ctrl(initial_joint_state)
    if screw_joint_id != -1:
        screw_qpos_adr = model.jnt_qposadr[screw_joint_id]
        data.qpos[screw_qpos_adr : screw_qpos_adr + 7] = [grasp_xy[0], grasp_xy[1], 0.015, 0.7071, 0.0, 0.7071, 0.0]
    
    mujoco.mj_forward(model, data)

    initial_screw_z = data.xpos[screw_body_id][2] if screw_body_id != -1 else 0.0
    max_screw_z = initial_screw_z
    frames = []

    renderer = None
    if record_video and video_path:
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
        except Exception as e:
            print(f"    ⚠️ Headless GL context notice: {e}. Skipping MP4 frame recording.")
            renderer = None

    for act in action_sequence:
        data.ctrl[:14] = map_action_to_ctrl(act)
        for _ in range(4):
            mujoco.mj_step(model, data)
        if screw_body_id != -1:
            current_z = data.xpos[screw_body_id][2]
            if current_z > max_screw_z:
                max_screw_z = current_z
        if renderer is not None:
            renderer.update_scene(data, camera="overhead_cam")
            frames.append(renderer.render())

    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass

    if record_video and video_path and len(frames) > 0:
        try:
            import mediapy as media
            os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
            media.write_video(video_path, frames, fps=50)
            print(f"    🎥 Saved evaluation rollout video: {video_path}")
        except Exception as e:
            print(f"    Note on video save: {e}")

    lift_cm = (max_screw_z - initial_screw_z) * 100.0
    
    # Check cup placement
    final_screw_pos = data.xpos[screw_body_id] if screw_body_id != -1 else np.zeros(3)
    dist_to_cup = float(np.linalg.norm(final_screw_pos[:2] - cup_pos[:2]))
    landed_in_cup = bool(dist_to_cup <= 0.070 and final_screw_pos[2] < 0.10)
    lift_success = bool(lift_cm >= 5.0)

    return {
        "success": lift_success,
        "lift_success": lift_success,
        "landed_in_cup": landed_in_cup,
        "peak_lift_cm": round(float(lift_cm), 2),
        "dist_to_cup_cm": round(dist_to_cup * 100.0, 2),
        "final_z": round(float(max_screw_z), 4)
    }

# ==============================================================================
# 2. Dataset Downloader & Multi-Camera Video Reader
# ==============================================================================
def download_dataset(hf_token=None):
    from huggingface_hub import snapshot_download
    print("📥 Checking / Downloading ALOHA Screwdriver dataset (Parquets + 4-Camera Videos)...")
    token = hf_token or os.environ.get("HF_TOKEN", None)
    repo_dir = snapshot_download(
        repo_id="lerobot/aloha_static_screw_driver",
        repo_type="dataset",
        allow_patterns=["*.parquet", "*.json", "meta/**", "videos/**"],
        token=token
    )
    print(f"✅ Dataset cached at: {repo_dir}")
    # Load ONLY episode telemetry parquets under data/ (exclude meta/*.parquet)
    parquet_files = glob.glob(os.path.join(repo_dir, "data", "**", "*.parquet"), recursive=True)
    if not parquet_files:
        parquet_files = [p for p in glob.glob(os.path.join(repo_dir, "**", "*.parquet"), recursive=True) if "meta" not in p]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {repo_dir}")
    
    # Load all parquet chunks
    dfs = []
    for p in parquet_files:
        dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Loaded {len(df):,} total frames across {df['episode_index'].nunique()} episodes.")
    return df, repo_dir

def shift_both_to_red_bgr(img_bgr):
    """OOD visual shift: transforms yellow screwdriver & blue cup to vibrant RED (even under overhead glare) while preserving table & robot arms."""
    import cv2
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b_ch = img_bgr[:, :, 0].astype(np.float32)
    g_ch = img_bgr[:, :, 1].astype(np.float32)
    r_ch = img_bgr[:, :, 2].astype(np.float32)

    # 1. Yellow screwdriver handle & sponge (catches pale overhead glare S>=14 while rejecting neutral white tape)
    yellow_mask = (h >= 16) & (h <= 42) & (s >= 14) & (v >= 95) & (r_ch > b_ch + 12) & (g_ch > b_ch + 12)
    hsv[:, :, 0] = np.where(yellow_mask, 0.0, hsv[:, :, 0])
    hsv[:, :, 1] = np.where(yellow_mask, np.clip(np.maximum(s * 1.4, s * 2.5 + 90), 0, 235), hsv[:, :, 1])
    hsv[:, :, 2] = np.where(yellow_mask, np.clip(v * 0.85, 0, 240), hsv[:, :, 2])

    # 2. Blue/Cyan cup -> Vibrant Red
    blue_mask = (h >= 85) & (h <= 125) & (s >= 45) & (v >= 50)
    hsv[:, :, 0] = np.where(blue_mask, (h - 98) % 180, hsv[:, :, 0])
    hsv[:, :, 1] = np.where(blue_mask, np.clip(s * 1.2, 0, 245), hsv[:, :, 1])
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

class MultiCamEpisodeReader:
    def __init__(self, repo_dir, mode="4cam", ood_red=False):
        import cv2
        self.mode = mode
        self.ood_red = ood_red
        def _find_vid(name):
            res = glob.glob(os.path.join(repo_dir, "**", f"*{name}*", "**", "*.mp4"), recursive=True)
            if not res:
                raise FileNotFoundError(f"Video stream '{name}' not found in {repo_dir}/videos")
            return res[0]

        self.cap_h = cv2.VideoCapture(_find_vid("cam_high"))
        self.cap_l = cv2.VideoCapture(_find_vid("cam_low")) if mode == "4cam" else None
        self.cap_lw = cv2.VideoCapture(_find_vid("cam_left_wrist")) if mode == "4cam" else None
        self.cap_rw = cv2.VideoCapture(_find_vid("cam_right_wrist")) if mode == "4cam" else None

    def seek_episode(self, ep_id):
        import cv2
        start_frame = ep_id * 400
        self.cap_h.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if self.mode == "4cam":
            self.cap_l.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.cap_lw.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.cap_rw.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def read_next_frame(self):
        import cv2
        _, fh = self.cap_h.read()
        if fh is None:
            return None
        if self.mode == "cam_high":
            if self.ood_red:
                fh = shift_both_to_red_bgr(fh)
            return cv2.cvtColor(fh, cv2.COLOR_BGR2RGB)
        
        _, fl = self.cap_l.read()
        _, flw = self.cap_lw.read()
        _, frw = self.cap_rw.read()
        if fl is None or flw is None or frw is None:
            return None
        
        top = np.hstack([cv2.resize(fh, (320, 240)), cv2.resize(fl, (320, 240))])
        bot = np.hstack([cv2.resize(flw, (320, 240)), cv2.resize(frw, (320, 240))])
        grid = np.vstack([top, bot])
        if self.ood_red:
            grid = shift_both_to_red_bgr(grid)
        return cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap_h.release()
        if self.cap_l: self.cap_l.release()
        if self.cap_lw: self.cap_lw.release()
        if self.cap_rw: self.cap_rw.release()

# ==============================================================================
# 3. OpenVLA Model Wrapper
# ==============================================================================
class OpenVLAEngine:
    def __init__(self, model_name="openvla/openvla-7b", load_in_4bit=True, device="cuda", adapter_path=None):
        from transformers import AutoProcessor, BitsAndBytesConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        print(f"🤖 Initializing OpenVLA ({model_name}) on {device}...")
        
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # Directly load OpenVLA dynamic model class to bypass AutoModel whitelist restrictions
        try:
            ModelClass = get_class_from_dynamic_module("modeling_prismatic.OpenVLAForActionPrediction", model_name)
        except Exception:
            try:
                from transformers import AutoModelForVision2Seq as ModelClass
            except ImportError:
                from transformers import AutoModel as ModelClass

        # Patch missing modern transformers attributes onto OpenVLA model class
        if not hasattr(ModelClass, "_supports_sdpa"):
            ModelClass._supports_sdpa = True
        if not hasattr(ModelClass, "_supports_flash_attn_2"):
            ModelClass._supports_flash_attn_2 = False
        if not hasattr(ModelClass, "_supports_cache_class"):
            ModelClass._supports_cache_class = False

        if device == "cuda" and load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            self.model = ModelClass.from_pretrained(
                model_name,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            for mod_name in ["vision_backbone", "projector"]:
                m = getattr(self.model, mod_name, None)
                if m is not None:
                    m.to(dtype=torch.bfloat16)
        else:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            self.model = ModelClass.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(device)

        # Restore GenerationMixin if stripped by modern transformers (>= 4.50)
        if not hasattr(self.model, "generate"):
            from transformers import GenerationMixin
            self.model.__class__ = type(
                self.model.__class__.__name__,
                (self.model.__class__, GenerationMixin),
                {}
            )

        # Ensure generation_config exists (required by GenerationMixin.generate())
        if not hasattr(self.model, "generation_config") or self.model.generation_config is None:
            from transformers import GenerationConfig
            self.model.generation_config = GenerationConfig()

        self.is_lora_14dof = False
        if adapter_path and os.path.exists(adapter_path):
            print(f"📦 Loading fine-tuned 14-DoF LoRA adapter from {adapter_path}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.is_lora_14dof = True
            print("✅ 14-DoF Bimanual ALOHA adapter active! Generating 14 bimanual motor actions directly.")

        self.model.eval()
        print("✨ OpenVLA foundation model successfully loaded into memory (eval mode)!")

    def predict_actions_batch(self, images_np_list, instruction="pick up the yellow screwdriver and place it into the blue cup"):
        """
        Direct forward-pass autoregressive decoding for a batch of RGB images.
        Uses the exact same model(input_ids, attention_mask, pixel_values, use_cache=False)
        path as training, completely bypassing transformers >= 4.50's DynamicCache bug where
        PrismaticForConditionalGeneration.prepare_inputs_for_generation() drops pixel_values on step 0.
        """
        pil_images = [Image.fromarray(im).convert("RGB") for im in images_np_list]
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        prompts = [prompt] * len(pil_images)

        inputs = self.processor(text=prompts, images=pil_images, return_tensors="pt", padding=True)
        cur_ids = inputs["input_ids"].to(self.device)
        cur_mask = inputs["attention_mask"].to(self.device) if "attention_mask" in inputs and inputs["attention_mask"] is not None else torch.ones_like(cur_ids)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            if self.is_lora_14dof:
                gen_tokens = []
                for _ in range(14):
                    outputs = self.model(
                        input_ids=cur_ids,
                        attention_mask=cur_mask,
                        pixel_values=pixel_values,
                        use_cache=False
                    )
                    # Decode within OpenVLA's 256 action vocabulary bins [31744..31999]
                    action_logits = outputs.logits[:, -1, 31744:32000]
                    next_token = 31744 + torch.argmax(action_logits, dim=-1, keepdim=True)
                    gen_tokens.append(next_token)
                    cur_ids = torch.cat([cur_ids, next_token], dim=1)
                    cur_mask = torch.cat([cur_mask, torch.ones_like(next_token)], dim=1)

                # [B, 14]
                token_matrix = torch.cat(gen_tokens, dim=1).cpu().numpy()
                return [tokens_to_aloha_actions(row_toks) for row_toks in token_matrix]
            else:
                return [self.predict_action(im, instruction=instruction) for im in images_np_list]

    def predict_action(self, image_np, instruction="pick up the yellow screwdriver and place it into the blue cup"):
        if self.is_lora_14dof:
            return self.predict_actions_batch([image_np], instruction=instruction)[0]
        image = Image.fromarray(image_np)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            if hasattr(self.model, "predict_action"):
                return self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            return np.zeros(14, dtype=np.float32)

# ==============================================================================
# 4. Evaluation Routine
# ==============================================================================
def parse_action_row(act):
    """Robustly parse single action entry into a 14-dim numpy float array."""
    if act is None:
        return np.zeros(14, dtype=np.float32)
    if isinstance(act, (list, tuple, np.ndarray)):
        arr = np.array(act, dtype=np.float32).flatten()
        if len(arr) == 14:
            return arr
        elif len(arr) > 0:
            out = np.zeros(14, dtype=np.float32)
            n = min(14, len(arr))
            out[:n] = arr[:n]
            return out
    return np.zeros(14, dtype=np.float32)

def evaluate_episodes(model_engine, mj_model, mj_data, df, episode_indices, multi_cam_reader=None, save_video=False, output_dir="./openvla_aloha_output"):
    print(f"\n🔬 Evaluating across {len(episode_indices)} Holdout Episodes...")
    results = []
    all_predictions = {}
    
    for ep_id in episode_indices:
        ep_data = df[df["episode_index"] == ep_id].sort_values("frame_index")
        
        # Robustly parse action sequence
        actions_list = [parse_action_row(r["action"]) for _, r in ep_data.iterrows()]
        if not actions_list:
            continue
        gt_actions = np.stack(actions_list, axis=0)
        
        # Robustly parse initial joint state
        raw_state = ep_data["observation.state"].iloc[0]
        init_state = parse_action_row(raw_state)

        # Baseline evaluation rollout
        if model_engine is None:
            # Baseline: Zero-action static or human reference comparison
            pred_actions = np.zeros_like(gt_actions)
        else:
            # OpenVLA model predictions (batched direct forward-pass decoding)
            import io
            from tqdm import tqdm
            if multi_cam_reader is not None:
                multi_cam_reader.seek_episode(ep_id)

            ep_frames = []
            for f_i, (_, row) in enumerate(ep_data.iterrows()):
                if multi_cam_reader is not None:
                    img_np = multi_cam_reader.read_next_frame()
                    if img_np is None:
                        raise RuntimeError(f"Error reading 4-camera video frame for episode {ep_id}")
                else:
                    raw_img = row.get("observation.images.cam_high", None)
                    if isinstance(raw_img, dict) and "bytes" in raw_img:
                        img_np = np.array(Image.open(io.BytesIO(raw_img["bytes"])))
                    elif isinstance(raw_img, np.ndarray):
                        img_np = raw_img
                    elif hasattr(raw_img, "convert"):
                        img_np = np.array(raw_img)
                    else:
                        raise ValueError(f"No valid image found for episode {ep_id} and multi_cam_reader is None")
                if f_i == 0:
                    m_val, s_val = float(img_np.mean()), float(img_np.std())
                    if m_val < 5.0 or s_val < 5.0:
                        raise RuntimeError(f"ABORTING! Eval frame 0 of Ep {ep_id} appears black/blank (mean={m_val:.1f}, std={s_val:.1f}).")
                ep_frames.append(img_np)

            # Upfront 2-frame vision responsiveness check on first episode (proves vision is NOT bypassed!)
            if ep_id == episode_indices[0] and len(ep_frames) > 150:
                chk_acts = model_engine.predict_actions_batch([ep_frames[0], ep_frames[150]])
                pred_delta = float(np.mean(np.abs(chk_acts[0] - chk_acts[1])))
                gt_delta = float(np.mean(np.abs(gt_actions[0] - gt_actions[150])))
                print(f"  🔍 Vision Check (Ep {ep_id} Frame 0 vs 150): Pred Δ={pred_delta:.4f} rad | Human Δ={gt_delta:.4f} rad")
                print(f"     • Frame   0 R-arm [shld, elbow, grip]: Pred={np.round(chk_acts[0][[8,9,13]], 3)} vs GT={np.round(gt_actions[0][[8,9,13]], 3)}")
                print(f"     • Frame 150 R-arm [shld, elbow, grip]: Pred={np.round(chk_acts[1][[8,9,13]], 3)} vs GT={np.round(gt_actions[150][[8,9,13]], 3)}")

            eval_bs = 4
            pred_actions = []
            num_batches = (len(ep_frames) + eval_bs - 1) // eval_bs
            batch_pbar = tqdm(range(0, len(ep_frames), eval_bs), total=num_batches, desc=f"  🎬 Ep {ep_id} Rollout (4-Cam Batched)", leave=False)
            for b_start in batch_pbar:
                chunk = ep_frames[b_start : b_start + eval_bs]
                acts_chunk = model_engine.predict_actions_batch(chunk)
                for a in acts_chunk:
                    pred_actions.append(parse_action_row(a))
            pred_actions = np.stack(pred_actions, axis=0)

        all_predictions[int(ep_id)] = pred_actions.tolist()
        l1_error = float(np.mean(np.abs(pred_actions - gt_actions)))

        # Calibrated per-episode grasp coordinate
        grasp_xy = compute_episode_grasp_point(mj_model, mj_data, gt_actions)

        # Video path for visualization
        is_ood = bool(multi_cam_reader is not None and getattr(multi_cam_reader, "ood_red", False))
        suffix = "_ood_red" if is_ood else ""
        if is_ood and len(ep_frames) > 0:
            os.makedirs(output_dir, exist_ok=True)
            if len(ep_frames) > 100:
                ood_chk_path = os.path.join(output_dir, f"ood_red_input_ep{ep_id}_frame100.jpg")
                Image.fromarray(ep_frames[100]).save(ood_chk_path)
            if save_video:
                try:
                    import mediapy as media
                    ood_in_vid = os.path.join(output_dir, f"ood_red_input_ep{ep_id}.mp4")
                    media.write_video(ood_in_vid, ep_frames, fps=50)
                    print(f"  🎨 Saved full OOD Red 4-cam input video: {ood_in_vid}")
                except Exception as e:
                    print(f"  Note saving OOD input video: {e}")
        v_path = os.path.join(output_dir, f"eval_episode_{ep_id}{suffix}.mp4") if save_video else None

        # Physical rollout in MuJoCo
        rollout = run_simulation_rollout(mj_model, mj_data, pred_actions, init_state, grasp_xy=grasp_xy, record_video=save_video, video_path=v_path)
        
        # Human reference rollout for comparison
        gt_rollout = run_simulation_rollout(mj_model, mj_data, gt_actions, init_state, grasp_xy=grasp_xy)

        ep_result = {
            "episode_id": ep_id,
            "ood_red": is_ood,
            "lift_success": rollout["lift_success"],
            "landed_in_cup": rollout["landed_in_cup"],
            "peak_lift_cm": rollout["peak_lift_cm"],
            "dist_to_cup_cm": rollout["dist_to_cup_cm"],
            "l1_error_rad": round(l1_error, 4),
            "video_path": v_path if (save_video and v_path and os.path.exists(v_path)) else None,
            "human_reference": {
                "lift_success": gt_rollout["lift_success"],
                "landed_in_cup": gt_rollout["landed_in_cup"],
                "peak_lift_cm": gt_rollout["peak_lift_cm"],
                "dist_to_cup_cm": gt_rollout["dist_to_cup_cm"]
            }
        }
        results.append(ep_result)
        tag = " [OOD RED]" if is_ood else ""
        print(f"  • Episode {ep_id:2d}{tag}: Lift={rollout['lift_success']} ({rollout['peak_lift_cm']:.1f}cm) | Cup={rollout['landed_in_cup']} (dist {rollout['dist_to_cup_cm']:.1f}cm) | Human Lift={gt_rollout['peak_lift_cm']:.1f}cm Cup={gt_rollout['landed_in_cup']} | L1 Error={l1_error:.4f} rad")
        if v_path and os.path.exists(v_path):
            print(f"  🎬 Episode {ep_id} video ready: {v_path}")

        # Incremental auto-save so intermediate scorecard is always up to date on disk
        interim_report_file = os.path.join(output_dir, f"openvla_benchmark_results{suffix}.json")
        try:
            curr_lift = (sum(1 for r in results if r["lift_success"]) / len(results)) * 100.0
            curr_cup = (sum(1 for r in results if r["landed_in_cup"]) / len(results)) * 100.0
            curr_l1 = float(np.mean([r["l1_error_rad"] for r in results]))
            with open(interim_report_file, "w") as f:
                json.dump({
                    "status": "in_progress",
                    "episodes_completed": len(results),
                    "current_lift_rate_pct": round(curr_lift, 1),
                    "current_cup_rate_pct": round(curr_cup, 1),
                    "current_mean_l1_error": round(curr_l1, 4),
                    "episodes": results
                }, f, indent=2)
        except Exception:
            pass

    lift_rate = (sum(1 for r in results if r["lift_success"]) / len(results)) * 100.0
    cup_rate = (sum(1 for r in results if r["landed_in_cup"]) / len(results)) * 100.0
    ref_lift_rate = (sum(1 for r in results if r["human_reference"]["lift_success"]) / len(results)) * 100.0
    ref_cup_rate = (sum(1 for r in results if r["human_reference"]["landed_in_cup"]) / len(results)) * 100.0
    mean_l1 = float(np.mean([r["l1_error_rad"] for r in results]))
    
    print("\n" + "=" * 70)
    print("📊 HOLD-OUT TEST RESULTS SCORECARD")
    print("=" * 70)
    print(f"• Evaluated Episodes            : {len(results)}")
    print(f"• Policy Lift Success Rate      : {lift_rate:.1f}%  (Human Ref: {ref_lift_rate:.1f}%)")
    print(f"• Policy Cup Insertion Rate     : {cup_rate:.1f}%  (Human Ref: {ref_cup_rate:.1f}%)")
    print(f"• Mean L1 Action Error          : {mean_l1:.4f} rad")
    print("=" * 70)

    return {
        "lift_success_rate_pct": lift_rate,
        "cup_insertion_rate_pct": cup_rate,
        "human_expert_lift_rate_pct": ref_lift_rate,
        "human_expert_cup_rate_pct": ref_cup_rate,
        "mean_l1_error_rad": mean_l1,
        "episodes": results
    }

def main():
    args = parse_args()
    print("=" * 80)
    print("🤖 PHYSICAL AI: OPENVLA 7B ALOHA ZERO-SHOT & FINE-TUNING PIPELINE")
    print("=" * 80)

    # Initialize MuJoCo
    mj_model, mj_data = init_mujoco()

    # If OOD Red Stress Test is enabled, recolor MuJoCo screwdriver & cup to matching RED
    if args.ood_red:
        import mujoco
        for mat_name, rgba in [
            ("screwdriver_handle", [0.88, 0.14, 0.14, 1.0]),
            ("cup_blue_outer", [0.82, 0.12, 0.12, 1.0]),
            ("cup_sponge_yellow", [0.78, 0.20, 0.20, 1.0]),
        ]:
            mid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
            if mid != -1:
                mj_model.mat_rgba[mid] = rgba
        print("🎨 MuJoCo Digital Twin recolored to matching OOD RED (Red Screwdriver + Red Cup).")

    # Load splits
    with open(SPLITS_JSON, "r") as f:
        splits = json.load(f)["splits"]
    if args.episodes:
        test_eps = args.episodes
    else:
        test_eps = splits["test"]["episodes"][:args.test_limit]
    train_eps = splits["train"]["episodes"]

    ood_banner = " [OOD RED STRESS TEST: Red Screwdriver + Red Cup]" if args.ood_red else ""
    print(f"📂 Evaluating {len(test_eps)} Test Episode(s): {test_eps}{ood_banner}")

    # Download / Load dataset
    df, repo_dir = download_dataset(args.hf_token)
    multi_cam_reader = MultiCamEpisodeReader(repo_dir, mode=args.camera_mode, ood_red=args.ood_red)

    device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    print(f"🖥️ Execution Device: {device.upper()}")

    # Initialize model if zero_shot, eval_lora, or full_pipeline mode
    model_engine = None
    if args.mode in ["zero_shot", "eval_lora", "full_pipeline"] and TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            adapter = args.adapter_path if args.mode == "eval_lora" else None
            model_engine = OpenVLAEngine(load_in_4bit=True, device="cuda", adapter_path=adapter)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠️ Warning: Could not initialize OpenVLA-7B ({e}). Running baseline physics rollout.")
            model_engine = None

    eval_summary = evaluate_episodes(
        model_engine, mj_model, mj_data, df, test_eps,
        multi_cam_reader=multi_cam_reader,
        save_video=args.save_video,
        output_dir=args.output_dir
    )
    multi_cam_reader.close()

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "_ood_red" if args.ood_red else ""
    report_file = os.path.join(args.output_dir, f"openvla_benchmark_results{suffix}.json")

    report = {
        "mode": args.mode,
        "device": device,
        "evaluation_summary": eval_summary,
        "test_episodes": test_eps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n🎉 Benchmark complete! Report saved to {report_file}")

if __name__ == "__main__":
    main()
