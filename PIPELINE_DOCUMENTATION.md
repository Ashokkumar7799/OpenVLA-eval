# End-to-End Physical AI Fine-Tuning & Evaluation Pipeline
## 14-DoF Bimanual ALOHA Robot with OpenVLA-7B Foundation Model

---

## 1. System Overview & Problem Formulation

### 1.1 The Physical Task
The robot must autonomously execute a multi-stage, high-precision bimanual manipulation sequence using the `lerobot/aloha_static_screw_driver` dataset (50 total demonstrations, 20,051 frames recorded at 50 Hz):
1. **Right Arm Grasp**: Reach down, align parallel jaw grippers, and firmly grasp a yellow screwdriver resting flat on the table.
2. **Right Arm Lift**: Lift the screwdriver vertically off the table ($\ge 5\text{ cm}$).
3. **Mid-Air Bimanual Handover**: Coordinate both arms in mid-air to transfer the screwdriver shaft from the right gripper to the left gripper.
4. **Left Arm Cup Placement**: Move the screwdriver over a tapered blue container and insert it inside.

### 1.2 The Core Research Comparison
- **Model A (Real-Only Baseline)**: OpenVLA-7B fine-tuned strictly on the 30 real physical demonstrations (Episodes 0–29, 12,000 frames).
- **Model B (Real + AI Video Augmented)**: OpenVLA-7B fine-tuned on the 30 real physical demonstrations plus synthetic visual variations generated via the generative video pipeline in `E:\mediapipe\AI` (varying table textures, ambient lighting, shadows, distractor objects, and cup appearance).
- **Evaluation Benchmark**: Both models are evaluated in closed-loop simulation on the **15 unseen holdout test episodes (Episodes 35–49)** inside a calibrated physics digital twin.

---

## 2. MuJoCo Digital Twin Physics Calibration

To evaluate physical manipulation accurately without physical hardware risk, a 1-to-1 digital twin of the real ALOHA workcell was built and calibrated in MuJoCo (`mujoco_menagerie/aloha/scene_with_screwdriver.xml`). Because the LeRobot dataset only logs robot joint telemetry and RGB video—not 3D object ground-truth poses or simulator actuator units—a **5-stage calibration process** is required to align the real robot with MuJoCo physics:

### 2.1 Stage 1: Joint & Gripper Actuator Unit Calibration (`scale_gripper`)
* **Challenge**: The 6 revolute arm joints per arm map directly in radians (`action[0:6]` $\to$ Left Arm, `action[7:13]` $\to$ Right Arm). However, the real ALOHA leader-follower gripper telemetry spans roughly $[0.0, 0.69]$, whereas MuJoCo Menagerie's `aloha.xml` expects linear parallel-jaw slide positions in meters spanning $[0.002\text{ m}, 0.037\text{ m}]$.
* **Calibration Equation**:
  $$\text{norm} = \text{clip}\left(\frac{a_{\text{grip}}}{0.69},\; 0.0,\; 1.0\right)$$
  $$\text{ctrl}_{\text{grip}} = 0.002 + \text{norm} \times (0.037 - 0.002)$$
* **Symmetrical Finger Initialization**: On episode reset (`mj_resetData`), each parallel gripper consists of two coupled slide joints (`qpos[6], qpos[7]` for Left; `qpos[14], qpos[15]` for Right). Each finger is initialized to half the total opening: `qpos_finger = scale_gripper(a_grip) / 2.0`.

### 2.2 Stage 2: Per-Episode Dynamic Screwdriver Placement via Forward Kinematics (`compute_episode_grasp_point`)
* **Challenge**: Human operators place the yellow screwdriver at different tabletop coordinates across an $\sim 18\text{ cm}$ span from episode to episode, and no 3D object pose labels exist in the parquet files.
* **Calibration Algorithm**:
  1. **Kinematic Replay (No Physics Stepping)**: Loop over the first 220 frames of the episode's ground-truth human actions, assign joint positions directly to `data.qpos`, and call `mujoco.mj_forward(model, data)`.
  2. **Gripper Midpoint Tracking**: At every frame $f \in [0, 220)$, compute the 3D Cartesian midpoint between the right arm's two finger links:
     $$\mathbf{p}_f = \frac{\mathbf{x}_{\text{right/left\_finger\_link}}(f) + \mathbf{x}_{\text{right/right\_finger\_link}}(f)}{2}$$
  3. **Minimum-$Z$ Grasp Detection**: Identify the frame $f^* = \arg\min_f (\mathbf{p}_f)_z$ where the right gripper reaches its lowest vertical height above the table (the exact moment of table-level tool contact).
  4. **Object Spawning**: Spawn the screwdriver's 6-DoF `freejoint` horizontally flat on the tabletop (`quat = [0.7071, 0, 0.7071, 0]`, i.e., $90^\circ$ pitch around $Y$) at $[x = (\mathbf{p}_{f^*})_x,\; y = (\mathbf{p}_{f^*})_y,\; z = 0.015\text{ m}]$.

### 2.3 Stage 3: Hollow Cup Geometric & Collision Calibration (`generate_cup.py`)
* **Challenge**: Standard MuJoCo primitive cylinders are strictly convex solids; any object dropped onto a cylinder bounces off the top face instead of entering the container.
* **Calibration Process**:
  1. **Visual OBJ Meshes (`contype="0" conaffinity="0"`)**: Generated 64-sector tapered meshes (`cup_outer.obj`, `cup_inner.obj`, `cup_rim.obj`) visually matched to the real overhead video frames:
     - **Height**: $4.5\text{ cm}$
     - **Top Diameter**: $14.0\text{ cm}$ outer, $13.2\text{ cm}$ inner
     - **Bottom Diameter**: $7.0\text{ cm}$ outer, $6.2\text{ cm}$ inner
  2. **Hollow Physics Wall Proxies (`contype="1" conaffinity="1"`)**: Constructed **24 invisible tangential box geoms** (`cup_col_0`..`cup_col_23`, placed every $15^\circ$ at radius $5.25\text{ cm}$) to form a true hollow rim, plus a solid base plate (`cup_base_collision`) and a high-friction interior yellow sponge (`friction="1.5 0.1 0.002"`).
  3. **Drop-Zone Coordinate Alignment**: Calibrated via forward kinematics of the left arm's release trajectories across episodes to fixed tabletop coordinates $[X = -0.12\text{ m},\; Y = 0.19\text{ m},\; Z = 0.001\text{ m}]$.

### 2.4 Stage 4: Contact Dynamics & Control Frequency Calibration
* **Torsional Grasp Friction (`condim="4"`)**: Configured the screwdriver handle cylinder (`mass="0.04 kg"`, radius `1.3 cm`) with 4-dimensional contact friction (`friction="2.0 0.1 0.002"` for sliding, torsional, and rolling resistance) so parallel-jaw pinch grasps lock reliably during mid-air lift and handover without numerical slip.
* **50 Hz Sub-Stepping Synchronization**: Real demonstrations run at **50 Hz** ($\Delta t = 20\text{ ms/frame}$). Each action command executes **4 MuJoCo physics sub-steps** (`for _ in range(4): mujoco.mj_step(model, data)`) to maintain numerical solver stability while matching real-time kinematics.

### 2.5 Stage 5: Human Expert Upper-Bound Verification
Before evaluating any neural policy, all 50 ground-truth human trajectories are replayed through the calibrated Digital Twin to verify simulation fidelity:
* **Grasp & Lift Success ($\ge 5\text{ cm}$)**: **88% (44 / 50 episodes)**
* **Full Hollow-Cup Placement**: **70% (35 / 50 episodes)**
* This establishes the physics-verified **Human Expert Reference** reported alongside every Model A and Model B evaluation run.

---

## 3. Foundation Model Action Representation

OpenVLA-7B is an autoregressive vision-language-action model built on LLaMA-2 with dual vision encoders (DINOv2 + SigLIP).

### 3.1 14-DoF Joint Ordering
The ALOHA action vector is a 14-dimensional continuous array:
$$\mathbf{a} = [\underbrace{\theta_0, \theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6}_{\text{Left Arm: Waist, Shoulder, Elbow, Forearm Roll, Wrist Pitch, Wrist Roll, Gripper}}, \underbrace{\theta_7, \theta_8, \theta_9, \theta_{10}, \theta_{11}, \theta_{12}, \theta_{13}}_{\text{Right Arm: Waist, Shoulder, Elbow, Forearm Roll, Wrist Pitch, Wrist Roll, Gripper}}]$$

### 3.2 Continuous-to-Discrete Discretization
Each continuous actuator command $a_i$ is mapped to a discrete bin index $b_i \in [0, 255]$ using empirical min/max bounds from `meta/stats.json`:
$$\text{norm}_i = \text{clip}\left(\frac{a_i - \text{min}_i}{\text{max}_i - \text{min}_i}, 0.0, 1.0\right)$$
$$b_i = \min(\lfloor \text{norm}_i \times 256 \rfloor, 255)$$

### 3.3 Vocabulary Injection
OpenVLA reserves the final 256 token IDs in its 32,000 LLaMA vocabulary (`31744` to `31999`) for actions:
$$\text{Token ID}_i = 31744 + b_i$$

### 3.4 4-Camera Synchronized Multi-View Vision Architecture
Rather than relying on a single overhead view, the vision encoder receives a **synchronized $2 \times 2$ multi-camera mosaic** ($640 \times 480 \times 3$) at every timestep $t$:

```
+------------------------------------+------------------------------------+
| Top-Left: cam_high (320x240)       | Top-Right: cam_low (320x240)       |
| • Global tabletop macro view       | • Front depth & clearance view     |
| • Tracks both arms and cup target  | • Table plane angle                |
+------------------------------------+------------------------------------+
| Bottom-Left: cam_left_wrist (320)  | Bottom-Right: cam_right_wrist (320)|
| • Left gripper close-up            | • Right gripper grasp precision    |
| • Cup stabilization alignment      | • Screwdriver handle contact view  |
+------------------------------------+------------------------------------+
```

* **Zero Latency Penalty**: OpenVLA's fused DinoV2 + SigLIP encoder natively processes the single $(640 \times 480)$ composite image in standard $1.36\text{ s/it}$ forward passes with no architecture or projector modifications.
* **Dual-Scale Perception**: Combines macro-scale trajectory navigation with micro-scale millimeter grasp alignment.

---

## 4. QLoRA Fine-Tuning Pipeline (`train_openvla_aloha_lora.py`)

### 4.1 Memory & Parameter Footprint
Fine-tuning a 7.5B parameter model on an NVIDIA L4 (24 GB VRAM) requires Quantized Low-Rank Adaptation (QLoRA):
- **Base Model**: Loaded in 4-bit NormalFloat (NF4) with `bfloat16` compute dtype ($\sim 4.5\text{ GB}$ base footprint).
- **LoRA Configuration**:
  - Target Modules: `q_proj`, `v_proj`, `k_proj`, `o_proj` (self-attention projections).
  - Rank ($r$): $32$
  - Scaling ($\alpha$): $64$
  - Dropout: $0.05$
  - **Trainable Parameters**: $33,554,432$ ($0.44\%$ of model). All other $7.54\text{B}$ parameters are completely frozen.
  - **Peak VRAM**: $\sim 9.2\text{ GB}$ (comfortably fits in 24 GB).

### 4.2 High-Speed NVMe Image Cache (`./dataset_cache_4cam`)
In LeRobot 2.0 format, raw telemetry resides in `.parquet` files while video frames are encoded in compressed H.264 `.mp4` containers. To prevent random-seek decoding latency during training:
1. `prepare_image_cache()` extracts and tiles all 12,000 frames (30 episodes $\times$ 400 frames) sequentially at **229 FPS** into `./dataset_cache_4cam/frame_XXXXXX.jpg` in $\sim 1\text{--}2\text{ minutes}$ once.
2. The PyTorch `AlohaDataset` loads pre-cached JPEGs in **$< 0.5\text{ ms}$**, eliminating all video decoding bottlenecks.
3. **Hard Validation**: Strict assertion guards prevent silent black-frame fallbacks. If any image is missing, execution immediately halts.

### 4.3 Sequence Composition & Attention Masking
Each training step constructs the following multimodal sequence:
$$\mathbf{S} = [\underbrace{\text{BOS}}_{\text{Token 0}}, \underbrace{\mathbf{V}_1, \dots, \mathbf{V}_{256}}_{\text{256 DINOv2+SigLIP Patch Embeddings}}, \underbrace{P_1, \dots, P_M}_{\text{Prompt Tokens}}, \underbrace{T_0, \dots, T_{13}}_{\text{14 Action Tokens}}]$$

- **Prompt**: `"In: What action should the robot take to pick up the yellow screwdriver and place it into the blue cup?\nOut:"`
- **Label Masking**: All visual patch tokens and prompt tokens are assigned target label `-100` (`torch.nn.CrossEntropyLoss(ignore_index=-100)`).
- **Objective Function**:
  $$\mathcal{L} = -\frac{1}{14} \sum_{i=0}^{13} \log P\left(T_i \mid \text{Image}, \text{Prompt}, T_{<i}\right)$$

---

## 5. Post-Training Evaluation Engine (`colab_runner.py`)

When training completes, the adapter weights are saved to `./openvla_aloha_output/model_a_raw_lora` ($\sim 95\text{ MB}$).

### 5.1 Closed-Loop Execution Loop
The evaluation engine runs closed-loop control in headless MuJoCo across the 15 holdout test episodes:
1. **Camera Observation**: MuJoCo's overhead camera renders an RGB image of the current scene state.
2. **Autoregressive Generation**: OpenVLA predicts 14 tokens:
   `out_tokens = model.generate(input_ids, max_new_tokens=14)`
3. **De-quantization (Tokens to Physical Radians)**:
   The predicted token IDs ($31744 \le T_i \le 31999$) are mapped inversely back to continuous joint radians:
   $$a_i = \text{min}_i + \frac{T_i - 31744}{255} \times (\text{max}_i - \text{min}_i)$$
4. **Gripper Calibration**:
   $$\text{slide} = 0.002 + \text{clip}\left(\frac{a}{0.69}, 0, 1\right) \times 0.035$$
5. **Actuation & Physics Step**: 
   The continuous radians are sent to the 14 MuJoCo actuators (6 joints + calibrated linear slide gripper per arm) and the simulation advances.

### 5.2 Verification Metrics
1. **Grasp & Lift Success**: Right arm gripper lifts screwdriver to height $Z \ge 0.05\text{ m}$.
2. **Cup Placement Success**: Screwdriver center is located within $R \le 0.070\text{ m}$ of the cup center $[-0.12, 0.19]$ at final frame and height $Z < 0.10\text{ m}$.
3. **Trajectory Tracking Error (L1 Error)**: Mean absolute error in radians of predicted joint trajectories versus expert demonstration telemetry.

### 5.3 Per-Episode Video Recording & Incremental Checkpointing
* **Immediate MP4 Video Export**: `eval_episode_{ep_id}.mp4` is encoded and saved via `mediapy` immediately as each individual episode completes its rollout. Inspection does not require waiting for the full 15-episode test batch.
* **Incremental Scorecard Flushing**: Real-time evaluation metrics are written to `openvla_benchmark_results.json` after every episode, ensuring all benchmark data is preserved if an evaluation run is paused or interrupted.

---

## 6. Day 2: AI Video Data Augmentation & Model B

The complete experimental hypothesis compares:

| Configuration | Model A (Baseline) | Model B (Augmented) |
| :--- | :--- | :--- |
| **Training Demonstrations** | 30 Real Physical Episodes | 30 Real Physical Episodes + AI Video Variations |
| **Visual Diversity** | Static black table, fixed lab lighting | Varied table textures (wood, metal, granite), varied lighting/shadows, distractor items |
| **Model Architecture** | OpenVLA-7B (4-bit LoRA Rank 32) | OpenVLA-7B (4-bit LoRA Rank 32) |
| **Evaluation Set** | 15 Unseen Holdout Episodes (35–49) | 15 Unseen Holdout Episodes (35–49) |
| **Evaluation Environment** | MuJoCo Physics Digital Twin | MuJoCo Physics Digital Twin |

The goal is to determine if generative video models can serve as effective visual data augmeters for physical robot policies, improving grasping robustness and spatial generalization without collecting additional hardware demonstrations.
