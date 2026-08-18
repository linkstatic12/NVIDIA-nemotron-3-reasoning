# NVIDIA Nemotron-3 Reasoning Challenge: Upgraded SFT + GRPO Pipeline

This repository contains the training pipeline implemented in the provided notebook[cite: 3]. It is specifically engineered to train the Nemotron-3 Nano 30B model using an upgraded Supervised Fine-Tuning (SFT) and Group Relative Policy Optimisation (GRPO) pipeline on an RTX PRO 6000 (Blackwell SM 12.0) GPU[cite: 3].

## 📋 Required Data Sources
To run this pipeline successfully in a Kaggle environment, you must add the following data sources[cite: 3]:
* **Competition dataset:** `nvidia-nemotron-3-reasoning-challenge`[cite: 3]
* **Offline packages:** `nvidia-nemotron-offline-packages`[cite: 3]
* **Model weights:** `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`[cite: 3]
* **Utility script:** `ryanholbrook/nvidia-utility-script` (Critical for Blackwell ptxas support)[cite: 3]

## 🏗️ Pipeline Architecture
The workflow is divided into four main phases[cite: 3]:

* **Phase 0: Exploratory Data Analysis (EDA)**
  * Classifies puzzles into specific types (e.g., bit manipulation, algebra, sequence, number theory)[cite: 3].
  * Analyzes the distribution of numerical answers[cite: 3].
* **Phase 1: Supervised Fine-Tuning (SFT)**
  * Utilizes LoRA with a rank (`r`) of 32 (the maximum allowed by the competition)[cite: 3].
  * Targets Mamba-H SSM hybrid layers using the regex: `.*\.(in_proj|out_proj|up_proj|down_proj)$`[cite: 3].
  * Trains for 2 epochs with a learning rate of 2e-4 using the `adamw_torch` optimizer[cite: 3].
* **Phase 2: Group Relative Policy Optimisation (GRPO)**
  * Implements reinforcement learning using 8 completions sampled per prompt[cite: 3].
  * Applies a conservative KL penalty (`beta=0.001`) to prevent policy collapse[cite: 3].
  * Utilizes three specific reward signals:
    1. `reward_correct_answer` (+1.0): Primary reward for exact matches or answers within a 1e-4 relative numerical tolerance[cite: 3].
    2. `reward_has_boxed` (+0.2): Format compliance reward for placing the final answer inside `\boxed{}`[cite: 3].
    3. `reward_length` (±0.05–0.1): Length shaping that penalizes trivially short responses (under 20 words) or extreme verbosity (over 1500 words)[cite: 3].
* **Phase 3: Submission Packaging**
  * Saves the resulting LoRA adapter weights[cite: 3].
  * Validates the LoRA rank configuration[cite: 3].
  * Zips exactly three files (`adapter_model.safetensors`, `adapter_config.json`, `README.md`) into a `submission.zip` file[cite: 3].

## 🛠️ Blackwell (SM 12.0) GPU Fixes
Running the Nemotron-H model on a Blackwell GPU requires several critical patches to prevent compilation crashes[cite: 3]. This notebook applies the following five fixes before training[cite: 3]:

1. **Bypass Triton RMSNorm Kernel:** Replaces the hardware-incompatible Triton `rmsnorm_fn` with an identical pure-PyTorch fallback[cite: 3].
2. **Make PTXAS Binary Executable:** Copies the `ptxas-blackwell` binary from the read-only Kaggle utility script into `/tmp` and grants execution permissions[cite: 3].
3. **Stub `mamba_ssm.modules.mamba3`:** Prevents cutlass C++ crashes triggered by Mamba3 imports by providing a dummy stub[cite: 3].
4. **Disable Nemotron-H Fast Path:** Sets `is_fast_path_available = False` after loading the model to avoid broken CUDA kernels[cite: 3].
5. **Triton Env Vars & Version Spoof:** Spoofs the ptxas version to `12.0` and points the `TRITON_PTXAS_BLACKWELL_PATH` to `/tmp/ptxas-blackwell` prior to the first Triton JIT compilation[cite: 3].

## 🚀 Environment Setup
* **Dependencies:** Pre-installed packages (`torch`, `transformers`, `peft`, `triton`, `bitsandbytes`, `kagglehub`) from the Kaggle image must **not** be re-installed, as they carry Blackwell-specific patches[cite: 3]. Only `datasets` and `trl` are installed via offline wheels using `--ignore-installed`[cite: 3].
* **Memory/Compute:** Runs in full `bfloat16` precision across 96 GB VRAM without bitsandbytes quantisation[cite: 3]. All layers are mapped directly to GPU 0 (`device_map={'': 0}`) to maximize throughput[cite: 3].
