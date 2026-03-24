# MimicKit Continual Learning (CL) System

Sequential motion learning without catastrophic forgetting. A single policy learns multiple motions (walk → run → kick → backflip ...) one at a time, retaining all previously learned skills.

## Quick Start

```bash
cd mimickit
python cl_run.py --curriculum_config data/curriculums/cl_humanoid_example.yaml
```

With custom seed and fewer environments (for testing):
```bash
python cl_run.py \
  --curriculum_config data/curriculums/cl_humanoid_example.yaml \
  --rand_seed 42 \
  --num_envs 64 \
  --max_samples 1000000
```

Resume from a specific stage:
```bash
python cl_run.py \
  --curriculum_config data/curriculums/cl_humanoid_example.yaml \
  --start_stage 3 \
  --model_file output/cl_humanoid/.../stage_02_punch/model.pt \
  --sgp_memory_file output/cl_humanoid/.../stage_02_punch/sgp_memory.pt
```

## Architecture

```
AMPCLAgent (extends AMPAgent)
  └─ AMPCLModel (extends AMPModel)
       ├─ actor:  input = [obs, motion_onehot] → FC layers → action
       ├─ critic: input = [obs, motion_onehot] → FC layers → value
       └─ disc:   input = disc_obs only (reset per task)
  └─ SGP: gradient projection to protect learned subspaces
  └─ CBP: selective neuron reinitialization (optional, disabled by default)
```

### Motion Conditioning

Actor and critic receive `cat([obs, one_hot])` as input, where `one_hot` is a `[max_motions]`-dimensional vector indicating which motion to perform. This follows the same latent-conditioning pattern as ASEAgent (which concatenates `[obs, z]`).

The discriminator is **not** conditioned on motion — it only sees the current motion's demo data and is reset between tasks.

### CL Algorithms

**SGP (Subspace Gradient Projection)** — Active by default.
- After each task, collect observations and forward through the actor
- SVD on per-layer activations extracts basis vectors of the learned subspace
- Before training a new task, build projection matrix `P = U @ U^T` from all historical bases
- During training, project gradients away from protected subspace: `grad -= grad @ P`
- This prevents the optimizer from overwriting previously learned weight directions

**CBP (Continual Backpropagation)** — Optional, disabled by default (`enable_cbp: false`).
- Tracks per-neuron utility (contribution = outgoing_weight × activation)
- Periodically reinitializes low-utility neurons with orthogonal initialization
- GPM energy ratio check protects neurons important to previous tasks
- Zeroes outgoing weights of reinitialized neurons to preserve network output

## Training Pipeline

For each stage in the curriculum:

1. **Build env + agent** — Load environment with the stage's motion file
2. **Load previous model** — If task_id > 0, load the previous stage's checkpoint
3. **Inject SGP state** — Build projection matrices from memory bank, compute anchor weights
4. **Prepare for new task** — Reset discriminator, set motion ID, update CBP task_id
5. **Train** — Standard AMP training with SGP gradient projection active
6. **Extract features** — Collect observations, SVD on activations, store in memory bank
7. **Save** — Model checkpoint + SGP memory bank

After all stages, evaluate retention on all learned motions.

## Output Directory Structure

Output is organized by date, seed, and model name:

```
output/cl_humanoid/
  └─ 2026-03-24_seed42_AMP_CL/
       ├── training_config.yaml       # Full training parameters
       ├── stage_00_walk/
       │   ├── model.pt               # Agent checkpoint (with CL metadata)
       │   ├── sgp_memory.pt          # SGP memory bank after this stage
       │   ├── env_config.yaml
       │   ├── agent_config.yaml
       │   ├── engine_config.yaml
       │   ├── stage_info.yaml
       │   └── tb_logs/               # TensorBoard logs (if logger=tb)
       ├── stage_01_run/
       │   └── ...
       ├── ...
       ├── curriculum_summary.yaml    # Summary of all stages
       ├── evaluation_results.yaml    # Per-motion test results
       ├── sgp_memory_final.pt        # Final SGP memory bank
       └── model.pt                   # Copy of final stage model
```

## Configuration

### Curriculum Config (`data/curriculums/cl_humanoid_example.yaml`)

```yaml
out_dir: "output/cl_humanoid"
env_config: "data/envs/amp_humanoid_env.yaml"
agent_config: "data/agents/amp_cl_humanoid_agent.yaml"
engine_config: "data/engines/newton_engine.yaml"
max_samples: 50000000   # per stage
num_envs: 4096
logger: "tb"

stages:
  - name: "walk"
    env_overrides:
      motion_file: "data/motions/humanoid/humanoid_walk.pkl"
  - name: "run"
    env_overrides:
      motion_file: "data/motions/humanoid/humanoid_run.pkl"
  # ... more stages
```

Each stage can also override `max_samples`, `agent_config`, `env_config`, etc.

### Agent Config (`data/agents/amp_cl_humanoid_agent.yaml`)

CL-specific parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_motions` | 20 | Maximum number of motion slots (one-hot dimension) |
| `enable_cbp` | false | Enable CBP neuron reinitialization |
| `cbp_replacement_rate` | 1e-4 | Fraction of neurons replaced per step (CBP) |
| `cbp_maturity_threshold` | 1000 | Min age before neuron is eligible for replacement |
| `sgp_threshold` | 0.98 | SVD variance threshold for feature extraction |
| `sgp_gpm_ratio_threshold` | 0.02 | GPM energy ratio for CBP safety check |

All standard AMP parameters (actor/critic/disc networks, learning rates, PPO hyperparams, etc.) are inherited.

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--curriculum_config` | Path to curriculum YAML (required) |
| `--rand_seed` | Random seed (auto-generated from time if not set) |
| `--num_envs` | Override num_envs from curriculum |
| `--max_samples` | Override max_samples per stage |
| `--start_stage` | Resume from this stage index (0-based) |
| `--end_stage` | Stop after this stage index |
| `--model_file` | Initial model checkpoint (for stage 0 or resuming) |
| `--sgp_memory_file` | SGP memory bank file (for resuming) |
| `--skip_eval` | Skip final evaluation across all motions |
| `--devices` | GPU devices (e.g., `cuda:0`) |

## File Reference

| File | Description |
|------|-------------|
| `cl_run.py` | Top-level training orchestration script |
| `learning/cl/amp_cl_agent.py` | CL agent with SGP/CBP integration |
| `learning/cl/amp_cl_model.py` | Model with motion one-hot conditioning |
| `learning/cl/sgp_memory.py` | SVD feature extraction and projection matrices |
| `learning/cl/cbp_linear.py` | CBP neuron utility tracking and reinitialization |
| `data/agents/amp_cl_humanoid_agent.yaml` | Example agent config |
| `data/curriculums/cl_humanoid_example.yaml` | Example 10-motion curriculum |

## Model Checkpoint Format

CL model checkpoints (`model.pt`) contain:

```python
{
    'model_state': state_dict,       # Full model weights + normalizers
    'current_task_id': int,          # Last trained task ID
    'sgp_feature_mats': list,        # Per-layer projection matrices
    'sgp_anchors': list,             # Per-layer anchor weight projections
}
```

Standard AMP checkpoints can also be loaded (backward compatible).

## TensorBoard Logging

Additional CL-specific metrics logged during training:

- `CL_Task_ID` — Current task being trained
- `CBP_Total_Replacements` — Cumulative neuron reinitializations (when CBP enabled)

## References

- SGP/CBP algorithms ported from [unitree-rl-cl](https://github.com/...)
- AMP: "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Animation" (Peng et al., 2021)
- GPM: "Gradient Projection Memory for Continual Learning" (Saha et al., 2021)
- CBP: "Loss of Plasticity in Deep Continual Learning" (Lyle et al., 2023)
