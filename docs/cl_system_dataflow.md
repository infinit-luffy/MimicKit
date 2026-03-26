# MimicKit Continual Learning System - Data Flow Documentation

## Overview

The CL system enables a single policy to sequentially learn multiple motion skills (walk, run, punch, kick, etc.) without catastrophic forgetting. Each motion is treated as a separate CL task. The core protection mechanism is **SGP (Subspace Gradient Projection)**, with optional **CBP (Continual Backpropagation)** for plasticity maintenance.

## Architecture

```
Class Hierarchy:
  BaseAgent -> PPOAgent -> AMPAgent -> AMPCLAgent    (learning/cl/amp_cl_agent.py)
  BaseModel -> PPOModel -> AMPModel -> AMPCLModel    (learning/cl/amp_cl_model.py)

Supporting modules:
  SGPMemoryBank                                      (learning/cl/sgp_memory.py)
  CBPLinear                                          (learning/cl/cbp_linear.py)

Orchestration:
  cl_run.py                                          (top-level training script)
```

## Network Structure

```
Actor Network (SGP protects all 3 Linear layers):
  Input: [norm_obs, motion_onehot]   dim = obs_dim + max_motions

  _actor_layers (nn.Sequential):
    Linear(obs_dim+max_motions, 1024)   ← SGP P[0], bias frozen after task 0
    ReLU
    Linear(1024, 512)                   ← SGP P[1], bias frozen after task 0
    ReLU

  _action_dist (DistributionGaussianDiagBuilder):
    _mean_net: Linear(512, action_dim)  ← SGP P[2], bias frozen after task 0
    _logstd: Parameter(action_dim)      (not projected, does not affect test mode)

Critic Network (not SGP-protected, conditioned on one-hot):
  Input: [norm_obs, motion_onehot]
  _critic_layers: Linear -> ReLU -> Linear -> ReLU
  _critic_out: Linear -> scalar value

Discriminator (reset per task, not conditioned on one-hot):
  _disc_layers: Linear -> ReLU -> Linear -> ReLU
  _disc_logits: Linear -> scalar logit
```

## Complete Training Flow (per curriculum)

```
cl_run.py: run()
│
├─ Build env ONCE (IsaacGym PhysX Foundation is a singleton)
├─ Create SGPMemoryBank(device, threshold=0.98)
│
├─ FOR each stage (task_id = 0, 1, 2, ...):
│   │
│   ├─ train_cl_stage()
│   │   │
│   │   ├─ A. Swap motion file:  env._load_motions(motion_file)
│   │   │
│   │   ├─ B. Build fresh agent: agent_builder.build_agent_from_config()
│   │   │      └─ AMPCLAgent.__init__()
│   │   │           ├─ AMPAgent.__init__() → builds model, optimizers, normalizers
│   │   │           ├─ _build_cbp_wrappers() → (no-op if enable_cbp=False)
│   │   │           └─ _init_cl_state() → motion_ids_buf, sgp_feature_mats=[]
│   │   │
│   │   ├─ C. Load previous model + inject CL protections:
│   │   │   │
│   │   │   ├─ [task_id == 0]:
│   │   │   │     agent.set_current_motion(0)  → one-hot = [1,0,0,...,0]
│   │   │   │
│   │   │   └─ [task_id > 0]:
│   │   │         agent.load(prev_model)  → weights + obs_norm stats
│   │   │         │
│   │   │         sgp_mem.build_projection_matrices()
│   │   │         │  └─ For each layer:
│   │   │         │       concat all historical U matrices → SVD → P = U_final @ U_final^T
│   │   │         │
│   │   │         sgp_mem.build_anchor_weights(actor_layers, P_list, output_layer)
│   │   │         │  └─ For each Linear: Anchor = W @ P
│   │   │         │
│   │   │         agent.prepare_for_new_task(task_id, feature_mats, anchors)
│   │   │              ├─ 1. CBP: set_task_id() → reset utility tracking
│   │   │              ├─ 2. _freeze_actor_bias() → all actor bias requires_grad=False
│   │   │              │       └─ Rebuild actor optimizer (excludes frozen params)
│   │   │              ├─ 3. Store sgp_feature_mats, sgp_anchors on agent
│   │   │              ├─ 4. _reset_discriminator()
│   │   │              │       ├─ model.reset_discriminator() → xavier reinit weights
│   │   │              │       ├─ disc_buffer.clear()
│   │   │              │       ├─ Reset disc_obs_norm (new Normalizer instance)
│   │   │              │       └─ Rebuild disc optimizer
│   │   │              └─ 5. set_current_motion(task_id) → update one-hot buffer
│   │   │
│   │   ├─ D. Train: agent.train_model(max_samples, out_dir, ...)
│   │   │   │
│   │   │   └─ [Training loop - see detailed flow below]
│   │   │
│   │   ├─ E. Extract SGP features for this task:
│   │   │      collect_obs_for_sgp(env, agent, n_steps=20)
│   │   │      │  └─ Roll out policy → collect [norm_obs, one_hot] per step
│   │   │      │     Returns: Tensor [n_steps * num_envs, obs_dim + max_motions]
│   │   │      │
│   │   │      sgp_mem.extract_features(actor_layers, obs_data, output_layer)
│   │   │      │  ├─ Hook all Linear layers (actor_layers + _mean_net)
│   │   │      │  ├─ Forward obs_data through actor_layers → _action_dist
│   │   │      │  ├─ Per hooked layer: R = concat(input activations)
│   │   │      │  ├─ G = R^T @ R  (covariance-like matrix)
│   │   │      │  ├─ SVD(G) → U, S, V
│   │   │      │  ├─ k = min dims to explain 98% variance
│   │   │      │  └─ Store U[:, :k] as basis for this task-layer
│   │   │      │
│   │   │      sgp_mem.memory_bank.append(features)
│   │   │
│   │   └─ F. Save model.pt + sgp_memory.pt
│   │
│   ├─ evaluate_after_stage()  → test all tasks 0..task_id
│   │      └─ For each learned task:
│   │           swap motion → load model → set_current_motion(j)
│   │           → test_model(10 episodes) → record mean_ep_len
│   │
│   └─ Append row to performance matrix R[i, :]
│
├─ print_performance_matrix()
├─ compute_cl_metrics() → AP, BWT, Forgetting
└─ Save cl_metrics.yaml + sgp_memory_final.pt
```

## Detailed Training Loop (Single Stage)

```
agent.train_model(max_samples):
│
└─ PER ITERATION:
    │
    ├─ 1. Rollout: _rollout_train(num_steps)
    │      FOR each step:
    │        ├─ _decide_action(obs, info)
    │        │     norm_obs = obs_norm.normalize(obs)   ← FROZEN after task 0
    │        │     actor_input = cat([norm_obs, motion_onehot])
    │        │     action_dist = model.eval_actor(norm_obs, motion_onehot)
    │        │     action = sample or mode (train vs test)
    │        │
    │        ├─ _record_data_pre_step()
    │        │     exp_buffer.record("obs", "action", "motion_onehot")
    │        │     obs_norm.record(obs)  ← SKIPPED if task_id > 0
    │        │
    │        ├─ env.step(action) → next_obs, reward, done, info
    │        │
    │        └─ _record_data_post_step()
    │              exp_buffer.record("next_obs", "reward", "done")
    │
    ├─ 2. Build train data: _build_train_data()
    │      ├─ _record_disc_demo_data() → sample demo obs from motion_lib
    │      ├─ _store_disc_replay_data() → add current obs to disc replay buffer
    │      ├─ _compute_rewards()
    │      │     ├─ task_reward (from env)
    │      │     ├─ disc_reward:
    │      │     │     norm_disc_obs = disc_obs_norm.normalize(disc_obs)
    │      │     │     disc_r = model.eval_disc(norm_disc_obs)  → sigmoid logit
    │      │     └─ r = task_weight * task_r + disc_weight * disc_r
    │      │
    │      └─ Compute advantages (GAE with motion-conditioned critic):
    │            V(s) = model.eval_critic(norm_obs, motion_onehot)
    │            adv = TD(lambda) returns - V(s)
    │            Normalize advantages
    │
    ├─ 3. Update actor: _update_actor(batch_size, num_steps)
    │      FOR each gradient step:
    │        ├─ Sample batch from exp_buffer
    │        ├─ _compute_actor_loss(batch)
    │        │     ├─ model.eval_actor(norm_obs, motion_onehot)
    │        │     ├─ PPO clipped surrogate loss
    │        │     ├─ + action_bound_loss
    │        │     ├─ + entropy bonus
    │        │     └─ + action regularization
    │        │
    │        ├─ actor_optimizer.step_with_grad_hook(loss, pre_step_fn)
    │        │     ├─ loss.backward()          ← compute gradients
    │        │     ├─ _apply_sgp_projection()  ← PROJECT GRADIENTS (core SGP)
    │        │     │     FOR each protected param (actor_layers + _mean_net):
    │        │     │       flat_grad = grad.view(out_dim, in_dim)
    │        │     │       projected = flat_grad @ P[layer]
    │        │     │       grad = grad - projected
    │        │     │       (removes gradient component in protected subspace)
    │        │     └─ optimizer.step()         ← update with projected gradients
    │        │
    │        ├─ _trigger_cbp_reinit()  (no-op if CBP disabled)
    │        └─ _apply_anchor_correction()  (no-op if CBP disabled)
    │
    ├─ 4. Update critic: _update_critic(batch_size, num_steps)
    │      ├─ model.eval_critic(norm_obs, motion_onehot)
    │      └─ MSE loss vs target values (NOT SGP-protected)
    │
    ├─ 5. Update discriminator: _update_disc(batch)
    │      ├─ Demo obs from motion_lib (current task's motion)
    │      ├─ Agent obs from replay buffer
    │      ├─ disc_obs_norm.normalize() → disc model
    │      └─ Binary cross-entropy + gradient penalty
    │
    └─ 6. Log: _log_train_info()
           ├─ Standard: return, ep_len, actor_loss, critic_loss, disc_loss
           ├─ CL_Task_ID
           └─ CBP_Total_Replacements
```

## SGP Protection Mechanism (Math)

```
Goal: When learning task k, do not modify weights in directions that are
      important for tasks 0..k-1.

1. FEATURE EXTRACTION (after completing task i):
   For each Linear layer l with weight W_l [out_dim, in_dim]:
     - Collect input activations R_l [N_samples, in_dim] during policy rollout
     - Compute covariance: G_l = R_l^T @ R_l  [in_dim, in_dim]
     - SVD(G_l) = U_l @ diag(S_l) @ V_l^T
     - Keep top-k columns of U_l (explaining 98% of variance)
     - Store U_l[:, :k] in memory bank

2. PROJECTION MATRIX CONSTRUCTION (before training task k):
   For each layer l:
     - Concatenate U matrices from all previous tasks: U_cat = [U_0 | U_1 | ... | U_{k-1}]
     - Re-orthogonalize: SVD(U_cat @ U_cat^T) → U_final (keep significant singular values)
     - Build projection: P_l = U_final @ U_final^T  [in_dim, in_dim]
     - P is idempotent (P^2 = P), projects onto protected subspace

3. GRADIENT PROJECTION (during training of task k):
   After loss.backward(), before optimizer.step():
     For each protected layer l:
       grad_l = W_l.grad                    [out_dim, in_dim]
       proj_l = grad_l @ P_l                [out_dim, in_dim]  (component in protected subspace)
       W_l.grad = grad_l - proj_l           (keep only orthogonal complement)

   Effect: Gradient can only update weights in directions orthogonal to the
   protected subspace. Previously learned representations are preserved.
```

## Protection Summary (What is Protected vs. Not)

```
PROTECTED (frozen or SGP-projected):
  ✓ _actor_layers weights         (SGP gradient projection)
  ✓ _action_dist._mean_net weight (SGP gradient projection)
  ✓ All actor bias parameters     (frozen requires_grad=False after task 0)
  ✓ _obs_norm (observation normalizer)  (stops updating after task 0)

NOT PROTECTED (changes freely per task):
  ✗ _critic_layers + _critic_out  (not SGP-protected, learns per-task values)
  ✗ _disc_layers + _disc_logits   (reset and retrained per task)
  ✗ _disc_obs_norm                (reset per task)
  ✗ _action_dist._logstd          (1D param, not projected; no effect in test mode)
```

## Key Configurations

```yaml
# data/agents/amp_cl_humanoid_agent.yaml
agent_name: "AMP_CL"
max_motions: 20          # one-hot dimension (supports up to 20 tasks)
enable_cbp: false        # CBP disabled, only SGP active
sgp_threshold: 0.98      # SVD variance threshold (higher = more protection)

# data/curriculums/cl_humanoid_example.yaml
max_samples: 50000000    # per-stage training budget
num_envs: 4096
stages:
  - name: "walk"         # task_id=0, one-hot=[1,0,...,0]
  - name: "run"          # task_id=1, one-hot=[0,1,...,0]
  - name: "punch"        # task_id=2, ...
  ...
```

## Output Directory Structure

```
output/cl_humanoid/
└── 2026-03-26_seed42_AMP_CL/
    ├── training_config.yaml          # full config snapshot
    ├── cl_metrics.yaml               # performance matrix + AP/BWT/FGT
    ├── sgp_memory_final.pt           # full SGP memory bank
    ├── stage_00_walk/
    │   ├── model.pt                  # checkpoint (weights + CL metadata)
    │   ├── sgp_memory.pt             # memory bank after this stage
    │   └── tb/                       # TensorBoard logs
    ├── stage_01_run/
    │   ├── model.pt
    │   ├── sgp_memory.pt
    │   └── tb/
    ...
```

## Evaluation Metrics

```
Performance Matrix R[i,j] = mean_ep_len on task j after training stage i

           task 0   task 1   task 2
stage 0:   R[0,0]     -        -
stage 1:   R[1,0]   R[1,1]     -
stage 2:   R[2,0]   R[2,1]   R[2,2]

Average Performance (AP) = mean(R[K, 0:K])        # final model, all tasks
Backward Transfer  (BWT) = mean(R[K,j] - R[j,j])  # negative = forgetting
Forgetting         (FGT) = mean(max_i R[i,j] - R[K,j])  # peak-to-final drop
```

## File Map

| File | Role |
|------|------|
| `cl_run.py` | Top-level orchestration: stage loop, eval, metrics |
| `learning/cl/amp_cl_agent.py` | CL agent: SGP projection, bias freeze, normalizer freeze, CBP |
| `learning/cl/amp_cl_model.py` | Model: one-hot conditioning, discriminator reset |
| `learning/cl/sgp_memory.py` | SVD feature extraction, P matrix construction, memory bank |
| `learning/cl/cbp_linear.py` | CBP neuron reinitialization (currently disabled) |
| `learning/mp_optimizer.py` | `step_with_grad_hook()` for SGP injection |
| `data/curriculums/cl_humanoid_example.yaml` | 10-motion curriculum definition |
| `data/agents/amp_cl_humanoid_agent.yaml` | CL agent config |
