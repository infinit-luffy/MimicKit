"""
AMP Continual Learning Agent.

Extends AMPAgent with:
- Motion one-hot conditioning (actor/critic receive [obs, one-hot])
- CBP (Continual Backpropagation) for selective neuron reinitialization
- SGP (Subspace Gradient Projection) to protect learned motion subspaces
- Per-task discriminator reset

Follows the same conditioning pattern as ASEAgent (ase_agent.py).
"""

import os
import numpy as np
import torch
import torch.nn as nn

import learning.amp_agent as amp_agent
import learning.base_agent as base_agent
import learning.cl.amp_cl_model as amp_cl_model
import learning.cl.cbp_linear as cbp_linear
import learning.cl.sgp_memory as sgp_memory
import learning.mp_optimizer as mp_optimizer
import learning.rl_util as rl_util
import util.mp_util as mp_util
import util.torch_util as torch_util
import envs.base_env as base_env
from util.logger import Logger


class AMPCLAgent(amp_agent.AMPAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        self._build_cbp_wrappers()
        self._init_cl_state()
        return

    def _load_params(self, config):
        super()._load_params(config)

        self._max_motions = config["max_motions"]
        self._enable_cbp = config.get("enable_cbp", False)
        self._cbp_replacement_rate = config.get("cbp_replacement_rate", 1e-4)
        self._cbp_maturity_threshold = config.get("cbp_maturity_threshold", 1000)
        self._sgp_threshold = config.get("sgp_threshold", 0.98)
        self._sgp_gpm_ratio_threshold = config.get("sgp_gpm_ratio_threshold", 0.02)
        return

    def _build_model(self, config):
        model_config = config["model"]
        model_config["max_motions"] = config["max_motions"]
        self._model = amp_cl_model.AMPCLModel(model_config, self._env)
        return

    # ------------------------------------------------------------------
    # CBP wrapper construction
    # ------------------------------------------------------------------

    def _build_cbp_wrappers(self):
        """Create CBPLinear modules wrapping consecutive Linear pairs in the actor."""
        self._cbp_modules = nn.ModuleList()
        self._layer_cbp_map = {}  # maps Linear.weight -> CBPLinear
        self._activation_hooks = []

        if not self._enable_cbp:
            Logger.print("CBP disabled by config (enable_cbp=False)")
            return

        linear_layers = []
        for module in self._model._actor_layers.modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(module)

        # Create CBP wrappers for consecutive pairs: (layer0, layer1), (layer1, layer2), ...
        for i in range(len(linear_layers) - 1):
            in_layer = linear_layers[i]
            out_layer = linear_layers[i + 1]
            cbp = cbp_linear.CBPLinear(
                in_layer=in_layer,
                out_layer=out_layer,
                replacement_rate=self._cbp_replacement_rate,
                maturity_threshold=self._cbp_maturity_threshold,
            )
            self._cbp_modules.append(cbp)
            self._layer_cbp_map[in_layer.weight] = cbp

        Logger.print("Built {} CBP wrappers for actor network".format(len(self._cbp_modules)))

        # Register activation hooks on ReLU layers to feed CBPLinear feature tracking
        relu_idx = 0
        for module in self._model._actor_layers.modules():
            if isinstance(module, (nn.ReLU, nn.ELU, nn.GELU)):
                if relu_idx < len(self._cbp_modules):
                    cbp_mod = self._cbp_modules[relu_idx]

                    def make_hook(cbp_m):
                        def hook(mod, inp, out):
                            # Feed output of activation (=input to next Linear) into CBP
                            with torch.no_grad():
                                x = out.detach().mean(dim=0)
                                if cbp_m.decay_rate == 0:
                                    cbp_m.features = x
                                else:
                                    if cbp_m.features is None:
                                        cbp_m.features = (1 - cbp_m.decay_rate) * x
                                    else:
                                        cbp_m.features = (cbp_m.features * cbp_m.decay_rate
                                                          + (1 - cbp_m.decay_rate) * x)
                        return hook

                    h = module.register_forward_hook(make_hook(cbp_mod))
                    self._activation_hooks.append(h)
                    relu_idx += 1
        return

    def _init_cl_state(self):
        """Initialize continual learning state."""
        num_envs = self.get_num_envs()
        self._current_task_id = 0
        self._motion_ids_buf = torch.zeros(num_envs, device=self._device, dtype=torch.long)
        self._motion_onehot_buf = self._model.get_motion_onehot(self._motion_ids_buf)

        # SGP state
        self._sgp_feature_mats = []  # per-layer projection matrices
        self._sgp_anchors = []       # per-layer anchor weight projections
        return

    # ------------------------------------------------------------------
    # Motion management
    # ------------------------------------------------------------------

    def set_current_motion(self, task_id):
        """Set all environments to imitate a specific motion."""
        self._current_task_id = task_id
        num_envs = self.get_num_envs()
        self._motion_ids_buf = torch.full(
            (num_envs,), task_id, device=self._device, dtype=torch.long
        )
        self._motion_onehot_buf = self._model.get_motion_onehot(self._motion_ids_buf)
        return

    # ------------------------------------------------------------------
    # Action decision (override PPOAgent._decide_action)
    # ------------------------------------------------------------------

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        motion_onehot = self._motion_onehot_buf

        norm_action_dist = self._model.eval_actor(norm_obs, motion_onehot)

        if self._mode == base_agent.AgentMode.TRAIN:
            norm_a_rand = norm_action_dist.sample()
            norm_a_mode = norm_action_dist.mode

            exp_prob = self._get_exp_prob()
            exp_prob = torch.full([norm_a_rand.shape[0], 1], exp_prob,
                                  device=self._device, dtype=torch.float)
            rand_action_mask = torch.bernoulli(exp_prob)
            norm_a = torch.where(rand_action_mask == 1.0, norm_a_rand, norm_a_mode)
            rand_action_mask = rand_action_mask.squeeze(-1)

        elif self._mode == base_agent.AgentMode.TEST:
            norm_a = norm_action_dist.mode
            rand_action_mask = torch.zeros_like(norm_a[..., 0])
        else:
            assert False, "Unsupported agent mode: {}".format(self._mode)

        norm_a_logp = norm_action_dist.log_prob(norm_a)
        norm_a = norm_a.detach()
        norm_a_logp = norm_a_logp.detach()
        a = self._a_norm.unnormalize(norm_a)

        a_info = {
            "a_logp": norm_a_logp,
            "rand_action_mask": rand_action_mask
        }
        return a, a_info

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("motion_onehot", self._motion_onehot_buf)
        return

    # ------------------------------------------------------------------
    # Training data (override to pass one-hot to critic for advantage)
    # ------------------------------------------------------------------

    def _build_train_data(self):
        self.eval()

        self._record_disc_demo_data()
        self._store_disc_replay_data()

        reward_info = self._compute_rewards()

        # Compute advantages with motion-conditioned critic
        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        motion_onehot = self._exp_buffer.get_data("motion_onehot")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")

        norm_next_obs = self._obs_norm.normalize(next_obs)
        next_critic_inputs = {"obs": norm_next_obs, "motion_onehot": motion_onehot}
        next_vals = torch_util.eval_minibatch(
            self._model.eval_critic, next_critic_inputs, self._critic_eval_batch_size
        )
        next_vals = next_vals.squeeze(-1).detach()

        succ_val = self._compute_succ_val()
        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        next_vals[succ_mask] = succ_val

        fail_val = self._compute_fail_val()
        fail_mask = (done == base_env.DoneFlags.FAIL.value)
        next_vals[fail_mask] = fail_val

        new_vals = rl_util.compute_td_lambda_return(r, next_vals, done,
                                                     self._discount, self._td_lambda)

        norm_obs = self._obs_norm.normalize(obs)
        critic_inputs = {"obs": norm_obs, "motion_onehot": motion_onehot}
        vals = torch_util.eval_minibatch(
            self._model.eval_critic, critic_inputs, self._critic_eval_batch_size
        )
        vals = vals.squeeze(-1).detach()
        adv = new_vals - vals

        rand_action_mask_flat = (rand_action_mask == 1.0).flatten()
        adv_flat = adv.flatten()
        rand_action_adv = adv_flat[rand_action_mask_flat]
        adv_mean, adv_std = mp_util.calc_mean_std(rand_action_adv)
        norm_adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
        norm_adv = torch.clamp(norm_adv, -self._norm_adv_clip, self._norm_adv_clip)

        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("adv", norm_adv)

        info = {
            **reward_info,
            "adv_mean": adv_mean,
            "adv_std": adv_std
        }
        return info

    # ------------------------------------------------------------------
    # Loss computation (override to pass one-hot)
    # ------------------------------------------------------------------

    def _compute_actor_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        norm_a = self._a_norm.normalize(batch["action"])
        old_a_logp = batch["a_logp"]
        adv = batch["adv"]
        rand_action_mask = batch["rand_action_mask"]
        motion_onehot = batch["motion_onehot"]

        rand_action_mask = (rand_action_mask == 1.0)
        norm_obs = norm_obs[rand_action_mask]
        norm_a = norm_a[rand_action_mask]
        old_a_logp = old_a_logp[rand_action_mask]
        adv = adv[rand_action_mask]
        motion_onehot = motion_onehot[rand_action_mask]

        a_dist = self._model.eval_actor(norm_obs, motion_onehot)
        a_logp = a_dist.log_prob(norm_a)

        a_ratio = torch.exp(a_logp - old_a_logp)
        actor_loss0 = adv * a_ratio
        actor_loss1 = adv * torch.clamp(a_ratio, 1.0 - self._ppo_clip_ratio,
                                         1.0 + self._ppo_clip_ratio)
        actor_loss = torch.minimum(actor_loss0, actor_loss1)
        actor_loss = -torch.mean(actor_loss)

        clip_frac = (torch.abs(a_ratio - 1.0) > self._ppo_clip_ratio).type(torch.float)
        clip_frac = torch.mean(clip_frac)
        imp_ratio = torch.mean(a_ratio)

        info = {
            "actor_loss": actor_loss,
            "clip_frac": clip_frac.detach(),
            "imp_ratio": imp_ratio.detach()
        }

        if self._action_bound_weight != 0:
            action_bound_loss = self._compute_action_bound_loss(a_dist)
            if action_bound_loss is not None:
                action_bound_loss = torch.mean(action_bound_loss)
                actor_loss += self._action_bound_weight * action_bound_loss
                info["action_bound_loss"] = action_bound_loss.detach()

        if self._action_entropy_weight != 0:
            action_entropy = a_dist.entropy()
            action_entropy = torch.mean(action_entropy)
            actor_loss += -self._action_entropy_weight * action_entropy
            info["action_entropy"] = action_entropy.detach()

        if self._action_reg_weight != 0:
            action_reg_loss = a_dist.param_reg()
            action_reg_loss = torch.mean(action_reg_loss)
            actor_loss += self._action_reg_weight * action_reg_loss
            info["action_reg_loss"] = action_reg_loss.detach()

        return info

    def _compute_critic_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        tar_val = batch["tar_val"]
        motion_onehot = batch["motion_onehot"]
        pred = self._model.eval_critic(obs=norm_obs, motion_onehot=motion_onehot)
        pred = pred.squeeze(-1)

        diff = tar_val - pred
        loss = torch.mean(torch.square(diff))

        info = {
            "critic_loss": loss
        }
        return info

    # ------------------------------------------------------------------
    # Actor update with SGP gradient projection and CBP
    # ------------------------------------------------------------------

    def _update_actor(self, batch_size, num_steps):
        info = dict()

        for i in range(num_steps):
            batch = self._exp_buffer.sample(batch_size)
            loss_info = self._compute_actor_loss(batch)
            loss = loss_info["actor_loss"]

            # Use step_with_grad_hook to inject SGP projection
            self._actor_optimizer.step_with_grad_hook(
                loss, pre_step_fn=self._apply_sgp_projection
            )

            # CBP reinit + anchor correction after optimizer step
            self._trigger_cbp_reinit()
            self._apply_anchor_correction()

            torch_util.add_torch_dict(loss_info, info)

        torch_util.scale_torch_dict(1.0 / num_steps, info)
        return info

    # ------------------------------------------------------------------
    # SGP: Gradient projection onto orthogonal complement of protected subspace
    # ------------------------------------------------------------------

    def _apply_sgp_projection(self):
        """Project actor gradients away from protected task subspaces."""
        if len(self._sgp_feature_mats) == 0:
            return

        kk = 0
        self._sgp_call_count = getattr(self, '_sgp_call_count', 0) + 1

        for name, param in self._model._actor_layers.named_parameters():
            if "weight" in name and param.dim() > 1:
                if kk >= len(self._sgp_feature_mats):
                    break

                P = self._sgp_feature_mats[kk]
                if P is not None and param.grad is not None:
                    if P.device != param.device:
                        P = P.to(param.device)
                        self._sgp_feature_mats[kk] = P

                    sz = param.grad.data.size(0)
                    flat_grad = param.grad.data.view(sz, -1)

                    grad_norm_before = flat_grad.norm().item()
                    proj = torch.mm(flat_grad, P)
                    proj_norm = proj.norm().item()
                    param.grad.data = param.grad.data - proj.view(param.grad.shape)
                    grad_norm_after = param.grad.data.norm().item()

                    # Log every 100 calls
                    if self._sgp_call_count % 100 == 1:
                        print("[SGP proj] layer={} P_shape={} grad_before={:.4f} projected={:.4f} grad_after={:.4f}".format(
                            name, list(P.shape), grad_norm_before, proj_norm, grad_norm_after))

                kk += 1

    # ------------------------------------------------------------------
    # CBP: Trigger neuron reinitialization
    # ------------------------------------------------------------------

    def _trigger_cbp_reinit(self):
        """Trigger CBP reinit on all CBP modules and reset optimizer state."""
        if not self._enable_cbp or len(self._cbp_modules) == 0:
            return

        # Build per-layer feature_mat / next_feature_mat from SGP projection matrices
        for idx, cbp_mod in enumerate(self._cbp_modules):
            if not cbp_mod.training:
                continue
            if cbp_mod.features is None:
                continue

            feature_mat = None
            next_feature_mat = None
            if idx < len(self._sgp_feature_mats):
                feature_mat = self._sgp_feature_mats[idx]
            if (idx + 1) < len(self._sgp_feature_mats):
                next_feature_mat = self._sgp_feature_mats[idx + 1]

            features_to_replace = cbp_mod.reinit(feature_mat, next_feature_mat)

            # Reset optimizer state for replaced neurons
            if features_to_replace.numel() > 0:
                cbp_mod.update_optim_params_adam(
                    features_to_replace, self._actor_optimizer._optimizer
                )

    # ------------------------------------------------------------------
    # Anchor correction: fix weight drift in protected subspace after CBP reinit
    # ------------------------------------------------------------------

    def _apply_anchor_correction(self):
        """Correct weight drift for reset neurons in the protected subspace."""
        if not self._enable_cbp or len(self._sgp_anchors) == 0:
            return

        kk = 0
        with torch.no_grad():
            for name, param in self._model._actor_layers.named_parameters():
                if "weight" in name and param.dim() > 1:
                    if kk >= len(self._sgp_anchors):
                        break

                    Anchor = self._sgp_anchors[kk]
                    P = self._sgp_feature_mats[kk] if kk < len(self._sgp_feature_mats) else None
                    cbp_module = self._layer_cbp_map.get(param)

                    if (cbp_module is not None and Anchor is not None and P is not None
                            and getattr(cbp_module, 'reset_mask', None) is not None):

                        device = param.device
                        Anchor = Anchor.to(device)
                        P = P.to(device)
                        Mask = cbp_module.reset_mask.to(device)

                        sz = param.data.size(0)
                        flat_w = param.data.view(sz, -1)
                        current_proj = torch.mm(flat_w, P)
                        delta = current_proj - Anchor
                        mask_broadcast = Mask.view(-1, 1)
                        delta_masked = delta * mask_broadcast
                        param.data = param.data - delta_masked.view(param.shape)

                    kk += 1

    # ------------------------------------------------------------------
    # Task transition: prepare for a new CL task
    # ------------------------------------------------------------------

    def prepare_for_new_task(self, task_id, sgp_feature_mats, sgp_anchors):
        """Prepare the agent for continual learning of a new motion task.

        Args:
            task_id: Integer ID of the new task.
            sgp_feature_mats: List of projection matrices from SGPMemoryBank.
            sgp_anchors: List of anchor weight projections from SGPMemoryBank.
        """
        Logger.print("Preparing for CL task {} ...".format(task_id))

        # 1. Set task_id on all CBP modules (resets utility tracking)
        if self._enable_cbp:
            for cbp_mod in self._cbp_modules:
                cbp_mod.set_task_id(task_id)

        # 2. Store SGP projection matrices and anchors
        self._sgp_feature_mats = sgp_feature_mats
        self._sgp_anchors = sgp_anchors
        self._sgp_call_count = 0

        num_valid = sum(1 for p in sgp_feature_mats if p is not None)
        Logger.print("SGP: {} projection matrices loaded ({} valid)".format(
            len(sgp_feature_mats), num_valid))
        for i, P in enumerate(sgp_feature_mats):
            if P is not None:
                Logger.print("  P[{}]: shape={}, rank~={}".format(
                    i, list(P.shape), (P.diagonal() > 0.5).sum().item()))

        # 3. Reset discriminator for new motion
        self._reset_discriminator()

        # 4. Update motion embedding
        self.set_current_motion(task_id)

        Logger.print("CL task {} preparation complete".format(task_id))
        return

    def _reset_discriminator(self):
        """Reset discriminator weights and replay buffer for a new motion."""
        Logger.print("Resetting discriminator for new task...")

        # Reset model weights
        self._model.reset_discriminator()

        # Clear replay buffer
        self._disc_buffer.clear()

        # Rebuild disc optimizer with fresh state
        disc_config = self._config["disc_optimizer"]
        disc_params = list(self._model.get_disc_params())
        disc_params = [p for p in disc_params if p.requires_grad]
        self._disc_optimizer = mp_optimizer.MPOptimizer(disc_config, disc_params)
        return

    # ------------------------------------------------------------------
    # Save / Load with CL state
    # ------------------------------------------------------------------

    def save(self, out_file):
        if mp_util.is_root_proc():
            state_dict = self.state_dict()
            # Add CL metadata
            cl_state = {
                'model_state': state_dict,
                'current_task_id': self._current_task_id,
                'sgp_feature_mats': self._sgp_feature_mats,
                'sgp_anchors': self._sgp_anchors,
            }
            torch.save(cl_state, out_file)
        return

    def load(self, in_file):
        data = torch.load(in_file, map_location=self._device, weights_only=False)

        # Support both CL-format and standard format checkpoints
        if isinstance(data, dict) and 'model_state' in data:
            self.load_state_dict(data['model_state'], strict=False)
            self._current_task_id = data.get('current_task_id', 0)
            self._sgp_feature_mats = data.get('sgp_feature_mats', [])
            self._sgp_anchors = data.get('sgp_anchors', [])
        else:
            # Standard checkpoint (e.g., from pre-trained AMP model)
            self.load_state_dict(data, strict=False)

        self._sync_optimizer()
        Logger.print("Loaded CL model from {:s}".format(in_file))
        return

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        super()._log_train_info(train_info, test_info, env_diag_info, start_time)
        self._logger.log("CL_Task_ID", self._current_task_id)

        # Log CBP replacement stats
        total_replacements = sum(cbp.total_replacements for cbp in self._cbp_modules)
        self._logger.log("CBP_Total_Replacements", total_replacements)
        return
