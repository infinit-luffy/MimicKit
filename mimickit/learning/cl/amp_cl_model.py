"""
AMP Continual Learning Model.

Extends AMPModel with motion one-hot conditioning for the actor and critic.
The discriminator is NOT conditioned on motion — it only sees the current
motion's demo data and is reset between CL tasks.

Follows the same conditioning pattern as ASEModel (ase_model.py), where
eval_actor(obs, z) concatenates [obs, z] before the network.
"""

import gymnasium.spaces as spaces
import numpy as np
import torch

import learning.amp_model as amp_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class AMPCLModel(amp_model.AMPModel):
    def __init__(self, config, env):
        self._max_motions = config["max_motions"]
        super().__init__(config, env)
        return

    def eval_actor(self, obs, motion_onehot):
        in_data = torch.cat([obs, motion_onehot], dim=-1)
        h = self._actor_layers(in_data)
        a_dist = self._action_dist(h)
        return a_dist

    def eval_critic(self, obs, motion_onehot):
        in_data = torch.cat([obs, motion_onehot], dim=-1)
        h = self._critic_layers(in_data)
        val = self._critic_out(h)
        return val

    def get_motion_onehot(self, motion_ids, device=None):
        """Build one-hot vectors from motion IDs.

        Args:
            motion_ids: Tensor of shape [num_envs] with integer motion IDs.
            device: Target device (defaults to motion_ids.device).

        Returns:
            Tensor of shape [num_envs, max_motions].
        """
        if device is None:
            device = motion_ids.device
        onehot = torch.zeros(motion_ids.shape[0], self._max_motions,
                             device=device, dtype=torch.float32)
        onehot.scatter_(1, motion_ids.unsqueeze(1).long(), 1.0)
        return onehot

    def _build_actor_input_dict(self, env):
        obs_space = env.get_obs_space()
        motion_space = self._build_motion_onehot_space()
        input_dict = {
            "obs": obs_space,
            "motion_onehot": motion_space,
        }
        return input_dict

    def _build_critic_input_dict(self, env):
        obs_space = env.get_obs_space()
        motion_space = self._build_motion_onehot_space()
        input_dict = {
            "obs": obs_space,
            "motion_onehot": motion_space,
        }
        return input_dict

    def _build_motion_onehot_space(self):
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=[self._max_motions],
            dtype=np.float32,
        )

    def reset_discriminator(self):
        """Reinitialize discriminator weights for a new motion task."""
        init_output_scale = 1.0

        # Reset disc hidden layers
        for module in self._disc_layers.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # Reset disc output logits
        torch.nn.init.uniform_(self._disc_logits.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._disc_logits.bias)
