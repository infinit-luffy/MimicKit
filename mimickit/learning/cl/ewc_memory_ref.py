"""
Reference-style EWC memory aligned with ref_code/ewc.py.

This version keeps the main reference choices:
- Fisher is estimated from replayed training batches, not fresh env rollouts
- only the actor feature extractor (_actor_layers) is protected
- online EWC accumulates Fisher by direct summation, matching the ref comments
"""

import os
import torch


class ReferenceEWCMemory:
    """Reference-style EWC memory for comparison against the main implementation."""

    def __init__(self, device, ewc_lambda=1000.0, online=False, gamma=0.95):
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma

        self.tasks = []
        self._online_fisher = {}
        self._online_params = {}
        self._task_count = 0

    def _get_ref_params(self, agent):
        for name, param in agent._model._actor_layers.named_parameters():
            if param.requires_grad:
                yield name.replace(".", "__"), param

    def compute_fisher(self, agent, env=None, n_steps=20):
        del env, n_steps

        agent.eval()
        fisher = {}
        for name, param in self._get_ref_params(agent):
            fisher[name] = torch.zeros_like(param)

        sample_count = agent._exp_buffer.get_sample_count()
        if sample_count == 0:
            print("[CL:EWC_REF] Experience buffer empty; returning zero Fisher")
            return fisher

        obs_flat = agent._exp_buffer.get_data_flat("obs")[:sample_count]
        motion_flat = agent._exp_buffer.get_data_flat("motion_onehot")[:sample_count]
        perm = torch.randperm(sample_count, device=obs_flat.device)
        num_mini_batch = min(32, sample_count)
        chunk_size = (sample_count + num_mini_batch - 1) // num_mini_batch

        total_batch = 0
        for start in range(0, sample_count, chunk_size):
            idx = perm[start:start + chunk_size]
            obs_batch = obs_flat[idx]
            motion_onehot = motion_flat[idx]
            batch_size_t = obs_batch.shape[0]

            norm_obs = agent._obs_norm.normalize(obs_batch)
            in_data = torch.cat([norm_obs, motion_onehot], dim=-1)

            agent._model.zero_grad()
            actor_features = agent._model._actor_layers(in_data)
            batch_action_dist = agent._model._action_dist(actor_features)
            sampled_actions = batch_action_dist.sample()
            sampled_action_log_probs = batch_action_dist.log_prob(sampled_actions)
            (-sampled_action_log_probs.mean()).backward()

            for name, param in self._get_ref_params(agent):
                if param.grad is not None:
                    fisher[name] += batch_size_t * (param.grad.detach() ** 2)

            total_batch += batch_size_t

        if total_batch > 0:
            for name in fisher:
                fisher[name] /= float(total_batch)

        print("[CL:EWC_REF] Fisher computed from {} samples, {} params".format(
            total_batch, len(fisher)))
        return fisher

    def register_task(self, agent, fisher):
        params = {}
        for name, param in self._get_ref_params(agent):
            params[name] = param.data.clone()

        if self.online:
            if self._task_count == 1:
                for name, curr in fisher.items():
                    if name in self._online_fisher:
                        fisher[name] = curr + self._online_fisher[name]

            self._online_fisher = {k: v.clone() for k, v in fisher.items()}
            self._online_params = params
            self._task_count = 1
            print("[CL:EWC_REF] Online Fisher updated")
        else:
            self.tasks.append({"params": params, "fisher": fisher})
            self._task_count += 1
            print("[CL:EWC_REF] Task registered. Total tasks: {}".format(len(self.tasks)))

    def compute_ewc_loss(self, agent):
        loss = torch.tensor(0.0, device=self.device)

        if self.online:
            if self._task_count == 0:
                return loss

            for name, param in self._get_ref_params(agent):
                if name in self._online_fisher:
                    fisher = self._online_fisher[name].to(param.device)
                    optimal = self._online_params[name].to(param.device)
                    loss = loss + (fisher * (param - optimal) ** 2).sum()
        else:
            for task_data in self.tasks:
                for name, param in self._get_ref_params(agent):
                    if name in task_data["fisher"]:
                        fisher = task_data["fisher"][name].to(param.device)
                        optimal = task_data["params"][name].to(param.device)
                        loss = loss + (fisher * (param - optimal) ** 2).sum()

        return 0.5 * self.ewc_lambda * loss

    def save(self, path):
        save_data = {
            "tasks": self.tasks,
            "online": self.online,
            "gamma": self.gamma,
            "ewc_lambda": self.ewc_lambda,
            "online_fisher": self._online_fisher,
            "online_params": self._online_params,
            "task_count": self._task_count,
        }
        torch.save(save_data, path)

    def load(self, path):
        if not os.path.exists(path):
            return

        save_data = torch.load(path, map_location=self.device, weights_only=False)
        self.tasks = save_data.get("tasks", [])
        self.online = save_data.get("online", self.online)
        self.gamma = save_data.get("gamma", self.gamma)
        self.ewc_lambda = save_data.get("ewc_lambda", self.ewc_lambda)
        self._online_fisher = save_data.get("online_fisher", {})
        self._online_params = save_data.get("online_params", {})
        self._task_count = save_data.get("task_count", len(self.tasks))
