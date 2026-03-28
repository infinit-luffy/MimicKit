"""
Elastic Weight Consolidation (EWC) Memory for Continual Learning.

Computes and stores the diagonal Fisher Information Matrix and optimal
parameter values after each task. During training of new tasks, adds
a quadratic penalty to prevent drift from optimal parameters.

Supports both standard EWC (per-task Fisher) and online EWC (running average).
"""

import os
import torch


class EWCMemory:
    """EWC memory for continual learning protection.

    Args:
        device: Torch device.
        ewc_lambda: Regularization strength.
        online: If True, use online EWC with exponential decay.
        gamma: Decay factor for online EWC Fisher accumulation.
    """

    def __init__(self, device, ewc_lambda=1000.0, online=False, gamma=0.95):
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma

        # Standard EWC: list of {"params": {name: tensor}, "fisher": {name: tensor}}
        self.tasks = []

        # Online EWC: accumulated state
        self._online_fisher = {}
        self._online_params = {}

    def compute_fisher(self, agent, env, n_steps=20):
        """Estimate diagonal Fisher Information Matrix via policy rollouts.

        Uses batch-averaged log_prob gradients as Fisher approximation:
        F_i ≈ E[ (d log pi(a|s) / d theta_i)^2 ]

        Args:
            agent: Trained AMPCLAgent with _get_protected_params().
            env: MimicKit environment.
            n_steps: Number of rollout steps for Fisher estimation.

        Returns:
            dict: {param_name: fisher_diagonal} for protected parameters.
        """
        agent.eval()
        fisher = {}
        for name, param in agent._get_protected_params():
            fisher[name] = torch.zeros_like(param)

        obs, info = env.reset()
        n_batches = 0

        for _ in range(n_steps):
            with torch.no_grad():
                action, _ = agent._decide_action(obs, info)

            # Compute log_prob WITH gradients, sampling fresh actions from pi
            # (Fisher = E_{a~pi}[grad^2], must sample from current policy)
            norm_obs = agent._obs_norm.normalize(obs)
            motion_onehot = agent._motion_onehot_buf
            a_dist = agent._model.eval_actor(norm_obs, motion_onehot)
            with torch.no_grad():
                sampled_norm_a = a_dist.sample()
            log_prob = a_dist.log_prob(sampled_norm_a).mean()

            agent._model.zero_grad()
            log_prob.backward()

            for name, param in agent._get_protected_params():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            n_batches += 1

            with torch.no_grad():
                obs, _, _, info = env.step(action)

        for name in fisher:
            fisher[name] /= max(n_batches, 1)

        print("[CL:EWC] Fisher computed from {} batches, {} params".format(
            n_batches, len(fisher)))
        return fisher

    def register_task(self, agent, fisher):
        """Store current parameter values and Fisher after completing a task.

        Args:
            agent: Trained agent with _get_protected_params().
            fisher: Dict of Fisher diagonals from compute_fisher().
        """
        params = {}
        for name, param in agent._get_protected_params():
            params[name] = param.data.clone()

        if self.online:
            for name in fisher:
                if name in self._online_fisher:
                    self._online_fisher[name] = (self.gamma * self._online_fisher[name]
                                                  + fisher[name])
                else:
                    self._online_fisher[name] = fisher[name].clone()
            self._online_params = params
            print("[CL:EWC] Online Fisher updated (gamma={})".format(self.gamma))
        else:
            self.tasks.append({"params": params, "fisher": fisher})
            print("[CL:EWC] Task registered. Total tasks: {}".format(len(self.tasks)))

    def compute_ewc_loss(self, agent):
        """Compute the EWC regularization penalty.

        loss = (lambda/2) * sum_tasks sum_params F_i * (theta_i - theta*_i)^2

        Args:
            agent: Current agent with _get_protected_params().

        Returns:
            Scalar loss tensor.
        """
        loss = torch.tensor(0.0, device=self.device)

        if self.online:
            if not self._online_fisher:
                return loss
            for name, param in agent._get_protected_params():
                if name in self._online_fisher:
                    fisher = self._online_fisher[name].to(param.device)
                    optimal = self._online_params[name].to(param.device)
                    loss = loss + (fisher * (param - optimal) ** 2).sum()
        else:
            for task_data in self.tasks:
                for name, param in agent._get_protected_params():
                    if name in task_data["fisher"]:
                        fisher = task_data["fisher"][name].to(param.device)
                        optimal = task_data["params"][name].to(param.device)
                        loss = loss + (fisher * (param - optimal) ** 2).sum()

        return 0.5 * self.ewc_lambda * loss

    def save(self, path):
        """Save EWC memory to disk."""
        save_data = {
            'tasks': self.tasks,
            'online': self.online,
            'gamma': self.gamma,
            'ewc_lambda': self.ewc_lambda,
            'online_fisher': self._online_fisher,
            'online_params': self._online_params,
        }
        torch.save(save_data, path)

    def load(self, path):
        """Load EWC memory from disk."""
        if not os.path.exists(path):
            return
        save_data = torch.load(path, map_location=self.device, weights_only=False)
        self.tasks = save_data.get('tasks', [])
        self.online = save_data.get('online', self.online)
        self.gamma = save_data.get('gamma', self.gamma)
        self.ewc_lambda = save_data.get('ewc_lambda', self.ewc_lambda)
        self._online_fisher = save_data.get('online_fisher', {})
        self._online_params = save_data.get('online_params', {})
