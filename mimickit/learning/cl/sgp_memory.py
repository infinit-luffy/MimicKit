"""
Subspace Gradient Projection (SGP) Memory Bank.

Ported from unitree-rl-cl: legged_gym/scripts/train_cl.py
Manages per-task feature subspaces extracted via SVD from layer activations.
Used to build projection matrices that protect learned task representations
during continual learning.
"""

import os
import torch
import torch.nn as nn


class SGPMemoryBank:
    """Manages SGP feature memory across continual learning tasks.

    For each completed task, stores per-layer SVD basis vectors (U matrices)
    of the actor network's activations. Before training a new task, builds
    projection matrices P = U @ U^T that protect previously learned subspaces.
    """

    def __init__(self, device, threshold=0.98):
        self.device = device
        self.threshold = threshold
        self.memory_bank = []  # list[list[Tensor]]: per-task, per-layer U matrices

    def collect_observations(self, env, agent, n_steps=20):
        """Run the trained policy and collect observations.

        Args:
            env: The MimicKit environment.
            agent: The trained AMPCLAgent.
            n_steps: Number of rollout steps.

        Returns:
            Tensor of shape [n_steps * num_envs, obs_dim].
        """
        agent.eval()
        obs, info = env.reset()
        obs_list = []

        with torch.no_grad():
            for _ in range(n_steps):
                action, action_info = agent._decide_action(obs, info)
                obs, r, done, info = env.step(action)
                obs_list.append(obs.clone())

        return torch.cat(obs_list, dim=0)

    def extract_features(self, actor_layers, obs_data):
        """Extract SVD-based feature subspaces from actor layer activations.

        Hooks into each nn.Linear layer in actor_layers, forwards obs_data,
        and computes SVD on the input activations to get basis vectors.

        Args:
            actor_layers: nn.Sequential actor network.
            obs_data: Tensor [N, input_dim] of collected observations
                      (should include motion one-hot already concatenated).

        Returns:
            list[Tensor]: Per-layer U basis matrices.
        """
        actor_layers.eval()
        activations = {}
        hooks = []

        # Find all Linear layers and register hooks to capture their inputs
        linear_names = []
        for name, module in actor_layers.named_modules():
            if isinstance(module, nn.Linear):
                linear_names.append(name)

                def get_hook(layer_name):
                    def hook(mod, inp, out):
                        if layer_name not in activations:
                            activations[layer_name] = []
                        activations[layer_name].append(inp[0].detach())
                    return hook

                h = module.register_forward_hook(get_hook(name))
                hooks.append(h)

        # Forward in batches
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(obs_data), batch_size):
                batch = obs_data[i:i + batch_size].to(next(actor_layers.parameters()).device)
                actor_layers(batch)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute SVD per layer
        current_task_features = []
        for name in linear_names:
            if name not in activations:
                current_task_features.append(None)
                continue

            R = torch.cat(activations[name], dim=0)
            G = torch.mm(R.t(), R)
            U, S, V = torch.svd(G)

            # Select top-k dimensions by variance threshold
            total_variance = torch.cumsum(S, dim=0)
            variance_ratio = total_variance / (torch.sum(S) + 1e-8)
            k = (variance_ratio < self.threshold).sum().item()
            k = max(k, 1)

            basis_U = U[:, :k]
            current_task_features.append(basis_U)

        return current_task_features

    def build_projection_matrices(self):
        """Build projection matrices from all stored task features.

        For each layer, concatenates all historical U matrices, re-orthogonalizes,
        and builds P = U_final @ U_final^T.

        Returns:
            list[Tensor or None]: Per-layer projection matrices.
        """
        if len(self.memory_bank) == 0:
            return []

        num_layers = len(self.memory_bank[0])
        projection_matrices = []

        for layer_idx in range(num_layers):
            # Collect U matrices from all tasks for this layer
            history_Us = []
            for task_feats in self.memory_bank:
                if layer_idx < len(task_feats) and task_feats[layer_idx] is not None:
                    history_Us.append(task_feats[layer_idx].to(self.device))

            if len(history_Us) == 0:
                projection_matrices.append(None)
                continue

            # Concatenate and re-orthogonalize
            U_concat = torch.cat(history_Us, dim=1)
            # SVD to get orthogonal basis of the union subspace
            U_final, S, _ = torch.svd(torch.mm(U_concat, U_concat.t()))

            # Keep dimensions with significant singular values
            significant = S > 1e-6
            U_final = U_final[:, significant]

            # Build projection matrix P = U @ U^T
            P = torch.mm(U_final, U_final.t())
            projection_matrices.append(P)

        return projection_matrices

    def build_anchor_weights(self, actor_layers, projection_matrices):
        """Compute anchor weights for each Linear layer.

        Anchor = W @ P, representing the current weight's projection onto
        the protected subspace. Used for post-reinit correction.

        Args:
            actor_layers: nn.Sequential actor network.
            projection_matrices: list of P matrices from build_projection_matrices().

        Returns:
            list[Tensor or None]: Per-layer anchor projections.
        """
        anchors = []
        kk = 0
        for module in actor_layers.modules():
            if isinstance(module, nn.Linear):
                if kk < len(projection_matrices) and projection_matrices[kk] is not None:
                    P = projection_matrices[kk].to(module.weight.device)
                    sz = module.weight.data.size(0)
                    flat_w = module.weight.data.view(sz, -1)
                    anchor = torch.mm(flat_w, P)
                    anchors.append(anchor)
                else:
                    anchors.append(None)
                kk += 1
        return anchors

    def save(self, path):
        """Save memory bank to disk."""
        save_data = {
            'memory_bank': self.memory_bank,
            'threshold': self.threshold,
        }
        torch.save(save_data, path)

    def load(self, path):
        """Load memory bank from disk."""
        if not os.path.exists(path):
            return
        save_data = torch.load(path, map_location=self.device, weights_only=False)
        self.memory_bank = save_data['memory_bank']
        self.threshold = save_data.get('threshold', self.threshold)
