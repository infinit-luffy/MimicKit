"""
Continual Backpropagation (CBP) Linear Module.

Ported from unitree-rl-cl: rsl_rl/modules/my_cl_modules/cbp_linear.py
Adapted for MimicKit's FC-only architecture.

CBP selectively reinitializes low-utility neurons to maintain network plasticity
during continual learning, while GPM-based energy ratio checks protect neurons
important to previously learned tasks.
"""

import torch
from torch import nn
from math import sqrt


def log_features(m, i, o):
    """Forward hook that tracks running mean of input activations."""
    with torch.no_grad():
        x = i[0].mean(dim=0)
        if m.decay_rate == 0:
            m.features = x
        else:
            if m.features is None:
                m.features = (1 - m.decay_rate) * x
            else:
                m.features = m.features * m.decay_rate + (1 - m.decay_rate) * x


def get_layer_bound(layer, init, gain):
    if isinstance(layer, nn.Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


class CBPLinear(nn.Module):
    """Continual Backpropagation wrapper for a pair of consecutive Linear layers.

    Tracks neuron utility in `in_layer` and selectively reinitializes low-utility
    neurons while protecting neurons important to previous tasks via GPM energy
    ratio checks.

    Args:
        in_layer: The Linear layer whose output neurons are tracked.
        out_layer: The next Linear layer (outgoing weights zeroed on reinit).
        replacement_rate: Fraction of eligible neurons replaced per step.
        maturity_threshold: Minimum age before a neuron is eligible for replacement.
        init: Weight initialization method ('kaiming', 'xavier', 'lecun', 'default').
        act_type: Activation function type (for gain calculation).
        util_type: Utility metric ('contribution' or 'weight').
        decay_rate: EMA decay for feature tracking.
        util_threshold: Utility threshold (unused in current selection logic).
        max_replace_per_step: Hard cap on replacements per step.
        accumulate: Whether to accumulate fractional replacement counts.
    """

    def __init__(
            self,
            in_layer: nn.Linear,
            out_layer: nn.Linear,
            replacement_rate=1e-4,
            maturity_threshold=1000,
            init='kaiming',
            act_type='relu',
            util_type='contribution',
            decay_rate=0.99,
            util_threshold=1e-4,
            max_replace_per_step=None,
            accumulate=True,
    ):
        super().__init__()
        self.in_layer = in_layer
        self.out_layer = out_layer

        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        self.decay_rate = decay_rate
        self.util_threshold = util_threshold
        self.max_replace_per_step = max_replace_per_step
        self.features = None
        self.accumulate = accumulate
        self.task_id = 0

        self.register_forward_hook(log_features)

        device = in_layer.weight.device
        num_features = in_layer.out_features

        self.util = nn.Parameter(torch.zeros(num_features, device=device), requires_grad=False)
        self.bias_corrected_util = nn.Parameter(torch.zeros(num_features, device=device), requires_grad=False)
        self.ages = nn.Parameter(torch.zeros(num_features, device=device), requires_grad=False)
        self.accumulated_num_features_to_replace = nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
        self.mean_feature_act = nn.Parameter(torch.zeros(num_features, device=device), requires_grad=False)

        self.bound = get_layer_bound(layer=in_layer, init=init,
                                     gain=nn.init.calculate_gain(nonlinearity=act_type))
        self.total_replacements = 0
        self.last_replacements = 0
        self.reset_mask = None

    def forward(self, _input):
        return _input

    def set_task_id(self, task_id):
        """Reset utility tracking for a new task."""
        self.task_id = task_id
        device = self.in_layer.weight.device
        num_features = self.in_layer.out_features
        self.util.data.zero_()
        self.bias_corrected_util.data.zero_()
        self.ages.data.zero_()
        self.accumulated_num_features_to_replace.data.zero_()
        self.mean_feature_act.data.zero_()
        self.features = None
        self.reset_mask = None

    def get_features_to_reinit(self, feature_mat=None, next_feature_mat=None, gpm_ratio_threshold=0.02):
        """Select low-utility neurons for reinitialization, with GPM protection.

        Args:
            feature_mat: Projection matrix for input-side GPM protection.
            next_feature_mat: Feature matrix for output-side GPM protection.
            gpm_ratio_threshold: Energy ratio threshold for GPM safety check.

        Returns:
            Tensor of neuron indices to replace.
        """
        device = self.util.device
        features_to_replace = torch.empty(0, dtype=torch.long, device=device)
        self.ages += 1

        # 1. Maturity check
        eligible_feature_indices = torch.where(self.ages > self.maturity_threshold)[0]
        if eligible_feature_indices.shape[0] == 0:
            return features_to_replace

        # 2. Compute utility
        output_weight_mag = self.out_layer.weight.data.abs().mean(dim=0)

        bias_correction = 1 - self.decay_rate ** self.ages
        self.mean_feature_act *= self.decay_rate
        # self.features is already [num_features] (mean over batch in hook)
        feat_vals = self.features if self.features.dim() == 1 else self.features.mean(dim=0)
        self.mean_feature_act += (1 - self.decay_rate) * feat_vals

        if self.util_type == 'weight':
            self.util.data = output_weight_mag
        elif self.util_type == 'contribution':
            dims = [i for i in range(self.features.ndim - 1)]
            self.util.data = output_weight_mag * self.features.abs().mean(dim=dims)
        else:
            self.util.data = torch.zeros_like(output_weight_mag)

        self.bias_corrected_util.data = self.util.data / bias_correction

        # 3. Determine target replacement count
        num_new_features_to_replace = self.replacement_rate * eligible_feature_indices.shape[0]
        self.accumulated_num_features_to_replace += num_new_features_to_replace

        if self.accumulate:
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
            self.accumulated_num_features_to_replace -= num_new_features_to_replace
        else:
            if num_new_features_to_replace < 1:
                if torch.rand(1) <= num_new_features_to_replace:
                    num_new_features_to_replace = 1
            num_new_features_to_replace = int(num_new_features_to_replace)

        if num_new_features_to_replace == 0:
            return features_to_replace

        # 4. Select candidates by lowest utility
        num_candidates = min(len(eligible_feature_indices), num_new_features_to_replace * 2)
        candidate_indices_util = torch.topk(
            -self.bias_corrected_util.data[eligible_feature_indices], num_candidates
        )[1]
        candidates_from_util = eligible_feature_indices[candidate_indices_util]

        # 5. GPM protection (for task_id > 0)
        if self.task_id != 0:
            final_safe_set = set(candidates_from_util.tolist())

            # Input-side check
            if feature_mat is not None:
                W_flat = self.in_layer.weight.data.view(self.in_layer.weight.data.size(0), -1)
                w_norms = torch.norm(W_flat, p=2, dim=1) + 1e-8
                proj_energy = torch.norm(torch.mm(W_flat, feature_mat), p=2, dim=1)
                energy_ratios = proj_energy / w_norms
                input_safe_mask = energy_ratios < gpm_ratio_threshold
                safe_indices_input = torch.where(input_safe_mask)[0]
                final_safe_set = final_safe_set.intersection(set(safe_indices_input.tolist()))

            # Output-side check
            if next_feature_mat is not None:
                out_proj_norms = torch.norm(next_feature_mat, p=2, dim=1)
                max_norm = out_proj_norms.max() + 1e-8
                out_ratios = out_proj_norms / max_norm
                output_safe_mask = out_ratios < (gpm_ratio_threshold / 2)
                safe_indices_output = torch.where(output_safe_mask)[0]
                final_safe_set = final_safe_set.intersection(set(safe_indices_output.tolist()))

            cand = torch.tensor(list(final_safe_set), device=device, dtype=torch.long)
        else:
            cand = candidates_from_util[:num_new_features_to_replace]

        # 6. Final count control
        actual_limit = num_new_features_to_replace
        if self.max_replace_per_step is not None:
            actual_limit = min(actual_limit, self.max_replace_per_step)

        if cand.numel() > actual_limit:
            ksel = torch.topk(-self.bias_corrected_util.data[cand], actual_limit)[1]
            cand = cand[ksel]

        features_to_replace = cand

        # Update reset mask
        num_features = self.util.shape[0]
        self.reset_mask = torch.zeros(num_features, device=device)
        self.reset_mask[features_to_replace] = 1.0

        return features_to_replace

    def reinit_features(self, features_to_replace):
        """Reinitialize selected neurons with orthogonal initialization."""
        with torch.no_grad():
            num_features_to_replace = features_to_replace.shape[0]
            if num_features_to_replace == 0:
                return

            # Orthogonal init for input weights
            temp_weight = torch.empty(
                num_features_to_replace, self.in_layer.in_features,
                device=self.util.device
            )
            torch.nn.init.orthogonal_(temp_weight, gain=1.0)
            self.in_layer.weight.data[features_to_replace, :] = temp_weight

            if isinstance(self.in_layer.bias, torch.Tensor):
                self.in_layer.bias.data[features_to_replace] = 0.0

            # Correct output bias and zero outgoing weights
            if isinstance(self.out_layer.bias, torch.Tensor):
                self.out_layer.bias.data += (
                    self.out_layer.weight.data[:, features_to_replace]
                    * self.mean_feature_act[features_to_replace]
                    / (1 - self.decay_rate ** self.ages[features_to_replace])
                ).sum(dim=1)

            self.out_layer.weight.data[:, features_to_replace] = 0
            self.ages[features_to_replace] = 0
            self.total_replacements += int(num_features_to_replace)
            self.last_replacements = int(num_features_to_replace)

    def reinit(self, feature_mat=None, next_feature_mat=None):
        """Perform selective reinitialization. Returns indices of replaced features."""
        features_to_replace = self.get_features_to_reinit(feature_mat, next_feature_mat)
        self.reinit_features(features_to_replace)
        return features_to_replace

    def update_optim_params_adam(self, features_to_replace, optimizer):
        """Reset Adam optimizer state for reinitialized neurons.

        Args:
            features_to_replace: Tensor of neuron indices that were reinitialized.
            optimizer: The torch.optim.Adam/AdamW optimizer instance.
        """
        if features_to_replace.numel() == 0:
            return

        # Reset input layer optimizer state
        if self.in_layer.weight in optimizer.state:
            state = optimizer.state[self.in_layer.weight]
            if 'exp_avg' in state:
                state['exp_avg'][features_to_replace, :] = 0.0
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'][features_to_replace, :] = 0.0

        if self.in_layer.bias is not None and self.in_layer.bias in optimizer.state:
            state = optimizer.state[self.in_layer.bias]
            if 'exp_avg' in state:
                state['exp_avg'][features_to_replace] = 0.0
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'][features_to_replace] = 0.0

        # Reset output layer optimizer state (columns)
        if self.out_layer.weight in optimizer.state:
            state = optimizer.state[self.out_layer.weight]
            if 'exp_avg' in state:
                state['exp_avg'][:, features_to_replace] = 0.0
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'][:, features_to_replace] = 0.0
