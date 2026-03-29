import numpy as np
import torch
import torch.nn as nn


class OneHotInjectNet(nn.Module):
    """FC network that re-injects the one-hot suffix before each hidden layer.

    Input x = [obs_features..., one_hot] where the last onehot_dim elements
    are the task one-hot. The one-hot is re-concatenated before every hidden
    Linear layer (after the first) so GPM/SGP can distinguish tasks at every
    layer, not just layer 0.

    For a 2-layer net with layer_sizes=[1024, 512] and onehot_dim=20:
      x (160) → Linear(160→1024) → Act → cat([h1024, oh20]) →
                Linear(1044→512) → Act → output (512)
    """

    def __init__(self, layer_pairs, onehot_dim):
        """
        Args:
            layer_pairs: list of (nn.Linear, activation_module) tuples,
                         one per hidden layer.
            onehot_dim: number of trailing one-hot dims in the input.
        """
        super().__init__()
        self.onehot_dim = onehot_dim
        # Flatten into ModuleList for proper parameter registration.
        all_mods = []
        for lin, act in layer_pairs:
            all_mods.extend([lin, act])
        self.layers = nn.ModuleList(all_mods)
        self._n_pairs = len(layer_pairs)

    def forward(self, x):
        one_hot = x[..., -self.onehot_dim:]
        h = x
        for i in range(self._n_pairs):
            lin = self.layers[2 * i]
            act = self.layers[2 * i + 1]
            h = act(lin(h))
            # After each hidden activation except the last, re-inject one-hot.
            if i < self._n_pairs - 1:
                h = torch.cat([h, one_hot], dim=-1)
        return h


def build_net(input_dict, activation):
    layer_sizes = [1024, 512]

    # Detect one-hot dimension for mid-network injection.
    onehot_dim = 0
    if "motion_onehot" in input_dict:
        onehot_dim = int(np.prod(input_dict["motion_onehot"].shape))

    input_dim = int(np.sum([np.prod(curr_input.shape)
                            for curr_input in input_dict.values()]))

    layer_pairs = []
    in_size = input_dim
    for j, out_size in enumerate(layer_sizes):
        curr_layer = torch.nn.Linear(in_size, out_size)
        torch.nn.init.zeros_(curr_layer.bias)
        layer_pairs.append((curr_layer, activation()))
        # Next layer's input = this layer's output + one-hot (re-injected),
        # except after the last hidden layer.
        if j < len(layer_sizes) - 1 and onehot_dim > 0:
            in_size = out_size + onehot_dim
        else:
            in_size = out_size

    if onehot_dim > 0:
        net = OneHotInjectNet(layer_pairs, onehot_dim)
    else:
        layers = []
        for lin, act in layer_pairs:
            layers.extend([lin, act])
        net = torch.nn.Sequential(*layers)

    info = dict()
    return net, info
