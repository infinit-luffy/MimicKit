"""
Gradient Projection Memory for Continual Learning.

Supports two projection methods:
- GPM (Gradient Projection Memory): Hard projection that completely removes
  gradient components in the protected subspace. P = U @ U^T.
- SGP (Scaled Gradient Projection): Scales the projection by accumulated
  singular value importance. Allows some learning in less important directions.
  proj = U @ diag(alpha) @ U^T.

Both methods use incremental SVD updates across tasks, matching the reference
implementations in ref_code/gpm.py and ref_code/sgp.py.
"""

import os
import torch
import torch.nn as nn


class ProjectionMemoryBank:
    """Manages gradient projection memory across continual learning tasks.

    Maintains per-layer SVD basis vectors incrementally updated after each task.
    For SGP, also tracks importance (alpha) per basis direction.

    Args:
        device: Torch device for storing basis vectors.
        threshold: Base variance threshold for selecting SVD dimensions (0-1).
        method: "gpm" for hard projection, "sgp" for scaled projection.
        threshold_inc: Threshold increment per task. Effective threshold is
                       threshold + task_id * threshold_inc.
        scale_coff: SGP importance scaling coefficient. Controls how softly
                    importance decays: larger values -> more uniform importance.
                    importance = ((c+1)*S) / (c*S + max(S)), where c = scale_coff.
    """

    def __init__(self, device, threshold=0.98, method="gpm",
                 threshold_inc=0.0, scale_coff=5):
        self.device = device
        self.threshold = threshold
        self.threshold_inc = threshold_inc
        self.scale_coff = scale_coff
        self.method = method

        # Per-layer basis (incrementally maintained)
        self.feature_list = []       # list[Tensor|None]: [feat_dim, rank] per layer
        self.importance_list = []    # list[Tensor|None]: [rank] per layer (SGP only)
        self.num_tasks = 0

        # Legacy storage for backward compat with old saves
        self.memory_bank = []

    def update_memory(self, actor_layers, obs_data, task_id, output_layer=None):
        """Extract features and incrementally update projection memory.

        Call this after training each task. For GPM, expands the protected
        subspace. For SGP, also tracks importance per basis direction.

        Args:
            actor_layers: nn.Sequential actor network.
            obs_data: Tensor [N, input_dim] of observations (with one-hot).
            task_id: Integer task ID (0-based, used for threshold increment).
            output_layer: Optional output module to also protect.
        """
        activations = self._collect_activations(actor_layers, obs_data, output_layer)
        effective_threshold = self.threshold + task_id * self.threshold_inc

        print("[CL:{}] Updating memory for task {} (threshold={:.4f})".format(
            self.method.upper(), task_id, effective_threshold))

        if self.method == "gpm":
            self._update_gpm(activations, effective_threshold)
        else:
            self._update_sgp(activations, effective_threshold)

        self.num_tasks += 1

    def _collect_activations(self, actor_layers, obs_data, output_layer=None):
        """Collect per-layer input activations via forward hooks.

        Returns list of Gram matrices G = R^T @ R (feat_dim x feat_dim) per
        Linear layer, along with trace(G) for variance tracking.
        For SGP incremental update, also returns the raw activation matrix.

        Returns:
            list[dict|None]: Per-layer {"G": Gram, "act": activation_matrix}.
        """
        actor_layers.eval()
        raw_activations = {}
        hooks = []
        layer_names = []

        for name, module in actor_layers.named_modules():
            if isinstance(module, nn.Linear):
                full_name = "actor." + name
                layer_names.append(full_name)

                def get_hook(ln):
                    def hook(mod, inp, out):
                        if ln not in raw_activations:
                            raw_activations[ln] = []
                        raw_activations[ln].append(inp[0].detach())
                    return hook

                hooks.append(module.register_forward_hook(get_hook(full_name)))

        if output_layer is not None:
            output_layer.eval()
            for name, module in output_layer.named_modules():
                if isinstance(module, nn.Linear):
                    full_name = "output." + name
                    layer_names.append(full_name)

                    def get_hook(ln):
                        def hook(mod, inp, out):
                            if ln not in raw_activations:
                                raw_activations[ln] = []
                            raw_activations[ln].append(inp[0].detach())
                        return hook

                    hooks.append(module.register_forward_hook(get_hook(full_name)))

        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(obs_data), batch_size):
                batch = obs_data[i:i + batch_size].to(
                    next(actor_layers.parameters()).device)
                h = actor_layers(batch)
                if output_layer is not None:
                    output_layer(h)

        for h in hooks:
            h.remove()

        result = []
        for name in layer_names:
            if name in raw_activations:
                R = torch.cat(raw_activations[name], dim=0)  # [N, feat_dim]
                act = R.t().contiguous().cpu()  # [feat_dim, N] on CPU for SVD
                result.append({"act": act, "name": name})
                print("[CL:proj] {}: [{}, {}] samples".format(
                    name, R.shape[1], R.shape[0]))
            else:
                print("[CL:proj] {}: NO activations!".format(name))
                result.append(None)

        return result

    def _update_gpm(self, activations, threshold):
        """Incremental GPM update (ref: ref_code/gpm.py get_GPM).

        First task: SVD on Gram matrix, keep top-r directions.
        Later tasks: SVD on residual Gram (I-P)G(I-P), expand basis.
        """
        if not self.feature_list:
            for i, data in enumerate(activations):
                if data is None:
                    self.feature_list.append(None)
                    continue
                act = data["act"]  # [feat_dim, N]
                U, S, _ = torch.linalg.svd(act, full_matrices=False)  # U: [feat_dim, min(feat_dim,N)]

                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / (sval_total + 1e-8)
                r = (torch.cumsum(sval_ratio, dim=0) < threshold).sum().item() + 1
                r = min(r, U.shape[1])

                self.feature_list.append(U[:, :r].to(self.device))
                print("[CL:GPM] Layer {}: r={}/{}".format(i, r, act.shape[0]))
        else:
            for i, data in enumerate(activations):
                if data is None or i >= len(self.feature_list) or self.feature_list[i] is None:
                    continue
                act = data["act"]  # [feat_dim, N]
                U_old = self.feature_list[i].to(act.device)

                U1, S1, _ = torch.linalg.svd(act, full_matrices=False)
                sval_total = (S1 ** 2).sum()

                # Residual activation: act_hat = (I - P) @ act
                act_hat = act - U_old @ (U_old.t() @ act)
                U, S, _ = torch.linalg.svd(act_hat, full_matrices=False)

                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / (sval_total + 1e-8)
                accumulated = ((sval_total - sval_hat) / (sval_total + 1e-8)).item()

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated < threshold:
                        accumulated += sval_ratio[ii].item()
                        r += 1
                    else:
                        break

                if r == 0:
                    print("[CL:GPM] Layer {}: skip (basis sufficient)".format(i))
                    continue

                Ui = torch.cat([U_old, U[:, :r]], dim=1)
                if Ui.shape[1] > Ui.shape[0]:
                    Ui = Ui[:, :Ui.shape[0]]
                # QR re-orthogonalizes in float32 to prevent numerical drift
                Q, _ = torch.linalg.qr(Ui)
                self.feature_list[i] = Q.to(self.device)

                print("[CL:GPM] Layer {}: +{} dims, total={}/{}".format(
                    i, r, self.feature_list[i].shape[1],
                    self.feature_list[i].shape[0]))

    def _update_sgp(self, activations, threshold):
        """Incremental SGP update with importance (ref: ref_code/sgp.py get_SGP).

        Importance formula: alpha = ((c+1)*S) / (c*S + max(S))
        Importance accumulates across tasks with clipping to [0, 1].
        """
        c = self.scale_coff

        if not self.feature_list:
            # First task
            for i, data in enumerate(activations):
                if data is None:
                    self.feature_list.append(None)
                    self.importance_list.append(None)
                    continue
                act = data["act"]  # [feat_dim, N]
                U, S, _ = torch.linalg.svd(act, full_matrices=False)

                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / (sval_total + 1e-8)
                r = (torch.cumsum(sval_ratio, dim=0) < threshold).sum().item() + 1
                r = min(r, U.shape[1])

                S_sel = S[:r]
                importance = ((c + 1) * S_sel) / (c * S_sel + S_sel.max() + 1e-8)

                self.feature_list.append(U[:, :r].to(self.device))
                self.importance_list.append(importance.to(self.device))
                print("[CL:SGP] Layer {}: r={}/{}, imp=[{:.4f},{:.4f}]".format(
                    i, r, act.shape[0],
                    importance.min().item(), importance.max().item()))
        else:
            # Subsequent tasks: incremental update
            for i, data in enumerate(activations):
                if (data is None or i >= len(self.feature_list)
                        or self.feature_list[i] is None):
                    continue
                act = data["act"]  # [feat_dim, N]
                U_old = self.feature_list[i].to(act.device)
                r_old = U_old.shape[1]

                _, S1, _ = torch.linalg.svd(act, full_matrices=False)
                sval_total = (S1 ** 2).sum()

                # --- Surrogate importance on old basis (ref: sgp.py Eq-4) ---
                # act_proj = U_old @ U_old^T @ act
                act_proj = U_old @ (U_old.t() @ act)  # [feat_dim, N]
                Uc, Sc, _ = torch.linalg.svd(act_proj, full_matrices=False)
                r_proj = min(r_old, (Sc > 1e-8).sum().item())
                r_proj = max(r_proj, 1)
                importance_new_on_old = torch.sqrt(
                    ((U_old.t() @ Uc[:, :r_proj]) ** 2) @ (Sc[:r_proj] ** 2)
                )

                # --- Residual for new directions ---
                act_hat = act - act_proj  # (I - P) @ act
                U, S_hat, _ = torch.linalg.svd(act_hat, full_matrices=False)

                sval_hat = (S_hat ** 2).sum()
                sval_ratio = (S_hat ** 2) / (sval_total + 1e-8)
                accumulated = ((sval_total - sval_hat) / (sval_total + 1e-8)).item()

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated < threshold:
                        accumulated += sval_ratio[ii].item()
                        r += 1
                    else:
                        break

                imp_old = self.importance_list[i].to(act.device)

                if r == 0:
                    # No new dimensions needed -- update importance only
                    importance = importance_new_on_old
                    importance = ((c + 1) * importance) / (
                        c * importance + importance.max() + 1e-8)
                    importance[:r_old] = torch.clamp(
                        importance[:r_old] + imp_old[:r_old], 0, 1)
                    self.importance_list[i] = importance.to(self.device)
                    print("[CL:SGP] Layer {}: skip expand, imp updated".format(i))
                else:
                    # Expand basis + update importance
                    importance = torch.cat([importance_new_on_old, S_hat[:r]])
                    importance = ((c + 1) * importance) / (
                        c * importance + importance.max() + 1e-8)
                    importance[:r_old] = torch.clamp(
                        importance[:r_old] + imp_old[:r_old], 0, 1)

                    # Preserve the old basis exactly so its per-column importance
                    # remains well-defined. Only orthogonalize the newly added block.
                    max_new_rank = max(0, U_old.shape[0] - r_old)
                    new_block = U[:, : min(r, max_new_rank)]
                    new_block, keep_mask = _orthonormalize_new_block(U_old, new_block)

                    if new_block.shape[1] == 0:
                        self.feature_list[i] = U_old.to(self.device)
                        self.importance_list[i] = importance[:r_old].to(self.device)
                        print("[CL:SGP] Layer {}: skip expand after re-orthogonalization".format(i))
                        continue

                    new_importance = importance[r_old:r_old + keep_mask.numel()][keep_mask]
                    Ui = torch.cat([U_old, new_block], dim=1)
                    importance = torch.cat([importance[:r_old], new_importance], dim=0).clamp(0.0, 1.0)

                    self.feature_list[i] = Ui.to(self.device)
                    self.importance_list[i] = importance.to(self.device)

                    print("[CL:SGP] Layer {}: +{} dims, total={}/{}, imp=[{:.4f},{:.4f}]".format(
                        i, new_block.shape[1], self.feature_list[i].shape[1],
                        self.feature_list[i].shape[0],
                        self.importance_list[i].min().item(),
                        self.importance_list[i].max().item()))

    def build_projection_matrices(self):
        """Build projection data from per-layer basis.

        For GPM: returns list of P matrices (P = U @ U^T).
        For SGP: returns list of dicts {"U": U, "alpha": importance}.

        Returns:
            list: Per-layer projection data, or empty list if no tasks stored.
        """
        if not self.feature_list:
            return []

        projection_data = []
        for i in range(len(self.feature_list)):
            U = self.feature_list[i]
            if U is None:
                projection_data.append(None)
                continue

            if self.method == "gpm":
                P = torch.mm(U, U.t())
                projection_data.append(P)
                print("[CL:GPM] Layer {}: P {}, rank={}/{}".format(
                    i, list(P.shape), U.shape[1], P.shape[0]))
            else:  # sgp
                imp = self.importance_list[i] if i < len(self.importance_list) else None
                if imp is not None:
                    projection_data.append({"U": U, "alpha": imp})
                    print("[CL:SGP] Layer {}: rank={}/{}, alpha=[{:.4f},{:.4f}]".format(
                        i, U.shape[1], U.shape[0],
                        imp.min().item(), imp.max().item()))
                else:
                    # Fallback to hard projection if no importance
                    P = torch.mm(U, U.t())
                    projection_data.append(P)

        return projection_data

    def build_anchor_weights(self, actor_layers, projection_data, output_layer=None):
        """Compute anchor weights for each Linear layer.

        For GPM: Anchor = W @ P.
        For SGP: Anchor = W @ U @ diag(alpha) @ U^T.

        Args:
            actor_layers: nn.Sequential actor network.
            projection_data: Per-layer projection data from build_projection_matrices().
            output_layer: Optional output module to also anchor.

        Returns:
            list[Tensor or None]: Per-layer anchor projections.
        """
        anchors = []
        kk = 0

        linear_modules = []
        for module in actor_layers.modules():
            if isinstance(module, nn.Linear):
                linear_modules.append(module)
        if output_layer is not None:
            for module in output_layer.modules():
                if isinstance(module, nn.Linear):
                    linear_modules.append(module)

        for module in linear_modules:
            if kk < len(projection_data) and projection_data[kk] is not None:
                data = projection_data[kk]
                sz = module.weight.data.size(0)
                flat_w = module.weight.data.view(sz, -1)
                anchor = _apply_projection(flat_w, data)
                anchors.append(anchor)
            else:
                anchors.append(None)
            kk += 1
        return anchors

    # ------------------------------------------------------------------
    # Legacy method (kept for backward compatibility)
    # ------------------------------------------------------------------

    def extract_features(self, actor_layers, obs_data, output_layer=None):
        """Legacy feature extraction. Prefer update_memory() instead."""
        activations = self._collect_activations(actor_layers, obs_data, output_layer)
        features = []
        for data in activations:
            if data is None:
                features.append(None)
                continue
            act = data["act"]
            U, S, _ = torch.linalg.svd(act, full_matrices=False)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / (sval_total + 1e-8)
            k = (torch.cumsum(sval_ratio, dim=0) < self.threshold).sum().item() + 1
            k = min(max(k, 1), U.shape[1])
            features.append({"U": U[:, :k], "S": S[:k]})
        return features

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        """Save memory state to disk."""
        save_data = {
            'method': self.method,
            'threshold': self.threshold,
            'threshold_inc': self.threshold_inc,
            'scale_coff': self.scale_coff,
            'num_tasks': self.num_tasks,
            'feature_list': self.feature_list,
            'importance_list': self.importance_list,
            'memory_bank': self.memory_bank,
        }
        torch.save(save_data, path)

    def load(self, path):
        """Load memory state from disk."""
        if not os.path.exists(path):
            return
        save_data = torch.load(path, map_location=self.device, weights_only=False)
        self.method = save_data.get('method', self.method)
        self.threshold = save_data.get('threshold', self.threshold)
        self.threshold_inc = save_data.get('threshold_inc', self.threshold_inc)
        self.scale_coff = save_data.get('scale_coff', self.scale_coff)
        self.num_tasks = save_data.get('num_tasks', 0)
        self.feature_list = save_data.get('feature_list', [])
        self.importance_list = save_data.get('importance_list', [])
        self.memory_bank = save_data.get('memory_bank', [])

        # Backward compat: rebuild feature_list from old memory_bank format
        if not self.feature_list and self.memory_bank:
            self._rebuild_from_memory_bank()

    def _rebuild_from_memory_bank(self):
        """Rebuild feature_list from legacy memory_bank format."""
        if not self.memory_bank:
            return
        num_layers = len(self.memory_bank[0])
        self.feature_list = []

        for layer_idx in range(num_layers):
            history_Us = []
            for task_feats in self.memory_bank:
                if layer_idx < len(task_feats) and task_feats[layer_idx] is not None:
                    feat = task_feats[layer_idx]
                    U = feat["U"] if isinstance(feat, dict) else feat
                    history_Us.append(U.to(self.device))

            if not history_Us:
                self.feature_list.append(None)
                continue

            U_concat = torch.cat(history_Us, dim=1)
            G = torch.mm(U_concat, U_concat.t())
            U_final, S, _ = torch.svd(G)
            significant = S > 1e-6
            self.feature_list.append(U_final[:, significant])

        self.num_tasks = len(self.memory_bank)
        print("[CL:proj] Rebuilt feature_list from memory_bank ({} tasks, {} layers)".format(
            self.num_tasks, len(self.feature_list)))


def _apply_projection(flat, data):
    """Apply projection to a flat [out_dim, in_dim] tensor.

    Returns the protected component flat @ M to subtract from gradients/weights.

    For GPM: M = U @ U^T.
    For SGP: M = U @ diag(alpha) @ U^T.
    """
    M = get_projection_matrix(data)
    if M is None:
        return torch.zeros_like(flat)
    return torch.mm(flat, M.to(flat.device))


def _orthonormalize_new_block(U_old, U_new, tol=1e-6):
    """Return a numerically cleaned new block while keeping the old basis fixed.

    Args:
        U_old: Existing orthonormal basis [feat_dim, r_old].
        U_new: Candidate expansion directions [feat_dim, r_new].
        tol: QR diagonal threshold used to drop numerically collapsed columns.

    Returns:
        tuple[Tensor, BoolTensor]: (orthonormalized_new_block, kept_column_mask).
    """
    if U_new.numel() == 0:
        return U_new, torch.zeros(U_new.shape[1], dtype=torch.bool, device=U_new.device)

    # Remove any drift back into the protected subspace before re-orthogonalizing.
    U_new = U_new - U_old @ (U_old.t() @ U_new)

    Q_new, R_new = torch.linalg.qr(U_new, mode='reduced')
    diag = torch.diagonal(R_new, 0).abs()
    keep = diag > tol

    if not torch.any(keep):
        return U_new[:, :0], keep

    return Q_new[:, keep], keep


def get_projection_matrix(data):
    """Convert projection data to a dense protection matrix M."""
    if data is None:
        return None
    if isinstance(data, dict):
        U = data["U"]
        alpha = data["alpha"].to(U.device)
        return torch.mm(U * alpha.unsqueeze(0), U.t())
    return data
