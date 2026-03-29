"""
Reference-style GPM/SGP memory updates for continual learning.

This module keeps the update rules aligned with ref_code/gpm.py and
ref_code/sgp.py as closely as possible within the current AMPCL interface:
- no QR re-orthogonalization after concatenating old and new bases
- no importance transfer after basis rotation
- the protected subspace is represented exactly as the ref code stores it
"""

import torch

from learning.cl.projection_memory import ProjectionMemoryBank


class ReferenceProjectionMemoryBank(ProjectionMemoryBank):
    """Reference-style projection memory matching ref_code update rules."""

    def __init__(self, device, threshold=0.98, method="gpm",
                 threshold_inc=0.0, scale_coff=5):
        super().__init__(
            device=device,
            threshold=threshold,
            method=method,
            threshold_inc=threshold_inc,
            scale_coff=scale_coff,
        )

    def update_memory(self, actor_layers, obs_data, task_id, output_layer=None):
        activations = self._collect_activations(actor_layers, obs_data, output_layer)
        effective_threshold = self.threshold + task_id * self.threshold_inc

        print("[CL:{}_REF] Updating memory for task {} (threshold={:.4f})".format(
            self.method.upper(), task_id, effective_threshold))

        if self.method == "gpm":
            self._update_gpm(activations, effective_threshold)
        else:
            self._update_sgp(activations, effective_threshold)

        self.num_tasks += 1

    def _update_gpm(self, activations, threshold):
        if not self.feature_list:
            for i, data in enumerate(activations):
                if data is None:
                    self.feature_list.append(None)
                    continue

                act = data["act"]
                U, S, _ = torch.linalg.svd(act, full_matrices=False)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / (sval_total + 1e-8)
                r = (torch.cumsum(sval_ratio, dim=0) < threshold).sum().item() + 1
                r = min(r, U.shape[1])

                self.feature_list.append(U[:, :r].to(self.device))
                print("[CL:GPM_REF] Layer {}: r={}/{}".format(i, r, act.shape[0]))
        else:
            for i, data in enumerate(activations):
                if data is None or i >= len(self.feature_list) or self.feature_list[i] is None:
                    continue

                act = data["act"]
                U_old = self.feature_list[i].to(act.device)

                _, S1, _ = torch.linalg.svd(act, full_matrices=False)
                sval_total = (S1 ** 2).sum()

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
                    print("[CL:GPM_REF] Layer {}: skip (basis sufficient)".format(i))
                    continue

                Ui = torch.cat([U_old, U[:, :r]], dim=1)
                if Ui.shape[1] > Ui.shape[0]:
                    Ui = Ui[:, :Ui.shape[0]]

                self.feature_list[i] = Ui.to(self.device)
                print("[CL:GPM_REF] Layer {}: +{} dims, total={}/{}".format(
                    i, r, self.feature_list[i].shape[1], self.feature_list[i].shape[0]))

    def _update_sgp(self, activations, threshold):
        c = self.scale_coff

        if not self.feature_list:
            for i, data in enumerate(activations):
                if data is None:
                    self.feature_list.append(None)
                    self.importance_list.append(None)
                    continue

                act = data["act"]
                U, S, _ = torch.linalg.svd(act, full_matrices=False)

                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / (sval_total + 1e-8)
                r = (torch.cumsum(sval_ratio, dim=0) < threshold).sum().item() + 1
                r = min(r, U.shape[1])

                importance = ((c + 1) * S[:r]) / (c * S[:r] + S[:r].max() + 1e-8)
                self.feature_list.append(U[:, :r].to(self.device))
                self.importance_list.append(importance.to(self.device))
                print("[CL:SGP_REF] Layer {}: r={}/{}, imp=[{:.4f},{:.4f}]".format(
                    i, r, act.shape[0], importance.min().item(), importance.max().item()))
        else:
            for i, data in enumerate(activations):
                if (data is None or i >= len(self.feature_list)
                        or self.feature_list[i] is None):
                    continue

                act = data["act"]
                U_old = self.feature_list[i].to(act.device)
                r_old = U_old.shape[1]

                _, S1, _ = torch.linalg.svd(act, full_matrices=False)
                sval_total = (S1 ** 2).sum()

                act_proj = U_old @ (U_old.t() @ act)
                Uc, Sc, _ = torch.linalg.svd(act_proj, full_matrices=False)
                r_proj = min(r_old, Uc.shape[1])
                importance_new_on_old = torch.sqrt(
                    ((U_old.t() @ Uc[:, :r_proj]) ** 2) @ (Sc[:r_proj] ** 2)
                )

                act_hat = act - act_proj
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
                    importance = importance_new_on_old
                    importance = ((c + 1) * importance) / (
                        c * importance + importance.max() + 1e-8)
                    importance[:r_old] = torch.clamp(
                        importance[:r_old] + imp_old[:r_old], 0, 1)
                    self.importance_list[i] = importance.to(self.device)
                    print("[CL:SGP_REF] Layer {}: skip expand, imp updated".format(i))
                    continue

                importance = torch.cat([importance_new_on_old, S_hat[:r]])
                importance = ((c + 1) * importance) / (
                    c * importance + importance.max() + 1e-8)
                importance[:r_old] = torch.clamp(
                    importance[:r_old] + imp_old[:r_old], 0, 1)

                Ui = torch.cat([U_old, U[:, :r]], dim=1)
                if Ui.shape[1] > Ui.shape[0]:
                    Ui = Ui[:, :Ui.shape[0]]
                    importance = importance[:Ui.shape[0]]

                self.feature_list[i] = Ui.to(self.device)
                self.importance_list[i] = importance.to(self.device)
                print("[CL:SGP_REF] Layer {}: +{} dims, total={}/{}, imp=[{:.4f},{:.4f}]".format(
                    i, r, self.feature_list[i].shape[1], self.feature_list[i].shape[0],
                    self.importance_list[i].min().item(),
                    self.importance_list[i].max().item()))
