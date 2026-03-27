import torch

import util.mp_util as mp_util

class MPOptimizer():
    CHECK_SYNC_STEPS = 1000

    def __init__(self, config, param_list):
        self._param_list = param_list
        self._grad_clip = float(config.get("grad_clip", 0.0))
        self._optimizer = self._build_optimizer(config, param_list)
        self._steps = 0
        
        if (mp_util.enable_mp()):
            self._param_buffer = self._build_param_buffer()

        self.sync()
        return
    
    def step(self, loss):
        self._optimizer.zero_grad()
        loss.backward()

        if (mp_util.enable_mp()):
            self._aggregate_mp_grads()

        if (self._enable_grad_clip()):
            self._clip_grads(self._grad_clip)

        self._optimizer.step()

        if (mp_util.enable_mp() and (self.get_steps() % self.CHECK_SYNC_STEPS == 0)):
            assert(self._check_synced()), "Network parameters desynchronized"

        self._steps += 1
        return

    def step_with_grad_hook(self, loss, pre_step_fn=None):
        """Like step(), but calls pre_step_fn after backward (before optimizer step).

        Used by continual learning to inject SGP gradient projection between
        loss.backward() and optimizer.step().

        Args:
            loss: The loss tensor to backpropagate.
            pre_step_fn: Optional callable invoked after backward + grad aggregation,
                         before grad clipping and optimizer step.
        """
        self._optimizer.zero_grad()
        loss.backward()

        if (mp_util.enable_mp()):
            self._aggregate_mp_grads()

        if (pre_step_fn is not None):
            pre_step_fn()

        if (self._enable_grad_clip()):
            self._clip_grads(self._grad_clip)

        self._optimizer.step()

        if (mp_util.enable_mp() and (self.get_steps() % self.CHECK_SYNC_STEPS == 0)):
            assert(self._check_synced()), "Network parameters desynchronized"

        self._steps += 1
        return

    def get_steps(self):
        return self._steps

    def sync(self):
        with torch.no_grad():
            for param in self._param_list:
                global_param = mp_util.broadcast(param)
                param.copy_(global_param)
        return

    def _build_optimizer(self, config, param_list):
        lr = float(config["learning_rate"])
        weight_decay = float(config.get("weight_decay", 0.0))
        optimizer_type = config["type"]

        if (optimizer_type == "SGD"):
            optimizer = torch.optim.SGD(param_list, lr, momentum=0.0, weight_decay=weight_decay)
        elif optimizer_type in ("Adam", "SGP_Adam", "Projection_Adam"):
            optimizer = torch.optim.AdamW(param_list, lr, weight_decay=weight_decay)
        else:
            assert(False), "Unsupported optimizer type: " + optimizer_type
        return optimizer
    
    def _build_param_buffer(self):
        buffer = torch.nn.utils.parameters_to_vector(self._param_list).clone().detach()
        return buffer
    
    def _check_synced(self):
        synced = True
        for param in self._param_list:
            global_param = mp_util.broadcast(param)
            param_synced = torch.equal(param, global_param)
            if (not param_synced):
                synced = False
        
        device = self._param_list[0].device
        buffer = torch.tensor([synced], dtype=torch.int, device=device)
        mp_util.reduce_min(buffer)
        synced = buffer.item() != 0

        return synced

    def _aggregate_mp_grads(self):
        grad_list = [p.grad for p in self._param_list]
        self._param_buffer[:] = torch.nn.utils.parameters_to_vector(grad_list)
        mp_util.reduce_inplace_mean(self._param_buffer)
        torch.nn.utils.vector_to_parameters(self._param_buffer, grad_list)
        return
    
    def _enable_grad_clip(self):
        return self._grad_clip > 0.0
    
    def _clip_grads(self, max_norm):
        torch.nn.utils.clip_grad_norm_(self._param_list, max_norm)
        return


class ProjectionAdamOptimizer():
    """Adam optimizer with double gradient projection for CL.

    For protected parameters, projects gradients both before and after Adam's
    momentum/variance update. The second projection corrects direction drift
    caused by Adam's per-element adaptive scaling (m / sqrt(v)), which would
    otherwise rotate the update back into the protected subspace.

    Supports both GPM (P matrix) and SGP (U + alpha dict) projection formats.
    When no projection data is set (task 0), behaves as standard Adam.
    """
    CHECK_SYNC_STEPS = 1000

    def __init__(self, config, param_list):
        self._param_list = param_list
        self._lr = float(config["learning_rate"])
        self._beta1 = float(config.get("beta1", 0.9))
        self._beta2 = float(config.get("beta2", 0.999))
        self._eps = float(config.get("eps", 1e-8))
        self._max_grad_norm = float(config.get("projection_max_grad_norm", 50.0))
        self._steps = 0
        self._log_interval = 100

        # Per-parameter Adam state
        self._m = [torch.zeros_like(p) for p in param_list]
        self._v = [torch.zeros_like(p) for p in param_list]
        self._t = [1 for _ in param_list]

        # Projection data (set when new task starts)
        self._projection_data = []

        if (mp_util.enable_mp()):
            self._param_buffer = self._build_param_buffer()

        self.sync()
        return

    def set_projection_data(self, projection_data):
        self._projection_data = projection_data if projection_data else []

    def _project(self, flat, data):
        """Apply projection to flat [out_dim, in_dim] tensor.

        Handles both GPM (P matrix) and SGP ({"U", "alpha"} dict) formats.
        Returns the projected component (to be subtracted from flat).
        """
        if isinstance(data, dict):
            U = data["U"].to(flat.device)
            alpha = data["alpha"].to(flat.device)
            proj = torch.mm(flat, U)
            proj = proj * alpha.unsqueeze(0)
            return torch.mm(proj, U.t())
        else:
            P = data.to(flat.device)
            return torch.mm(flat, P)

    def step(self, loss):
        for p in self._param_list:
            if p.grad is not None:
                p.grad.zero_()

        loss.backward()

        if (mp_util.enable_mp()):
            self._aggregate_mp_grads()

        do_log = (self._steps % self._log_interval == 0)

        for k, param in enumerate(self._param_list):
            if param.grad is None:
                self._t[k] += 1
                continue

            sz = param.grad.data.size(0)

            # Check if this param has projection data
            has_proj = (k < len(self._projection_data)
                        and self._projection_data[k] is not None
                        and param.dim() > 1)

            # 1st projection: project gradient before Adam update
            if has_proj:
                flat_grad = param.grad.data.view(sz, -1)
                proj = self._project(flat_grad, self._projection_data[k])
                param.grad.data = param.grad.data - proj.view(param.grad.shape)

            # Adam momentum / variance update
            self._m[k] = self._beta1 * self._m[k] + (1 - self._beta1) * param.grad.data
            self._v[k] = self._beta2 * self._v[k] + (1 - self._beta2) * param.grad.data ** 2

            m_hat = self._m[k] / (1 - self._beta1 ** self._t[k])
            v_hat = self._v[k] / (1 - self._beta2 ** self._t[k])
            grad_mod = m_hat / (torch.sqrt(v_hat) + self._eps)

            # 2nd projection: correct Adam's adaptive-scaling drift
            if has_proj:
                grad_before = grad_mod.norm().item()

                flat_mod = grad_mod.view(sz, -1)
                proj2 = self._project(flat_mod, self._projection_data[k])
                grad_mod = grad_mod - proj2.view(param.size())

                mod_norm = grad_mod.norm().item()
                if mod_norm > self._max_grad_norm:
                    grad_mod.mul_(self._max_grad_norm / (mod_norm + 1e-8))

                if do_log:
                    leak = proj2.norm().item()
                    print("[CL:ProjAdam] layer={} adam_norm={:.4f} leak={:.4f} final={:.4f}".format(
                        k, grad_before, leak, mod_norm))

            param.data = param.data - self._lr * grad_mod
            self._t[k] += 1

        if (mp_util.enable_mp() and (self._steps % self.CHECK_SYNC_STEPS == 0)):
            assert(self._check_synced()), "Network parameters desynchronized"

        self._steps += 1
        return

    def step_with_grad_hook(self, loss, pre_step_fn=None):
        """Interface-compatible with MPOptimizer. Ignores pre_step_fn."""
        self.step(loss)

    def get_steps(self):
        return self._steps

    def sync(self):
        with torch.no_grad():
            for param in self._param_list:
                global_param = mp_util.broadcast(param)
                param.copy_(global_param)
        return

    def _build_param_buffer(self):
        buffer = torch.nn.utils.parameters_to_vector(self._param_list).clone().detach()
        return buffer

    def _check_synced(self):
        synced = True
        for param in self._param_list:
            global_param = mp_util.broadcast(param)
            param_synced = torch.equal(param, global_param)
            if (not param_synced):
                synced = False

        device = self._param_list[0].device
        buffer = torch.tensor([synced], dtype=torch.int, device=device)
        mp_util.reduce_min(buffer)
        synced = buffer.item() != 0
        return synced

    def _aggregate_mp_grads(self):
        grad_list = [p.grad for p in self._param_list]
        self._param_buffer[:] = torch.nn.utils.parameters_to_vector(grad_list)
        mp_util.reduce_inplace_mean(self._param_buffer)
        torch.nn.utils.vector_to_parameters(self._param_buffer, grad_list)
        return


# Backward-compatible alias
SGPAdamOptimizer = ProjectionAdamOptimizer