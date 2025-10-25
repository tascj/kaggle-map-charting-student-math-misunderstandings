# Gradient-Release Adam
# Dose not support gradient accumulation


import torch
from torch.optim.optimizer import Optimizer

from offload_adam.kernels import (
    adam_step_fp32_master,
    adam_step_fp31_master,
    adam_step_fp32_master_custom_rounding,
)
from .kernels.stochastic_rounding import adam_step_stochastic_rounding


class Adam(Optimizer):
    configs = {
        "stochastic_rounding": {
            "step": adam_step_stochastic_rounding,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
            },
        },
        "fp32_master": {
            "step": adam_step_fp32_master,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "master_params": torch.float32,
            },
        },
        "fp31_master": {
            "step": adam_step_fp31_master,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "rounding_error": torch.int16,
            },
        },
        "fp32_master_custom_rounding": {
            "step": adam_step_fp32_master_custom_rounding,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "rounding_error": torch.int16,
            },
        },
    }

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        mode="stochastic_rounding",
        decoupled_weight_decay=False,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

        assert mode in self.configs, f"Invalid mode: {mode}"
        self.mode = mode
        self.config = self.configs[self.mode]
        self.step_fn = self.config["step"]
        self.decoupled_weight_decay = decoupled_weight_decay

        for group in self.param_groups:
            for param in group["params"]:
                param.register_post_accumulate_grad_hook(
                    self._create_post_accumulate_grad_hook(param, group)
                )

    def _create_post_accumulate_grad_hook(self, param, group):
        @torch.no_grad()
        def optimizer_step_hook(*unused):
            if param.grad is None:
                return
            grad = param.grad
            param.grad = None
            if grad.is_sparse:
                raise RuntimeError("Sparse gradient is not supported")

            state = self.state[param]

            # State initialization
            if len(state) == 0:
                state["step"] = 0
                for state_name, dtype in self.config["states"].items():
                    state[state_name] = torch.zeros_like(param, dtype=dtype)
                if "master_params" in state:
                    state["master_params"].copy_(param.data)
            state["step"] += 1

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            step = state["step"]

            device_states = {name: state[name] for name in self.config["states"]}
            device_states["grad"] = grad
            self.step_fn(
                param.data,
                device_states,
                lr,
                weight_decay,
                beta1,
                beta2,
                eps,
                step,
                decoupled_weight_decay=self.decoupled_weight_decay,
            )

        return optimizer_step_hook

    @torch.no_grad()
    def step(self, closure=None):
        """No-op."""
        return

    def zero_grad(self, set_to_none: bool = False):
        """No-op."""
        return
