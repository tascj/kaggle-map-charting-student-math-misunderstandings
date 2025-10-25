# Gradient-Release OffloadAdam
# Dose not support gradient accumulation

import torch
from torch.optim.optimizer import Optimizer

from offload_adam.pin_memory import PinnedMemoryManager
from offload_adam.kernels import (
    adam_step_fp32_master,
    adam_step_fp31_master,
    adam_step_fp32_master_custom_rounding,
)
from .kernels.stochastic_rounding import adam_step_stochastic_rounding


def get_leaf_modules_with_params(module):
    """Recursively collect leaf modules with parameters from a PyTorch model."""
    leaf_modules = []
    for child in module.children():
        if list(child.children()):  # If the module has children, recurse
            leaf_modules.extend(get_leaf_modules_with_params(child))
        elif any(
            p.requires_grad for p in child.parameters(recurse=False)
        ):  # Check if it has parameters
            leaf_modules.append(child)
    return leaf_modules


class OffloadAdam(Optimizer):
    """Adam optimizer that offloads gradients and optimizer states to host memory.

    Args:
        model (nn.Module): Model containing parameters to optimize
        lr (float, optional): Learning rate. Default: 1e-3
        betas (Tuple[float, float], optional): Coefficients for computing running averages of
            gradient and its square. Default: (0.9, 0.999)
        eps (float, optional): Term added to denominator to improve numerical stability. Default: 1e-8
        weight_decay (float, optional): Weight decay (L2 penalty). Default: 0.01
        mode (str, optional): Optimization step mode - one of 'stochastic_rounding', 'fp32_master',
            'fp31_master', or 'fp32_master_custom_rounding'. Default: 'stochastic_rounding'
        bucket_size (int, optional): Size of memory buckets in bytes. Default: 4GB
        decoupled_weight_decay (bool, optional): Whether to decouple weight decay. Default: False
    """

    supported_modes = {
        "stochastic_rounding": {
            "step": adam_step_stochastic_rounding,
            "offload": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
            },
        },
        "fp32_master": {
            "step": adam_step_fp32_master,
            "offload": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "master_params": torch.float32,
            },
        },
        "fp31_master": {
            "step": adam_step_fp31_master,
            "offload": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "rounding_error": torch.int16,
            },
        },
        "fp32_master_custom_rounding": {
            "step": adam_step_fp32_master_custom_rounding,
            "offload": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "rounding_error": torch.int16,
            },
        },
    }

    def __init__(
        self,
        model,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        mode="stochastic_rounding",
        bucket_size=8 * (1024**3),
        decoupled_weight_decay=False,
        verbose=0,
    ):
        assert (
            mode in self.supported_modes
        ), f"Invalid mode: {mode}, available modes: {self.supported_modes.keys()}"
        self.mode = mode
        self.step_fn = self.supported_modes[self.mode]["step"]

        modules = get_leaf_modules_with_params(model)
        params = []
        for module in modules:
            for p in module.parameters():
                if p.requires_grad:
                    params.append(p)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(OffloadAdam, self).__init__(params, defaults)

        self.decoupled_weight_decay = decoupled_weight_decay
        self.verbose = verbose

        self.device_states = {}
        self.d2h_events = {}
        self.h2d_events = {}
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

        param2group = {}
        for group in self.param_groups:
            for param in group["params"]:
                param2group[param] = group

        # hooks for h2d transfer
        for module in modules:
            module.register_full_backward_pre_hook(self.pre_backward_hook)
            for param in module.parameters():
                if not param.requires_grad:
                    continue

                group = param2group[param]
                param.register_post_accumulate_grad_hook(
                    self._create_post_accumulate_grad_hook(param, group)
                )
                self.device_states[param] = {}
                state = self.state[param]
                state["step"] = 0

        offload_config = self.supported_modes[self.mode]["offload"]
        self.offload_config = offload_config
        self.offload_state_keys = list(offload_config.keys())

        self.host_states = PinnedMemoryManager(
            params, self.offload_config, bucket_size, verbose=self.verbose
        )
        if self.mode == "fp32_master":
            for param in params:
                self.host_states.get(param, "master_params").copy_(param.data)
        if self.verbose > 0:
            print(
                f"Total memory requested: {self.host_states.total_memory_requested:.2f}GB,"
                f" Total memory allocated: {self.host_states.total_memory_allocated:.2f}GB"
            )

    def ensure_on_device(
        self, device_states, param, offload_state_keys, param_chunk=None
    ):
        """Ensure the given states are on the device.
        In case prefetch is not set, copy the states from host to device here."""
        if param in self.h2d_events:
            self.h2d_events[param].synchronize()
        for offload_state_key in offload_state_keys:
            if device_states.get(offload_state_key, None) is None:
                device_states[offload_state_key] = self.host_states.get(
                    param, offload_state_key, param_chunk
                ).to(param.device, non_blocking=True)

    def issue_h2d_transfer(self, param, offload_state_keys, param_chunk=None):
        """Issue host to device transfers for the given states"""
        main_stream = torch.cuda.current_stream()
        device_states = self.device_states[param]
        if param in self.d2h_events:
            self.d2h_events[param].synchronize()
        self.h2d_stream.wait_stream(main_stream)
        for offload_state_key in offload_state_keys:
            with torch.cuda.stream(self.h2d_stream):
                device_states[offload_state_key] = self.host_states.get(
                    param, offload_state_key, param_chunk
                ).to(param.device, non_blocking=True)
        self.h2d_events[param] = self.h2d_stream.record_event()

    def issue_d2h_transfer(self, param, offload_state_keys, param_chunk=None):
        """Issue device to host transfers for the given states"""
        main_stream = torch.cuda.current_stream()
        device_states = self.device_states[param]
        if param in self.h2d_events:
            self.h2d_events[param].synchronize()
        self.d2h_stream.wait_stream(main_stream)
        for offload_state_key in offload_state_keys:
            with torch.cuda.stream(self.d2h_stream):
                self.host_states.get(param, offload_state_key, param_chunk).copy_(
                    device_states[offload_state_key], non_blocking=True
                )
            # release device memory
            device_states[offload_state_key].record_stream(self.d2h_stream)
            device_states[offload_state_key] = None
        self.d2h_events[param] = self.d2h_stream.record_event()

    def pre_backward_hook(self, module, grad_output):
        for param in module.parameters():
            self.issue_h2d_transfer(param, self.offload_state_keys)

    def _create_post_accumulate_grad_hook(self, param, group):
        @torch.no_grad()
        def optimizer_step_hook(*unused):
            if param.grad is None:
                return

            # states prefetched in pre_backward_hook
            device_states = self.device_states[param]
            state = self.state[param]

            # handle gradients
            device_states["grad"] = param.grad
            param.grad = None

            # optimizer step
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            state = self.state[param]
            state["step"] += 1

            self.ensure_on_device(
                device_states,
                param,
                self.offload_state_keys,
            )
            self.step_fn(
                param.data,
                device_states,
                lr,
                weight_decay,
                beta1,
                beta2,
                eps,
                state["step"],
                decoupled_weight_decay=self.decoupled_weight_decay,
            )
            # Copy states back to Host
            self.issue_d2h_transfer(param, self.offload_state_keys)
            device_states["grad"] = None

        return optimizer_step_hook

    def step(self, closure=None):
        """No-op."""
        return

    def zero_grad(self, set_to_none: bool = False):
        """No-op."""
        return
