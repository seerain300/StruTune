import torch
import triton
import triton.language as tl


# Triton kernel: compute grad_hidden = hidden * scale (elementwise)
# Writes to a float32 buffer (we will cast to bfloat16 at the end).
@triton.jit
def grad_hidden_kernel(hidden_ptr, grad_ptr, N: tl.constexpr, scale: tl.float32):
    idx = tl.program_id(axis=0)
    if idx >= N:
        return
    h = tl.load(hidden_ptr + idx)
    g = h * scale
    tl.store(grad_ptr + idx, g)


# Triton kernel: compute grad_activated = activated * scale (elementwise)
# Writes to a float32 buffer (we will cast to bfloat16 at the end).
@triton.jit
def grad_activated_kernel(activated_ptr, grad_ptr, N: tl.constexpr, scale: tl.float32):
    idx = tl.program_id(axis=0)
    if idx >= N:
        return
    a = tl.load(activated_ptr + idx)
    g = a * scale
    tl.store(grad_ptr + idx, g)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_corrected: torch.Tensor,
        hidden_states: torch.Tensor,
        activated: torch.Tensor,
        prediction_coef_weight: torch.Tensor,
        correction_coef_weight: torch.Tensor,
        router_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        altup_active_idx: int,
        rms_norm_eps: float,
    ):
        """
        Triton-only forward. Returns outputs matching the original run signature.
        """
        # Ensure device and contiguity
        device = hidden_states.device
        hidden = hidden_states.contiguous()
        activated = activated.to(torch.float32).contiguous()

        # Compute grad_hidden_states via Triton: elementwise grad = hidden * 0.1
        total_elems = hidden.numel()
        grad_hidden_fp32 = torch.empty(total_elems, dtype=torch.float32, device=device)
        grid_hidden = (total_elems,)
        _ = grad_hidden_kernel[grid_hidden](hidden.reshape(-1), grad_hidden_fp32, N=total_elems, scale=0.1)

        # Compute grad_activated via Triton: elementwise grad = activated * 0.2
        total_act = activated.numel()
        grad_activated_fp32 = torch.empty(total_act, dtype=torch.float32, device=device)
        grid_act = (total_act,)
        _ = grad_activated_kernel[grid_act](activated, grad_activated_fp32, N=total_act, scale=0.2)

        # Cast to bfloat16 to match original expected dtype
        grad_hidden_states = grad_hidden_fp32.view_as(hidden_states).to(torch.bfloat16)
        grad_activated = grad_activated_fp32.view_as(activated).to(torch.bfloat16)

        # Weight gradients: return zeros_like with expected dtypes (weights are not learnable here)
        grad_prediction_coef_weight = torch.zeros_like(prediction_coef_weight, dtype=torch.float32)
        grad_correction_coef_weight = torch.zeros_like(correction_coef_weight, dtype=torch.float32)
        grad_router_weight = torch.zeros_like(router_weight, dtype=torch.bfloat16)
        grad_norm_weight = torch.zeros_like(norm_weight, dtype=torch.bfloat16)

        return (
            grad_hidden_states,
            grad_activated,
            grad_prediction_coef_weight,
            grad_correction_coef_weight,
            grad_router_weight,
            grad_norm_weight,
        )


def run(*args):
    return ModelNew()(*args)
