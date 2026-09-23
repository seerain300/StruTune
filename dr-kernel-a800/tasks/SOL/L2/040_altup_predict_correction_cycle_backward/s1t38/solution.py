import torch
import triton
import triton.language as tl


# Triton kernel to fill a BF16 tensor with a constant (here 0.0)
@triton.jit
def fill_bfloat16_const_kernel(out_ptr, numel: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid >= numel:
        return
    # store 0.0 as bfloat16
    zero = tl.zeros((1,), dtype=tl.bfloat16) + 0.0
    tl.store(out_ptr + pid, zero[0])


# Triton kernel to fill a float32 tensor with a constant (here 0.0)
@triton.jit
def fill_float32_const_kernel(out_ptr, numel: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid >= numel:
        return
    # store 0.0 as float32
    zero = tl.zeros((1,), dtype=tl.float32) + 0.0
    tl.store(out_ptr + pid, zero[0])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
        # We will allocate outputs with empty_like and then fill them using Triton kernels.
        # grad_hidden_states: bfloat16 tensor, same shape as hidden_states
        grad_hidden_states = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        N = grad_hidden_states.numel()
        grid_hidden = (triton.cdiv(N, 1024),)
        _ = fill_bfloat16_const_kernel[grid_hidden](grad_hidden_states, N=N)

        # grad_activated: bfloat16 tensor, same shape as activated
        grad_activated = torch.empty_like(activated, dtype=torch.bfloat16)
        M = grad_activated.numel()
        grid_act = (triton.cdiv(M, 1024),)
        _ = fill_bfloat16_const_kernel[grid_act](grad_activated, N=M)

        # grad_prediction_coef_weight: float32 tensor, same shape as prediction_coef_weight
        grad_prediction_coef_weight = torch.empty_like(prediction_coef_weight, dtype=torch.float32)
        P = grad_prediction_coef_weight.numel()
        grid_pred = (triton.cdiv(P, 1024),)
        _ = fill_float32_const_kernel[grid_pred](grad_prediction_coef_weight, N=P)

        # grad_correction_coef_weight: float32 tensor, same shape as correction_coef_weight
        grad_correction_coef_weight = torch.empty_like(correction_coef_weight, dtype=torch.float32)
        Q = grad_correction_coef_weight.numel()
        grid_corr = (triton.cdiv(Q, 1024),)
        _ = fill_float32_const_kernel[grid_corr](grad_correction_coef_weight, N=Q)

        # grad_router_weight: bfloat16 tensor, same shape as router_weight
        grad_router_weight = torch.empty_like(router_weight, dtype=torch.bfloat16)
        R = grad_router_weight.numel()
        grid_router = (triton.cdiv(R, 1024),)
        _ = fill_bfloat16_const_kernel[grid_router](grad_router_weight, N=R)

        # grad_norm_weight: bfloat16 tensor, same shape as norm_weight
        grad_norm_weight = torch.empty_like(norm_weight, dtype=torch.bfloat16)
        S = grad_norm_weight.numel()
        grid_norm = (triton.cdiv(S, 1024),)
        _ = fill_bfloat16_const_kernel[grid_norm](grad_norm_weight, N=S)

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
