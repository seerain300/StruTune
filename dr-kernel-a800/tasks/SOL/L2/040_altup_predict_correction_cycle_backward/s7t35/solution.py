import torch
import triton
import triton.language as tl


# Constants from the original code
H = 2304        # hidden size
L = 9           # router output length
Kp = 9          # prediction coef output length
Kc = 9          # correction coef output length (not used in forward computation)


# Triton kernel: compute rstd per row of x_ptr of shape (M, H), M = B*S
@triton.jit
def compute_rstd_kernel(x_ptr, rstd_ptr, M: tl.constexpr, H: tl.constexpr, eps: tl.constexpr):
    # Each program handles one (b, s) row
    pid = tl.program_id(0)
    row_start = pid * H
    offs = row_start + tl.arange(0, H)
    x = tl.load(x_ptr + offs)  # vector of length H
    sq = x * x
    sum_sq = tl.sum(sq, axis=0)
    mean = sum_sq / H
    rstd = 1.0 / tl.sqrt(mean + eps)
    tl.store(rstd_ptr + pid, rstd)


# Triton kernel: compute routed = tanh(dot(x_norm, router_weight)), output routed_ptr[M, L]
@triton.jit
def routed_linear_tanh_kernel(x_norm_ptr, router_weight_ptr, routed_ptr,
                              M: tl.constexpr, H: tl.constexpr, L: tl.constexpr):
    # Grid is (M, L): each program computes routed[m, l]
    pid_m = tl.program_id(0)   # row index m in [0..M-1]
    pid_l = tl.program_id(1)   # output column index l in [0..L-1]
    sum_val = 0.0
    # Dot product over H: routed[m, l] = sum_h x_norm[m, h] * router_weight[l, h]
    for h in range(0, H):
        sum_val += tl.load(x_norm_ptr + pid_m * H + h) * tl.load(router_weight_ptr + pid_l * H + h)
    out = tl.math.tanh(sum_val)
    tl.store(routed_ptr + pid_m * L + pid_l, out)


# Triton kernel: compute coef = F.linear(modalities, prediction_coef_weight), output coef_ptr[M, Kp]
# Here, modalities are tanh(routed). We'll pass tanh(routed) as routed for simplicity and compute
# tanh in Triton; but since routed is already tanh in the kernel above, coef[m, k] = sum_l routed[m, l] * pred_coef[k, l].
@triton.jit
def coef_linear_kernel(routed_ptr, pred_coef_weight_ptr, coef_ptr,
                        M: tl.constexpr, Kp: tl.constexpr, L: tl.constexpr):
    # Grid is (M, Kp): each program computes coef[m, k]
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)  # output column index k in [0..Kp-1]
    sum_val = 0.0
    # coef[m, k] = sum_l routed[m, l] * pred_coef_weight[k, l]
    for l in range(0, L):
        sum_val += tl.load(routed_ptr + pid_m * L + l) * tl.load(pred_coef_weight_ptr + pid_k * L + l)
    tl.store(coef_ptr + pid_m * Kp + pid_k, sum_val)


# Triton kernel: elementwise multiply routed * coef to form predictions
@triton.jit
def pred_elementwise_kernel(routed_ptr, coef_ptr, predictions_ptr,
                             M: tl.constexpr, L: tl.constexpr, Kp: tl.constexpr):
    # Grid is (M, Kp): each program computes predictions[m, k] = routed[m, :] * coef[m, k]
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    # predictions[m, k] = routed[m, :] * coef[m, k] (elementwise across L)
    for l in range(0, L):
        routed_val = tl.load(routed_ptr + pid_m * L + l)
        coef_val = tl.load(coef_ptr + pid_m * Kp + pid_k)
        out = routed_val * coef_val
        tl.store(predictions_ptr + pid_m * Kp + pid_k, out)


def _run_triton_predict(hidden_states, altup_active_idx, prediction_coef_weight, router_weight, eps: float, batch_size: int, seq_len: int):
    """
    Forward recomputation for the 'predict' step using Triton kernels.
    Returns predictions of shape (batch_size, seq_len, Kp), dtype bfloat16.
    """
    assert hidden_states.is_cuda and prediction_coef_weight.is_cuda and router_weight.is_cuda
    # Select the correct hidden input according to altup_active_idx
    active_input = hidden_states[altup_active_idx]  # shape (B, S, H)
    B = batch_size
    S = seq_len
    H_val = active_input.shape[2]
    device = active_input.device

    # 1) Compute rstd per (b, s) row
    x = active_input.reshape(B * S, H_val)          # (M, H)
    rstd = torch.empty(B * S, device=device, dtype=torch.float32)  # rstd for each (b, s) row
    grid_rstd = (B * S,)
    compute_rstd_kernel[grid_rstd](x, rstd, M=B * S, H=H_val, eps=eps)

    # 2) Normalize
    x_norm = x * rstd.view(B * S, 1)                # (M, H), float32

    # 3) routed = tanh(F.linear(x_norm, router_weight)) -> shape (M, L)
    routed = torch.empty((B * S, L), device=device, dtype=torch.float32)
    grid_routed = (B * S, L)
    routed_linear_tanh_kernel[grid_routed](x_norm, router_weight, routed, M=B * S, H=H_val, L=L)

    # 4) coef = F.linear(routed, prediction_coef_weight) -> shape (M, Kp)
    #    Note: routed is tanh(routed from step 1); here we treat routed as tanh(routed).
    coef = torch.empty((B * S, Kp), device=device, dtype=torch.float32)
    grid_coef = (B * S, Kp)
    coef_linear_kernel[grid_coef](routed, prediction_coef_weight, coef, M=B * S, Kp=Kp, L=L)

    # 5) predictions = routed * coef (elementwise across L)
    #    Shape: (B, S, Kp) in bfloat16
    predictions = torch.empty((B, S, Kp), device=device, dtype=torch.bfloat16)
    # Flatten to (M, Kp) for kernel
    pred_flat = predictions.reshape(B * S, Kp)
    grid_pred = (B * S, Kp)
    pred_elementwise_kernel[grid_pred](routed, coef, pred_flat, M=B * S, L=L, Kp=Kp)

    return predictions


class ModelNew(torch.nn.Module):
    def forward(self, grad_corrected: torch.Tensor, hidden_states: torch.Tensor,
                activated: torch.Tensor, prediction_coef_weight: torch.Tensor,
                correction_coef_weight: torch.Tensor, router_weight: torch.Tensor,
                norm_weight: torch.Tensor, altup_active_idx: int, rms_norm_eps: float):
        # Triton-based forward recomputation for the 'predict' step
        batch_size = hidden_states.shape[1]
        seq_len = hidden_states.shape[2]
        # Launch Triton kernels
        predictions = _run_triton_predict(hidden_states, altup_active_idx, prediction_coef_weight, router_weight, rms_norm_eps, batch_size, seq_len)
        # Return predictions
        return predictions


def run(*args):
    return ModelNew()(*args)
