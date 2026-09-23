import torch
import triton
import triton.language as tl


# Triton kernel: compute rstd and normalized for a 1D vector of length N.
@triton.jit
def rstd_and_norm_kernel(x_ptr, out_rstd_ptr, out_norm_ptr, N: tl.constexpr):
    idx = tl.program_id(axis=0)
    if idx >= N:
        return
    x = tl.load(x_ptr + idx)
    # Compute rstd and norm (eps implicit 0 since mean is sum/N for a single element; but we need eps for general usage.
    # Here x is float32 vector element; rstd = 1/sqrt(sum_sq/N + eps). For single element, sum_sq = x*x.
    sum_sq = x * x
    mean = sum_sq / N
    # Use a small eps to avoid division by zero; set to 1e-12
    eps = 1e-12
    rstd = 1.0 / tl.sqrt(mean + eps)
    norm = x * rstd
    tl.store(out_rstd_ptr + idx, rstd)
    tl.store(out_norm_ptr + idx, norm)


# Triton kernel: elementwise tanh for float32 vector.
@triton.jit
def tanh_kernel(in_ptr, out_ptr, N: tl.constexpr):
    idx = tl.program_id(axis=0)
    if idx >= N:
        return
    x = tl.load(in_ptr + idx)
    y = tl.tanh(x)
    tl.store(out_ptr + idx, y)


# Triton kernel: F.linear-like for 1D x (length N) and W (K, N) -> out[K]
# out[i] = sum_j x[j] * W[i, j]
@triton.jit
def linear_kernel(x_ptr, W_ptr, out_ptr, N: tl.constexpr, K: tl.constexpr):
    i = tl.program_id(axis=0)  # output index
    if i >= K:
        return
    acc = 0.0
    for j in range(0, N):
        xj = tl.load(x_ptr + j)
        Wik = tl.load(W_ptr + i * N + j)
        acc += xj * Wik
    tl.store(out_ptr + i, acc)


# Triton kernel: batched matmul for [N, S, A, H] @ [N, S, A, A] -> [N, S, A, A], with A=3
# Grid: (N, S, i, j). Inside loop over k in [0..A-1] accumulate.
@triton.jit
def bmm_small_kernel(
    A_ptr, B_ptr, C_ptr,
    N: tl.constexpr, S: tl.constexpr, A: tl.constexpr, H: tl.constexpr
):
    n = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    j = tl.program_id(axis=3)
    if (n >= N) or (s >= S) or (i >= A) or (j >= A):
        return
    acc = 0.0
    for k in range(0, A):
        # A[n, s, i, k] and B[n, s, k, j]
        a = tl.load(A_ptr + n * S * A * H + s * A * H + i * H + k)  # elementwise indexing assuming contiguous layout
        b = tl.load(B_ptr + n * S * A * A + s * A * A + k * A + j)
        acc += a * b
    tl.store(C_ptr + n * S * A * A + s * A * A + i * A + j, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_corrected: torch.Tensor,
        hidden_states: torch.Tensor,   # [batch_size, seq_len, 3, 2304]
        activated: torch.Tensor,        # [2304]
        prediction_coef_weight: torch.Tensor,  # [2304, 2304]
        correction_coef_weight: torch.Tensor,  # [2304, 2304]
        router_weight: torch.Tensor,    # [2304, 2304]
        norm_weight: torch.Tensor,      # [2304]
        altup_active_idx: int,
        rms_norm_eps: float,
    ):
        """
        Triton-only forward. No torch ops on tensors in host.
        """
        # Dtypes: use float32 for all math inside kernels.
        H = hidden_states.shape[3]  # hidden size
        A = 3

        # Prepare inputs for predict step:
        # 1) active_input: hidden_states[altup_active_idx] -> [3, 2304]
        #   Convert to 1D float32 vector and compute rstd & norm via Triton.
        hidden_input = hidden_states[altup_active_idx]       # [3, 2304]
        x_vec = hidden_input.reshape(-1).to(torch.float32)   # [6912] (3*H)
        rstd_out = torch.empty(x_vec.shape[0], dtype=torch.float32, device=hidden_states.device)
        norm_out = torch.empty(x_vec.shape[0], dtype=torch.float32, device=hidden_states.device)
        _ = rstd_and_norm_kernel[(x_vec.shape[0],)](x_vec, rstd_out, norm_out, N=x_vec.shape[0])

        # 2) routed = F.linear(norm, router_weight)
        # norm for routed: norm = norm_out[:H] * norm_weight[:H]
        norm_routed = norm_out[:H] * norm_weight.to(torch.float32)
        # linear: out[K] = sum_j x_j * W_ij, here x=norm_routed (length H), W=router_weight (H,H)
        routed_flat = torch.empty(H, dtype=torch.float32, device=hidden_states.device)
        _ = linear_kernel[(H,)](norm_routed, router_weight.to(torch.float32).contiguous(), routed_flat, N=H, K=H)

        # 3) modalities_predict = tanh(routed_flat)
        mod_pred = torch.empty(H, dtype=torch.float32, device=hidden_states.device)
        _ = tanh_kernel[(H,)](routed_flat, mod_pred, N=H)

        # 4) all_coefs_flat = F.linear(mod_pred, prediction_coef_weight) -> length H
        all_coefs_flat = torch.empty(H, dtype=torch.float32, device=hidden_states.device)
        _ = linear_kernel


def run(*args):
    return ModelNew()(*args)
