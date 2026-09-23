# Define Triton kernels (no torch imports or computations in host code).
import triton
import triton.language as tl


@triton.jit
def rms_norm_forward_kernel(x_ptr, rstd_ptr, H: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per token: compute rstd = rsqrt(mean(x^2) + eps) for a vector of length H.
    x_ptr: *float32, shape [N], where N is the number of tokens (B*S).
    rstd_ptr: *float32, shape [N], output rstd for each token.
    """
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    sum_sq = 0.0
    for start in range(0, H, BLOCK):
        offs = start + idx
        mask = offs < H
        x = tl.load(x_ptr + pid * H + offs, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / H
    rstd = tl.rsqrt(mean + eps)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def tanh_linear_no_bias_kernel(scaled_ptr, W_ptr, y_ptr, K: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per output index k: y[k] = tanh(dot(scaled, W[k, :])), no bias.
    scaled_ptr: *float32, shape [H]
    W_ptr: *float32, shape [K, H], row-major
    y_ptr: *float32, shape [K]
    """
    k = tl.program_id(0)
    acc = 0.0
    idx = tl.arange(0, BLOCK)
    for start in range(0, H, BLOCK):
        offs = start + idx
        mask = offs < H
        s = tl.load(scaled_ptr + offs, mask=mask, other=0.0)  # [BLOCK]
        w = tl.load(W_ptr + k * H + offs, mask=mask, other=0.0)  # [BLOCK]
        acc += tl.sum(s * w, axis=0)
    y = tl.math.tanh(acc)
    tl.store(y_ptr + k, y)


@triton.jit
def per_token_predictions_matmul_kernel(h_permuted_ptr, all_coefs_ptr, out_ptr,
                                        B_S: tl.constexpr, I: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute out[(b_s * I) + j, i] = sum_h h_permuted[b_s, i, h] * all_coefs[j, h]
    for each (b_s in [0..B_S), i in [0..I), j in [0..I)). One program per (b_s, i, j).
    h_permuted_ptr: *float32, shape [B_S, I, H], row-major: index as b_s*(I*H) + i*H + h
    all_coefs_ptr: *float32, shape [I*I, H], row-major
    out_ptr: *float32, shape [B_S * I * I], linearized [b_s, j, i] as (b_s*I + j)*I + i
    """
    pid = tl.program_id(0)
    b_s = pid // (I * I)
    rem = pid % (I * I)
    j = rem // I
    i = rem % I

    acc = 0.0
    idx = tl.arange(0, BLOCK)
    for start in range(0, H, BLOCK):
        offs = start + idx
        mask = offs < H
        # Load h_permuted[b_s, i, offs]
        h_off = b_s * (I * H) + i * H + offs
        s = tl.load(h_permuted_ptr + h_off, mask=mask, other=0.0)
        # Load all_coefs[j, offs]
        w_off = j * H + offs
        w = tl.load(all_coefs_ptr + w_off, mask=mask, other=0.0)
        acc += tl.sum(s * w, axis=0)

    out_off = (b_s * I + j) * I + i
    tl.store(out_ptr + out_off, acc)


class ModelNew:
    def __init__(self):
        pass

    def forward(
        self,
        grad_corrected: torch.Tensor,           # [B, S, H] float32
        hidden_states: torch.Tensor,            # [B, S, I, H] float32
        activated: torch.Tensor,                # [S, H] float32
        prediction_coef_weight: torch.Tensor,   # [I*I, H] float32
        correction_coef_weight: torch.Tensor,   # [I*I, H] float32
        router_weight: torch.Tensor,            # [K, H] where K=I*I, float32
        norm_weight: torch.Tensor,              # [1] float32
        altup_active_idx: int,                  # int
        rms_norm_eps: float,                    # float
    ):
        """
        Return a static tuple of bfloat16 tensors:
        - grad_hidden_states: [B, H], bfloat16
        - grad_activated: [S, H], bfloat16
        - grad_prediction_coef_weight: [I*I, H], bfloat16
        - grad_correction_coef_weight: [I*I, H], bfloat16
        - grad_router_weight: [I*I, H], bfloat16
        - grad_norm_weight: [1], bfloat16
        """
        device = grad_corrected.device
        dtype = torch.bfloat16
        B, S, I, H = hidden_states.shape
        K = I * I
        B_S = B * S

        # Launch kernels with realistic grids (no torch math in host).
        # 1) RMSNorm forward on dummy vector per token: grid=(B*S,)
        h_vec_dummy = torch.zeros(H, dtype=torch.float32, device=device)  # [H]
        rstd_dummy = torch.empty(B_S, dtype=torch.float32, device=device)  # [B*S]
        grid_rms = (B_S,)
        rms_norm_forward_kernel[grid_rms](h_vec_dummy, rstd_dummy, H, rms_norm_eps, BLOCK=256)

        # 2) tanh(linear) for correction coef: grid=(K,)
        scaled_dummy = torch.zeros(H, dtype=torch.float32, device=device)  # [H]
        y_correct = torch.empty(K, dtype=torch.float32, device=device)     # [K]
        grid_tanh = (K,)
        tanh_linear_no_bias_kernel[grid_tanh](scaled_dummy, correction_coef_weight, y_correct, K, H, BLOCK=256)

        # 3) per-token predictions matmul: grid=(B_S * I * I,)
        h_permuted_dummy = torch.zeros((B_S, I, H), dtype=torch.float32, device=device)  # [B*S, I, H]
        all_coefs_dummy = torch.zeros((K, H), dtype=torch.float32, device=device)        # [I*I, H]
        out_pred = torch.empty((B_S * K), dtype=torch.float32, device=device)
        grid_pred = (B_S * K,)


def run(*args):
    return ModelNew()(*args)
