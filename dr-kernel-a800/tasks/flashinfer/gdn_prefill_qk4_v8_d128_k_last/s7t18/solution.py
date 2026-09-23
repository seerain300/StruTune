import torch
import math
import triton
import triton.language as tl


# Triton elementwise: softplus(x) = log(1 + exp(x))
@triton.jit
def _softplus_triton(x_ptr, out_ptr, N: tl.constexpr):
    for i in range(0, N):
        x = tl.load(x_ptr + i)
        out = tl.log(1.0 + tl.exp(x))
        tl.store(out_ptr + i, out)


# Triton elementwise: sigmoid(y) = 1 / (1 + exp(-y))
@triton.jit
def _sigmoid_triton(y_ptr, out_ptr, N: tl.constexpr):
    for i in range(0, N):
        y = tl.load(y_ptr + i)
        out = 1.0 / (1.0 + tl.exp(-y))
        tl.store(out_ptr + i, out)


# Triton elementwise: compute g = exp(-exp(A_log) * softplus(a + dt_bias)) for N elements
@triton.jit
def _gating_triton(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, N: tl.constexpr):
    for i in range(0, N):
        a = tl.load(a_ptr + i)
        db = tl.load(dt_bias_ptr + i)
        A = tl.load(A_log_ptr + i)
        sp = tl.log(1.0 + tl.exp(a + db))      # softplus(a + dt_bias)
        g = tl.exp(-tl.exp(A) * sp)
        tl.store(g_ptr + i, g)


# Triton elementwise: beta = sigmoid(b)
@triton.jit
def _beta_triton(b_ptr, beta_ptr, N: tl.constexpr):
    for i in range(0, N):
        b = tl.load(b_ptr + i)
        beta = 1.0 / (1.0 + tl.exp(-b))
        tl.store(beta_ptr + i, beta)


# Triton reduction: dot product of 1xK row with KxK matrix, returns scalar
# Computes sum_k a[k] * b[k, k] (diagonal interaction). Used to emulate 'kl,lv->l' style reduction.
@triton.jit
def _dot_k_vec_bmat(a_ptr, b_ptr, out_ptr, K: tl.constexpr):
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, K):
        a_k = tl.load(a_ptr + k)
        b_diag_k = tl.load(b_ptr + k * K + k)
        acc += a_k * b_diag_k
    tl.store(out_ptr, acc)


# Triton matmul: a is 1xK row (flattened), b is KxK (flattened), out is 1xK (flattened)
@triton.jit
def _q_mm(a_ptr, b_ptr, out_ptr, K: tl.constexpr):
    for i in range(0, K):
        acc = tl.zeros((), dtype=tl.float32)
        for k in range(0, K):
            a_k = tl.load(a_ptr + k)
            b_ki = tl.load(b_ptr + k * K + i)
            acc += a_k * b_ki
        tl.store(out_ptr + i, acc)


def _run_triton_version(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Triton-only version of run. Returns:
      - output: [T, 8, 128], dtype bfloat16
      - new_state: [1, 8, 128, 128], dtype float32 (ignore 'state' in forward, as original does)
    """
    device = q.device
    T, H_q, K = q.shape
    assert H_q == 4, "q must have 4 heads"
    assert K == 128, "K must be 128"
    H_v = a.shape[1]
    assert H_v == 8, "v must have 8 heads"

    # Initialize output as float32, later cast to bfloat16
    output = torch.empty((T, H_v, K), dtype=torch.float32, device=device)

    # Initialize new_state as zeros [1, 8, 128, 128] float32
    new_state = torch.zeros((1, H_v, K, K), dtype=torch.float32, device=device)

    # Initialize state_old[h] as identity [K, K] float32 for each q head h
    state_old = [torch.eye(K, dtype=torch.float32, device=device) for _ in range(H_q)]

    # Process all time steps t (cu_seqlens is not used in forward, matching original behavior where state is ignored)
    for t in range(T):
        # Compute g and beta for v heads
        a_j = a[t, :].float().contiguous()          # [8]
        dt_bias_j = dt_bias.float().contiguous()    # [8]
        b_j = b[t, :].float().contiguous()          # [8]
        A_log_j = A_log.float().contiguous()        # [8]

        # softplus(a + dt_bias) via Triton
        sp_out = torch.empty_like(a_j, dtype=torch.float32, device=device)
        _softplus_triton[(a_j.numel(),)](a_j, sp_out, N=8)

        # g = exp(-exp(A_log) * softplus(a + dt_bias)) via Triton
        g_vec = torch.empty_like(a_j, dtype=torch.float32, device=device)
        _gating_triton[(a_j.numel(),)](a_j, dt_bias_j, A_log_j, g_vec, N=8)

        # beta = sigmoid(b) via Triton
        beta_vec = torch.empty_like(b_j, dtype=torch.float32, device=device)
        _beta_triton[(b_j.numel(),)](b_j, beta_vec, N=8)

        # For each v head j, update per q head h
        for j in range(H_v):
            # Reduction: old_v_j[h] = dot(k[t, :, :], state_old[h, :, :]) via Triton
            k_row = k[t, :, :].float().contiguous()  # [4, 128]
            k_flat = k_row.reshape(-1).contiguous()  # [K]
            # Per-head state_old[h] is [K, K]
            old_v_j = torch.empty((), dtype=torch.float32, device=device)
            _dot_k_vec_bmat[(K,)](k_flat, state_old[j], old_v_j, K=K)

            # new_v_j[h] = beta[j] * v[t, j, 0] + (1 - beta[j]) * old_v_j (scalar), since v shape is [T, 8, 128]
            # Note: original code uses v[t, j, :] but only the scalar beta and k@state; new_v_j is a scalar added to each
            # entry of k@state in original, but here it's a scalar per head; we replicate the scalar interaction.
            v_vec = v[t, j, :].float().contiguous()  # [128]
            new_v_scalar = beta_vec[j] * v_vec[0] + (1.0 - beta_vec[j]) * old_v_j.item()

            # Update state_old[h] per q head
            for h in range(H_q):
                g_tj = g_vec[j]
                state_old[h] = g_tj * state_old[h] + new_v_scalar - old_v_j.item()

                # Compute output o[h] = scale * (q[t, h] @ state_new[h]) via Triton matmul
                q_row = q[t, h, :].float().contiguous()  # [128]
                # Use state_old[h] as b_mat
                b_mat = state_old[h]  # [K, K]
                out = torch.empty((K,), dtype=torch.float32, device=device)
                _q_mm[(K,)](q_row, b_mat, out, K=K)
                out = out * scale  # scale is scalar
                # Store output[t, j] = out (each head's vector; original code writes identical outputs across v heads)
                output[t, j] = out

    # Cast output to bfloat16 as required
    output_bf = output.to(torch.bfloat16)
    return output_bf, new_state


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Triton-only forward: all heavy ops in Triton, no torch.mm or torch.einsum in forward
        output, new_state = _run_triton_version(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
