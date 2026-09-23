import torch
import math
import triton
import triton.language as tl


@triton.jit
def softplus_and_g(A_log_ptr, a_ptr, dt_bias_ptr, b_ptr, g_ptr, beta_ptr,
                    T: tl.constexpr, H: tl.constexpr,
                    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr):
    """
    Compute g and beta:
      g[t, j] = exp(-exp(A_log[j]) * softplus(a[t, j] + dt_bias[j]))
      beta[t, j] = 1 / (1 + exp(-b[t, j]))
    Store as g_ptr and beta_ptr of shape [T, H].
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    # ranges
    t_idx = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    h_idx = pid_h * BLOCK_H + tl.arange(0, H)  # H is constexpr=8
    mask_t = t_idx < T

    # Load dt_bias and A_log for j in h_idx
    A_log = tl.load(A_log_ptr + h_idx, mask=h_idx < H, other=0.0)  # [H]
    # Load a[t, j] for t in t_idx and j in h_idx
    a_vals = tl.load(a_ptr + t_idx[:, None] * H + h_idx[None, :],
                     mask=mask_t[:, None], other=0.0)  # [BLOCK_T, H]
    dt_vals = tl.load(dt_bias_ptr + h_idx, mask=h_idx < H, other=0.0)  # [H]
    b_vals = tl.load(b_ptr + t_idx[:, None] * H + h_idx[None, :],
                     mask=mask_t[:, None], other=0.0)  # [BLOCK_T, H]

    # Compute softplus(a + dt_bias)
    sum_ad = a_vals + dt_vals[None, :]
    sp = tl.log(1.0 + tl.exp(sum_ad))  # softplus
    # g = exp(-exp(A_log) * softplus)
    g_part = tl.exp(A_log[None, :]) * sp
    g_vals = tl.exp(-g_part)  # [BLOCK_T, H]

    # beta = 1 / (1 + exp(-b))
    beta_vals = 1.0 / (1.0 + tl.exp(-b_vals))

    # Store g and beta
    tl.store(g_ptr + t_idx[:, None] * H + h_idx[None, :], g_vals, mask=mask_t[:, None])
    tl.store(beta_ptr + t_idx[:, None] * H + h_idx[None, :], beta_vals, mask=mask_t[:, None])


@triton.jit
def dot_row_triton(A_ptr, B_ptr, Out_ptr, K: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute scalar = sum(A[K] * B[K]) for 1D A and B of length K.
    """
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs
        mask = k_idx < K
        a = tl.load(A_ptr + k_idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + k_idx, mask=mask, other=0.0)
        acc += tl.sum(a * b, axis=0)
    tl.store(Out_ptr, acc)


@triton.jit
def mm_row_triton(A_ptr, B_ptr, C_ptr, K: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C = A @ B where A is [1, K], B is [K, N], C is [1, N].
    Here N=128. We output a row vector of length 128.
    """
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((128,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs
        mask = k_idx < K
        a_row = tl.load(A_ptr + k_idx, mask=mask, other=0.0)  # [BLOCK_K]
        # Load B rows: B is [K, 128] contiguous; we index B[k, :] using k_idx
        b_block = tl.load(B_ptr + k_idx[:, None] * 128 + tl.arange(0, 128),
                          mask=mask[:, None], other=0.0)  # [BLOCK_K, 128]
        prod = a_row[:, None] * b_block  # [BLOCK_K, 128]
        acc += tl.sum(prod, axis=0)  # reduce over K tile -> [128]
    tl.store(C_ptr + tl.arange(0, 128), acc)


def _run_triton_version(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Triton-forward helper. Returns output [T, 8, 128] bfloat16 and new_state [1, 8, 128, 128] float32.
    """
    device = q.device
    T, H_q, K = q.shape
    _, H_k, _ = k.shape
    _, H_v, _ = v.shape

    # Ensure contiguity
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    state_single = state[0].contiguous()  # [8, 128, 128]
    state_curr = state_single[:4].float().contiguous()  # [4, 128, 128]

    # Allocate outputs
    output = torch.empty((T, H_v, K), dtype=torch.bfloat16, device=device)
    new_state = torch.empty((1, H_v, K, K), dtype=torch.float32, device=device)  # placeholder, will be filled per t

    # Prepare g and beta (compute in float32)
    g = torch.empty((T, H_v), dtype=torch.float32, device=device)
    beta = torch.empty((T, H_v), dtype=torch.float32, device=device)

    # Launch Triton softplus_and_g: 2D grid over (T, H_v)
    grid = (triton.cdiv(T, 64), triton.cdiv(H_v, 8))
    softplus_and_g[grid](A_log, a, dt_bias, b, g, beta, T, H_v, BLOCK_T=64, BLOCK_H=8)

    # Loop over time
    for t in range(T):
        # Update state and compute output per v head
        # We will compute o[h] for each head h and store output[t, j] = o[h] (same across j due to original behavior).
        q_exp_h = q[t].unsqueeze(0)  # [1, 128]
        for j in range(H_v):
            # Initialize scalars and vectors for recurrence
            old_v_vec = torch.zeros((H_q,), dtype=torch.float32, device=device)
            new_v_vec = torch.zeros((H_q,), dtype=torch.float32, device=device)

            # Compute old_v_j[h] = dot(k[t, :], state_curr[h, :, :])
            for h in range(H_q):
                k_row = k[t, h]  # [128]
                state_row = state_curr[h]  # [128]
                old_v_vec[h] = dot_row_triton[(1,)](k_row, state_row, torch.empty((), dtype=torch.float32, device=device),
                                                    K, 128).item()  # scalar

            # Compute new_v_j[h] = beta[t, j] * v[t, j, :] + (1 - beta[t, j]) * old_v_vec[h]
            v_row = v[t, j]  # [128], float32 (v is float32 in inputs)
            beta_tj = beta[t, j]
            # Apply per head:
            for h in range(H_q):
                new_v_vec[h] = beta_tj * float(v_row[h]) + (1.0 - beta_tj) * float(old_v_vec[h])

            # Update state_curr[h] using g[t, j]
            g_tj = g[t, j]
            for h in range(H_q):
                # state_old = g * state_old + new_v - old_v
                old_v_vec[h] = float(dot_row_triton[(1,)](k[t, h], state_curr[h], torch.empty((), dtype=torch.float32, device=device),
                                                          K, 128).item())
                new_state_curr = g_tj * float(state_curr[h].sum().item()) + new_v_vec[h] - old_v_vec[h]
                # Convert back to vector and store (keeping original layout [H_q, 128, 128] in new_state)
                # Note: we cannot write into 4D tensor directly from Triton; keep as torch update.
                # For output o, we need q_exp[t, h] @ state_curr[h].
                # Compute q_exp[t, h] is q[t, h], but original q_exp is from repeat_interleave; here we use q[t, h].
                # We'll compute mm via Triton for each h.
                # Prepare A = q[t, h] as [1,128], B = state_curr[h] as [128,128]
                A_row = q[t, h].view(1, 128).contiguous()
                B_mat = state_curr[h].view(128, 128).contiguous()
                C_row = torch.empty((128,), dtype=torch.float32, device=device)
                mm_row_triton[(1,)](A_row, B_mat, C_row, 128, 128)
                o_h = (scale * C_row).to(torch.bfloat16)
                # Store output[t, j, :] = o_h
                output[t, j] = o_h  # Triton output is scalar across K; but we created output as [T,H_v,K].
                # Since original output[t,j] has shape [128], we need to fill K dimension; copy o_h across K.
                # The original code writes the same o across K dims; we mimic that.
                # Broadcast to K dimension: output[t, j, :] = o_h
                # Using PyTorch for write: expand along K
                output[t, j, :] = o_h.expand(-1)  # bfloat16

        # Store new_state at segment 0: set new_state[0] = state_curr transposed to [H_q, K, K] format
        # We store as [1, 8, 128, 128] and fill only H_q=4 heads for j=0..3; for j>=4, keep zeros. The original new_state has 8 heads; we match [1,8,128,128].
        # For simplicity and correctness in Triton-only requirement, we fill first 4 heads:
        # new_state[0, :4, :, :] = state_curr. Others remain zero.
        new_state[0, :4, :, :] = state_curr.unsqueeze(0)  # [1, 4, 128, 128]

    return output, new_state


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        output, new_state = _run_triton_version(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
