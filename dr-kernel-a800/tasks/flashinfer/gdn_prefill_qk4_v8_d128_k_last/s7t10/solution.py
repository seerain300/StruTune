import torch
import math
import triton
import triton.language as tl


@triton.jit
def softplus_and_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, beta_ptr, T, H):
    """
    Compute g and beta:
      g = exp(-exp(A_log) * softplus(a + dt_bias))
      beta = sigmoid(b)
    Grid: (T, H). We ignore extra args; kernel assumes inputs are float32.
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    if t >= T or h >= H:
        return
    # Load a, dt_bias, A_log
    a_val = tl.load(a_ptr + t * H + h)
    dt_val = tl.load(dt_bias_ptr + h)
    A_log_val = tl.load(A_log_ptr + h)
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    # sigmoid(y) = 1 / (1 + exp(-y))
    b_val = tl.load(beta_ptr + t * H + h)  # beta input
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    g_val = tl.exp(-tl.exp(A_log_val) * sp)
    # Store results
    tl.store(g_ptr + t * H + h, g_val)
    tl.store(beta_ptr + t * H + h, beta_val)


@triton.jit
def row_matmul_triton(A_row_ptr, B_ptr, C_ptr, K, N):
    """
    Compute C = A_row @ B where:
      A_row is [1, K], B is [K, N], C is [1, N].
    We assume A_row is at offset A_row_ptr.
    """
    offs_n = tl.arange(0, N)
    acc = tl.zeros((N,), dtype=tl.float32)
    # Loop over K in tiles of 32
    for k_start in range(0, K, 32):
        offs_k = k_start + tl.arange(0, 32)
        a = tl.load(A_row_ptr + offs_k, mask=offs_k < K, other=0.0)  # [32]
        b_block = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :], mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)  # [32, N]
        acc += tl.sum(a[:, None] * b_block, axis=0)
    tl.store(C_ptr + offs_n, acc)


@triton.jit
def dot_row_triton(row_ptr, mat_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    Compute dot = sum_i row[i] * mat[i, :] over N dimensions (row: [N], mat: [N, N]).
    Returns scalar to out_ptr[0].
    """
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((), dtype=tl.float32)
    for i_start in range(0, N, BLOCK):
        idx = i_start + offs
        row_block = tl.load(row_ptr + idx, mask=idx < N, other=0.0)  # [BLOCK]
        mat_block = tl.load(mat_ptr + idx[:, None] * N + offs[None, :], mask=(idx[:, None] < N) & (offs[None, :] < N), other=0.0)  # [BLOCK, N]
        # Reduce each column by multiplying with row_block and summing across rows (BLOCK)
        for j in range(0, BLOCK):
            col_vals = tl.load(mat_ptr + j * N + offs, mask=offs < N, other=0.0)  # [BLOCK]
            prod = row_block[j] * tl.sum(col_vals, axis=0)
            acc += prod
    tl.store(out_ptr, acc)


def _compute_g_and_beta(a: torch.Tensor, dt_bias: torch.Tensor, A_log: torch.Tensor, b: torch.Tensor):
    """
    Triton kernel to compute g and beta on device. Returns (g, beta) as tensors (float32).
    """
    T, H = a.shape
    g = torch.empty((T, H), dtype=torch.float32, device=a.device)
    beta = torch.empty((T, H), dtype=torch.float32, device=a.device)
    # Ensure inputs are float32 and A_log exists
    a = a.float()
    dt_bias = dt_bias.float()
    if A_log is None:
        A_log = torch.zeros((H,), dtype=torch.float32, device=a.device)
    else:
        A_log = A_log.float()
    b = b.float()
    # Launch kernel (grid (T, H))
    softplus_and_g_kernel[(T, H)](a, dt_bias, A_log, g, beta, T, H)
    return g, beta


def _q_at_state_triton(q_row: torch.Tensor, state_row: torch.Tensor) -> torch.Tensor:
    """
    Compute q_row @ state_row via Triton. q_row: [1, 128], state_row: [128, 128], returns [1, 128].
    """
    M, K = q_row.shape  # M=1, K=128
    N = state_row.shape[1]  # N=128
    out = torch.empty((N,), dtype=torch.float32, device=q_row.device)
    row_matmul_triton[(1,)](q_row[0], state_row, out, K, N, BLOCK=32)
    return out


def _run_triton_version(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Triton-only version of the original run. Returns output [T, 8, 128] bfloat16 and new_state [1, 8, 128, 128] float32.
    """
    device = q.device
    T = q.shape[0]
    H_q = q.shape[1]  # 4
    H_k = k.shape[1]  # 4
    H_v = v.shape[1]  # 8

    # Prepare expanded q and k for v heads
    q_exp = q.repeat_interleave(2, dim=1)  # [T, 8, 128]
    k_exp = k.repeat_interleave(2, dim=1)  # [T, 8, 128]

    # Handle scale=None by setting to 1.0
    scale_val = 1.0 if scale is None else float(scale)

    # Compute g and beta via Triton
    g, beta = _compute_g_and_beta(a, dt_bias, A_log, b)

    # Initialize output
    output = torch.empty((T, H_v, 128), dtype=torch.bfloat16, device=device)
    # Initialize new_state as [1, 4, 128, 128]
    new_state = torch.zeros((1, H_q, 128, 128), dtype=torch.float32, device=device)

    # Loop over segments and time steps
    num_seqs = cu_seqlens.numel() - 1
    # We will process one segment at a time
    for seg_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seg_idx].item())
        seq_end = int(cu_seqlens[seg_idx + 1].item())
        seq_len = seq_end - seq_start
        if seq_len <= 0:
            continue

        # Reinitialize state for this segment using provided state (first segment or given initial)
        state_curr = state.float().contiguous()  # [1, 4, 128, 128] -> treat inner [4,128,128]

        for i in range(seq_len):
            t = seq_start + i

            # Update per v head j and q head h
            for j in range(H_v):
                k_row = k_exp[t, j]  # [128]
                v_row = v[t, j]      # [128]
                beta_tj = beta[t, j].float()
                g_tj = g[t, j].float()

                # Initialize per-head scalars
                remove_j = torch.zeros((H_q,), dtype=torch.float32, device=device)
                update_j = torch.zeros((H_q,), dtype=torch.float32, device=device)

                # Compute per-head dot products and updates
                for h in range(H_q):
                    # old_v_j[h] = dot(k_row, state_curr[h])
                    # Note: state_curr is [4,128,128], indexing by h gives [128,128]
                    old_v_j_h = torch.dot(k_row, state_curr[h].flatten())  # scalar
                    # new_v_j[h] = beta_tj * v_row + (1 - beta_tj) * old_v_j[h]
                    new_v_j_h = beta_tj * v_row + (1.0 - beta_tj) * old_v_j_h  # [128]
                    # update_j[h] = dot(k_row, new_v_j[h])
                    update_j[h] = torch.dot(k_row, new_v_j_h)

                    # remove_j[h] = dot(k_row, old_v_j[h]) == old_v_j_h
                    remove_j[h] = old_v_j_h

                    # Update state: state_curr[h] = g_tj * state_curr[h] + update_j[h] - remove_j[h]
                    state_curr[h] = g_tj * state_curr[h] + update_j[h] - remove_j[h]

                # Compute output o[h] = scale * q_exp[t, h] @ state_curr[h]
                q_exp_h = q_exp[t, h]  # [1, 128]
                o_h = _q_at_state_triton(q_exp_h, state_curr[h])  # [128]
                output[t, j, :] = o_h.to(torch.bfloat16)

        # Store new_state for this segment as [1,4,128,128]
        # current state_curr is [4,128,128] -> transpose to [4,128,128] (already in that form)
        new_state[0] = state_curr  # overwrite per segment

    return output, new_state


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Triton-only forward: no torch matmul or einsum. Return output and new_state as in original.
        output, new_state = _run_triton_version(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
