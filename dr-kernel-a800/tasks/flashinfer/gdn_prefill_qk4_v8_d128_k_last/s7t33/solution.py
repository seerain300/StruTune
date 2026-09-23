import torch
import math
import triton
import triton.language as tl


@triton.jit
def _softplus_and_sigmoid(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, b_ptr, beta_ptr, T: tl.int32, H: tl.int32):
    """
    Triton kernel computing:
      g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
    Shapes:
      a: [T, H], dt_bias: [H], A_log: [H], g: [T, H], b: [T, H], beta: [T, H]
    """
    t = tl.program_id(0)
    j = tl.program_id(1)
    if t >= 0 and t < T and j >= 0 and j < H:
        sum_ = tl.load(a_ptr + t * H + j) + tl.load(dt_bias_ptr + j)
        soft = tl.log(1.0 + tl.exp(sum_))
        gval = tl.exp(-tl.exp(tl.load(A_log_ptr + j)) * soft)
        bval = tl.load(b_ptr + t * H + j)
        bival = 1.0 / (1.0 + tl.exp(-bval))
        tl.store(g_ptr + t * H + j, gval)
        tl.store(beta_ptr + t * H + j, bival)


@triton.jit
def _row_matmul(A_ptr, B_ptr, C_ptr, N: tl.int32):
    """
    Compute C = A @ B where:
      A is [1, N], B is [N, N], C is [1, N]
    We assume A is row 0. Use a single program and iterate over N in chunks.
    N is 128, so we handle it in one chunk or loop for generality.
    """
    offs = tl.arange(0, 32)
    acc = tl.zeros((32,), dtype=tl.float32)
    # A is [1, N], we load row 0
    # Iterate over K dimension in chunks of 32
    for k in range(0, N, 32):
        a = tl.load(A_ptr + k + offs)  # [32]
        b = tl.load(B_ptr + k + offs, mask=k + offs < N, other=0.0)  # [32]
        acc += a * b
    # Write results to C[0, :]
    for j in range(0, 32):
        if k + j < N:
            tl.store(C_ptr + j, acc[j])


@triton.jit
def _qmm_row(A_ptr, B_ptr, C_ptr, N: tl.int32):
    """
    Alias for _row_matmul; used for clarity in forward.
    """
    _row_matmul(A_ptr, B_ptr, C_ptr, N)


def _run_triton_version(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Triton-optimized forward. Returns output [T, 8, 128], bfloat16,
    and new_state [1, 8, 128, 128], float32.
    """
    device = q.device
    T = q.shape[0]
    H_q = q.shape[1]
    H_k = k.shape[1]
    H_v = v.shape[1]
    assert H_k == 4 and H_q == 4 and H_v == 8
    assert q.shape[2] == 128 and k.shape[2] == 128 and v.shape[2] == 128
    num_seqs = cu_seqlens.numel() - 1
    # Prepare expanded q/k for v heads
    q_exp = q.repeat_interleave(2, dim=1)  # [T, 8, 128]
    k_exp = k.repeat_interleave(2, dim=1)  # [T, 8, 128]

    # Compute g and beta via Triton
    g = torch.empty((T, H_v), dtype=torch.float32, device=device)
    beta = torch.empty((T, H_v), dtype=torch.float32, device=device)
    # Launch Triton kernel over (T, H_v)
    grid_g = (T, H_v)
    _softplus_and_sigmoid[grid_g](a, dt_bias, A_log, g, b, beta, T, H_v)

    # Output buffer
    out = torch.empty((T, H_v, 128), dtype=torch.bfloat16, device=device)

    # Maintain state as [H_q, 128, 128] float32; original state shape is [1,8,128,128], but we use [4,128,128].
    # For segment 0, initialize with provided state (shape [1,8,128,128] -> [4,128,128] by selecting heads).
    # To keep types consistent, use float32. If state is None, initialize zeros.
    state_curr = []
    if state is not None:
        # Extract first 4 heads from provided state of shape [1,8,128,128] -> [4,128,128]
        # state is [1,8,128,128]; select heads by index -> [4,128,128]
        state_curr = [state[0, h].float().contiguous() for h in range(H_q)]
    else:
        for h in range(H_q):
            state_curr.append(torch.zeros((128, 128), dtype=torch.float32, device=device))

    # Process segments
    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        for t in range(seq_start, seq_end):
            # Update and compute outputs for each v head j
            for j in range(H_v):
                # Compute new_v_j for each q head h
                new_v_j = []
                for h in range(H_q):
                    # old_v_j[h] = dot(k_exp[t, j, :], state_curr[h])
                    k_vec = k_exp[t, j]  # [128]
                    old_v_j = torch.dot(k_vec, state_curr[h].squeeze(0))  # [4,128,128] -> squeeze(0) to [128,128]
                    v_vec = v[t, j]  # [128]
                    new_v_j.append(beta[t, j] * v_vec + (1.0 - beta[t, j]) * old_v_j)
                # Update state_curr[h]
                g_tj = g[t, j]
                for h in range(H_q):
                    # update_j[h] = dot(k_exp[t, j, :], new_v_j[h])
                    k_vec = k_exp[t, j]  # [128]
                    new_v_j_h = new_v_j[h]  # [128]
                    remove_j = old_v_j  # already computed above per h
                    state_curr[h] = g_tj * state_curr[h] + (torch.dot(k_vec, new_v_j_h) - torch.dot(k_vec, remove_j))

                # Compute output for this v head: o[h] = scale * (q_exp[t, h, :] @ state_curr[h])
                # Launch Triton kernel for each h
                q_row = q_exp[t, h]  # [128]
                # B is [128, 128]; we use state_curr[h] as B by transposing to [128, 128]
                B = state_curr[h].transpose(0, 1).contiguous()  # [128, 128]
                C = torch.empty((128,), dtype=torch.float32, device=device)
                _qmm_row[(1,)](q_row, B, C, 128)
                out[t, j] = (scale * C).to(torch.bfloat16)

    # Construct new_state as [1, 8, 128, 128] from state_curr [4, 128, 128]
    new_state = torch.empty((1, H_v, 128, 128), dtype=torch.float32, device=device)
    for h in range(H_q):
        for j in range(H_v):
            new_state[0, j] = state_curr[h]

    return out, new_state


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-optimized forward that:
          - computes g and beta with Triton kernel
          - performs q@state per head per v using Triton kernel
          - updates state (dot products) in torch
          - returns output [T, 8, 128] bfloat16 and new_state [1, 8, 128, 128] float32
        """
        # Ensure scale is a float to avoid Triton NoneType errors
        scale = scale if scale is not None else 1.0
        out, new_state = _run_triton_version(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
        return out, new_state


def run(*args):
    return ModelNew()(*args)
