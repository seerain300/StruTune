import torch
import triton
import triton.language as tl


@triton.jit
def compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, T, V):
    """
    Compute g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v])) for all t, v.
    g_ptr: [T*V] float32
    softplus(x) = log(1 + exp(x))
    """
    t = tl.program_id(0)  # one program per token t
    for v in range(0, V):
        a_val = tl.load(a_ptr + t * V + v).to(tl.float32)
        dt_val = tl.load(dt_bias_ptr + v).to(tl.float32)
        A_val = tl.load(A_log_ptr + v).to(tl.float32)
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))  # softplus
        g_val = tl.exp(-tl.exp(A_val) * sp)
        idx = t * V + v
        tl.store(g_ptr + idx, g_val)


@triton.jit
def compute_beta_kernel(b_ptr, beta_ptr, T, V):
    """
    Compute beta[t, v] = sigmoid(b[t, v]) for all t, v.
    beta_ptr: [T*V] float32
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    t = tl.program_id(0)
    for v in range(0, V):
        b_val = tl.load(b_ptr + t * V + v).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        idx = t * V + v
        tl.store(beta_ptr + idx, beta_val)


@triton.jit
def repeat_interleave_1d(x_ptr, out_ptr, T, V, K):
    """
    Repeat-interleave along dim=1 from [T, K] to [T, V*K]:
    For each t and v in [0,V), out[t, v*K + k] = x[t, k].
    """
    t = tl.program_id(0)
    for v in range(0, V):
        base = t * K
        for kk in range(0, K):
            val = tl.load(x_ptr + base + kk).to(tl.float32)
            out_idx = t * (V * K) + v * K + kk
            tl.store(out_ptr + out_idx, val)


@triton.jit
def update_state_kernel(q_ptr, k_ptr, v_ptr, state_old_ptr, g_ptr, beta_ptr,
                        new_state_ptr,
                        T, H, V, K, t, seq_idx):
    """
    Update state per token t within sequence block seq_idx:
    For each h in [0, H), v in [0, V):
      old_v[h, :] = sum_j k[t, h, j] * state_old[h, v, j]   (length K)
      new_v[h, :] = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v[h, :]
      state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
      state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
      state_new[h, v, :] = g[h, v] * state_old[h, v, :] - state_remove[h, :] + state_update[h, :]
    """
    for h in range(0, H):
        for v_i in range(0, V):
            # Scalars g and beta for this (h, v)
            g_idx = h * V + v_i
            g_val = tl.load(g_ptr + g_idx).to(tl.float32)
            beta_val = tl.load(beta_ptr + g_idx).to(tl.float32)

            # old_v and new_v as vectors of length K
            old_v = tl.zeros((K,), dtype=tl.float32)
            new_v = tl.zeros((K,), dtype=tl.float32)

            # Compute old_v = k[t, h, :] dot state_old[h, v, :]
            for j in range(0, K):
                k_j = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)
                s_j = tl.load(state_old_ptr + (h * V * K) + v_i * K + j).to(tl.float32)
                old_v[j] = k_j * s_j

            # new_v = beta * v[t, v, :] + (1 - beta) * old_v
            for j in range(0, K):
                vv_j = tl.load(v_ptr + t * (V * K) + v_i * K + j).to(tl.float32)
                new_v[j] = beta_val * vv_j + (1.0 - beta_val) * old_v[j]

            # state_remove and state_update (scalar)
            remove = 0.0
            update = 0.0
            for j in range(0, K):
                k_j = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)
                remove += k_j * old_v[j]
                update += k_j * new_v[j]

            # Update new_state[h, v, :]
            for j in range(0, K):
                s_old_j = tl.load(state_old_ptr + (h * V * K) + v_i * K + j).to(tl.float32)
                new_state_val = g_val * s_old_j - remove + update
                tl.store(new_state_ptr + ((seq_idx * H * V) + h * V + v_i) * K + j, new_state_val)


@triton.jit
def compute_output_kernel(q_exp_ptr, new_state_ptr, out_ptr, scale, T, H, V, K):
    """
    Compute output[t, v, k] = scale * q_exp[t, v, k] @ new_state[t, v, k] for all t, v.
    q_exp has shape [T, V, K]; new_state has shape [T, V, K, K] (we pass it as [T, V, K, K]).
    For each t, v:
      out[t, v, :] = scale * q_exp[t, v, :] @ new_state[t, v, :, :]
    """
    t = tl.program_id(0)
    for v in range(0, V):
        # Load q_exp[t, v, :] vector
        q_vec = tl.zeros((K,), dtype=tl.float32)
        base_q = t * (V * K)
        for j in range(0, K):
            q_vec[j] = tl.load(q_exp_ptr + base_q + v * K + j).to(tl.float32)

        # Load new_state[t, v, :, :] as matrix [K, K]
        A = tl.zeros((K, K), dtype=tl.float32)
        for j in range(0, K):
            for r in range(0, K):
                A[j, r] = tl.load(new_state_ptr + (t * V * K * K) + (v * K * K) + j * K + r).to(tl.float32)

        # out_vec = scale * q_vec @ A
        out_vec = tl.zeros((K,), dtype=tl.float32)
        for j in range(0, K):
            for r in range(0, K):
                out_vec[j] += A[j, r] * q_vec[r]
        out_vec = out_vec * scale

        # Store out[t, v, :]
        out_base = t * (V * K)
        for j in range(0, K):
            tl.store(out_ptr + out_base + v * K + j, out_vec[j])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-only implementation. Keeps all heavy computation in Triton kernels.
        """
        device = q.device
        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        a = a.contiguous()
        b = b.contiguous()
        dt_bias = dt_bias.contiguous()
        A_log = A_log.contiguous()
        if state is not None:
            state = state.contiguous()

        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        assert num_q_heads == 4
        assert num_k_heads == 4
        assert num_v_heads == 8
        assert head_size == 128
        H = num_q_heads
        V = num_v_heads
        K = head_size

        # Prepare flattened g and beta
        g_flat = torch.empty((total_seq_len * V,), dtype=torch.float32, device=device)
        beta_flat = torch.empty((total_seq_len * V,), dtype=torch.float32, device=device)

        # Kernel 1: compute g
        grid_g = (total_seq_len,)
        compute_g_kernel[grid_g](a, dt_bias, A_log, g_flat, total_seq_len, V, num_warps=1)
        # Kernel 2: compute beta
        grid_beta = (total_seq_len,)
        compute_beta_kernel[b](b, beta_flat, total_seq_len, V, num_warps=1)

        # Expand q and k to [T, V, K] via repeat_interleave on dim=1
        q_exp = torch.empty((total_seq_len, V, K), dtype=torch.float32, device=device)
        k_exp = torch.empty((total_seq_len, V, K), dtype=torch.float32, device=device)

        grid_rep = (total_seq_len,)
        repeat_interleave_1d[grid_rep](q.view(-1, K), q_exp.view(-1), total_seq_len, V, K, num_warps=1)
        repeat_interleave_1d[grid_rep](k.view(-1, K), k_exp.view(-1), total_seq_len, V, K, num_warps=1)

        # Prepare new_state for each sequence block: [num_seqs, H, V, K]
        num_seqs = cu_seqlens.numel() - 1
        new_state = torch.zeros((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # Perform per-token updates in sequence blocks
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Initialize state for this block: [H, V, K] (k-last: last dim is K)
            if state is None or seq_idx >= state.shape[0]:
                state_hvk = torch.zeros((H, V, K), dtype=torch.float32, device=device)
            else:
                state_hvk = state[seq_idx]  # [H, V, K], k-last

            for i in range(seq_len):
                t = seq_start + i
                # Launch Triton kernel to update state for this token t within this seq_idx
                grid_up = (1,)
                update_state_kernel[grid_up](
                    q_exp[t], k_exp[t], v[t], state_hvk, g_flat, beta_flat, new_state[seq_idx],
                    total_seq_len, H, V, K, t, seq_idx, num_warps=1
                )
                # Update state_hvk for next iteration
                state_hvk = new_state[seq_idx].clone()

        # Compute output: per token t, output[t, v, k] = scale * q_exp[t, v, k] @ new_state[t, v, k]
        out = torch.empty((total_seq_len, V, K), dtype=torch.float32, device=device)
        grid_out = (total_seq_len,)
        compute_output_kernel[grid_out](q_exp, new_state[-1], out, float(scale), total_seq_len, H, V, K, num_warps=1)

        # Return output as bfloat16 and new_state
        output = out.to(torch.bfloat16)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
