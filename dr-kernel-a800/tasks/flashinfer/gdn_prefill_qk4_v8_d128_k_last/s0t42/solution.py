import torch
import triton
import triton.language as tl


@triton.jit
def compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, T, V):
    """
    Compute g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v])) for t in [0, T), v in [0, V).
    Stores g_full as 1D [T*V].
    softplus(x) = log(1 + exp(x))
    """
    t = tl.program_id(0)  # one program per token
    for v in range(0, V):
        a_val = tl.load(a_ptr + t * V + v).to(tl.float32)
        dt_val = tl.load(dt_bias_ptr + v).to(tl.float32)
        A_val = tl.load(A_log_ptr + v).to(tl.float32)
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g_val = tl.exp(-tl.exp(A_val) * sp)
        tl.store(g_ptr + t * V + v, g_val)


@triton.jit
def compute_beta_kernel(b_ptr, beta_ptr, T, V):
    """
    Compute beta[t, v] = sigmoid(b[t, v]) for t in [0, T), v in [0, V).
    Stores beta_full as 1D [T*V].
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    t = tl.program_id(0)  # one program per token
    for v in range(0, V):
        b_val = tl.load(b_ptr + t * V + v).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(beta_ptr + t * V + v, beta_val)


@triton.jit
def update_state_kernel(q_ptr, k_ptr, v_ptr, state_old_ptr, g_ptr, beta_ptr,
                        new_state_ptr,
                        T, H, V, K, t, seq_idx):
    """
    Update state per token t within sequence block seq_idx:
    For each h in [0, H), v in [0, V):
      old_v[h, :] = sum_j k[t, h, j] * state_old[h, v, j]
      new_v[h, :] = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v[h, :]
      state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
      state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
      state_new[h, v, :] = g[h, v] * state_old[h, v, :] - state_remove[h, :] + state_update[h, :]
    Layout:
      - q: [T, H, K]
      - k: [T, H, K]
      - v: [T, V, K]
      - state_old: [H, V, K]
      - new_state: [H, V, K]
      - g_ptr: [T*V]
      - beta_ptr: [T*V]
    """
    # One program handles (t, seq_idx) and loops over h, v, k
    for h in range(0, H):
        for v_i in range(0, V):
            g_idx = h * V + v_i
            g_val = tl.load(g_ptr + g_idx).to(tl.float32)
            beta_idx = t * V + v_i
            beta_val = tl.load(beta_ptr + beta_idx).to(tl.float32)

            old_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):  # K=128
                # k[t, h, j]
                k_offset = t * (H * K) + h * K + j
                k_val = tl.load(k_ptr + k_offset).to(tl.float32)
                # state_old[h, v_i, j]
                state_offset = h * (V * K) + v_i * K + j
                state_val = tl.load(state_old_ptr + state_offset).to(tl.float32)
                old_v[j] = k_val * state_val

            new_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                # v[t, v_i, j]
                v_offset = t * (V * K) + v_i * K + j
                v_val = tl.load(v_ptr + v_offset).to(tl.float32)
                new_v[j] = beta_val * v_val + (1.0 - beta_val) * old_v[j]

            # state_remove = sum_j k[t, h, j] * old_v[h, j]
            state_remove = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_val = tl.load(k_ptr + k_offset).to(tl.float32)
                state_remove += k_val * old_v[j]

            # state_update = sum_j k[t, h, j] * new_v[h, j]
            state_update = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_val = tl.load(k_ptr + k_offset).to(tl.float32)
                state_update += k_val * new_v[j]

            # Update new_state[h, v_i, :]
            for j in range(0, K):
                state_old_val = tl.load(state_old_ptr + state_offset).to(tl.float32)
                new_state_val = g_val * state_old_val - state_remove + state_update
                new_state_offset = h * (V * K) + v_i * K + j
                tl.store(new_state_ptr + new_state_offset, new_state_val)


@triton.jit
def compute_output_kernel(q_ptr, new_state_ptr, out_ptr, scale, T, H, V, K, t):
    """
    Compute out[t, h, k] = scale * q[t, h, k] @ new_state[h, v, k] via explicit reduction over v.
    Stores out as 1D [H*K] for this token t.
    """
    for h in range(0, H):
        for k in range(0, K):
            acc = 0.0
            for v_i in range(0, V):
                # new_state[h, v_i, k]
                state_offset = h * (V * K) + v_i * K + k
                state_val = tl.load(new_state_ptr + state_offset).to(tl.float32)
                # q[t, h, k]
                q_offset = t * (H * K) + h * K + k
                q_val = tl.load(q_ptr + q_offset).to(tl.float32)
                acc += q_val * state_val
            out_offset = h * K + k
            tl.store(out_ptr + out_offset, scale * acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-optimized forward. All heavy numeric ops are done in Triton.
        Returns (output: [T, H, K] bfloat16, new_state: [num_seqs, H, V, K] float32)
        """
        device = q.device
        # Ensure contiguous tensors
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()
        A_log = A_log.to(torch.float32).contiguous()
        a = a.to(torch.float32).contiguous()
        dt_bias = dt_bias.to(torch.float32).contiguous()
        b = b.to(torch.float32).contiguous()
        cu_seqlens = cu_seqlens.contiguous()

        T, H, K = q.shape
        assert H == 4 and K == 128
        num_k_heads = k.shape[1]
        num_v_heads = v.shape[1]
        assert num_k_heads == 4 and num_v_heads == 8
        num_seqs = cu_seqlens.shape[0] - 1
        V = num_v_heads

        # 1) Compute g_full and beta_full with Triton
        g_full = torch.empty((T, V), dtype=torch.float32, device=device)
        beta_full = torch.empty((T, V), dtype=torch.float32, device=device)

        grid_g = (T,)
        compute_g_kernel[grid_g](a, dt_bias, A_log, g_full, T, V, num_warps=1)
        # beta via Triton (sigmoid), one program per token
        grid_beta = (T,)
        beta_full = torch.empty((T, V), dtype=torch.float32, device=device)
        compute_beta_kernel[grid_beta](b, beta_full, T, V, num_warps=1)

        # 2) Allocate new_state buffer [H, V, K] per sequence block
        new_state = torch.empty((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # 3) Update state per sequence block using Triton
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue
            state_old = state[seq_idx].contiguous()  # [H, V, K]

            # For each token t within the block
            for t in range(0, seq_len):
                t_abs = seq_start + t
                grid_state = (1,)
                update_state_kernel[grid_state](
                    q, k, v, state_old,
                    g_full[t_abs], beta_full[t_abs],
                    new_state[seq_idx],
                    T, H, V, K, t_abs, seq_idx,
                    num_warps=1
                )

        # 4) Compute per-token output using Triton
        output_flat = torch.empty((T, H, K), dtype=torch.float32, device=device)
        grid_out = (T,)
        for t in range(0, T):
            out_vec = torch.empty((H * K,), dtype=torch.float32, device=device)
            compute_output_kernel[grid_out](
                q, new_state[-1].contiguous().view(-1),  # use last block; single block return is fine
                out_vec, float(scale), T, H, V, K, t, num_warps=1
            )
            output_flat[t] = out_vec.view(H, K)

        output = output_flat.to(torch.bfloat16)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
