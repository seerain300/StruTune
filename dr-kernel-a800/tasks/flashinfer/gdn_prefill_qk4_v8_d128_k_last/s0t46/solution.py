import torch
import triton
import triton.language as tl


@triton.jit
def compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, T, V):
    """
    Compute g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v])) for all t, v.
    g_ptr: [T*V] float32
    T: total_seq_len, V: num_v_heads
    Note: a_ptr is [T*V], dt_bias_ptr and A_log_ptr are [V].
    """
    t = tl.program_id(0)  # one program per token
    for v in range(0, V):
        a_val = tl.load(a_ptr + t * V + v).to(tl.float32)
        dt_val = tl.load(dt_bias_ptr + v).to(tl.float32)
        A_val = tl.load(A_log_ptr + v).to(tl.float32)
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g_val = tl.exp(-tl.exp(A_val) * sp)
        idx = t * V + v
        tl.store(g_ptr + idx, g_val)


@triton.jit
def compute_beta_kernel(b_ptr, beta_ptr, T, V):
    """
    Compute beta[t, v] = sigmoid(b[t, v]) for all t, v.
    beta_ptr: [T*V] float32
    """
    t = tl.program_id(0)
    for v in range(0, V):
        b_val = tl.load(b_ptr + t * V + v).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        idx = t * V + v
        tl.store(beta_ptr + idx, beta_val)


@triton.jit
def update_state_kernel(q_ptr, k_ptr, v_ptr, state_old_ptr, g_ptr, beta_ptr, new_state_ptr,
                        T, H, V, K, t, seq_idx):
    """
    Update state for sequence block seq_idx, token t:
    For each h in [0, H), v in [0, V):
      old_v[h, :] = sum_j k[t, h, j] * state_old[h, v, j]
      new_v[h, :] = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v[h, :]
      state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
      state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
      state_new[h, v, :] = g[h, v] * state_old[h, v, :] - state_remove[h, :] + state_update[h, :]
    q_ptr: [T, H, K], k_ptr: [T, H, K], v_ptr: [T, V, K], state_old_ptr: [H, V, K], new_state_ptr: [H, V, K]
    Note: These are all flattened in a specific layout expected by the host. For simplicity, we assume contiguous
    and index linearly.
    """
    # Loop over h and v; compute per-k element updates
    for h in range(0, H):
        for v_i in range(0, V):
            # g_val and beta_val scalars
            g_idx = h * V + v_i
            g_val = tl.load(g_ptr + g_idx).to(tl.float32)
            beta_val = tl.load(beta_ptr + g_idx).to(tl.float32)

            # Load vectors:
            # old_v[h, k] = sum_j k[t, h, j] * state_old[h, v_i, j]
            old_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_elt = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)
                state_elt = tl.load(state_old_ptr + h * (V * K) + v_i * K + j).to(tl.float32)
                old_v += k_elt * state_elt

            # new_v[h, k] = beta[v_i] * v[t, v_i, k] + (1 - beta[v_i]) * old_v[k]
            new_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                vv_elt = tl.load(v_ptr + t * (V * K) + v_i * K + j).to(tl.float32)
                new_v[j] = beta_val * vv_elt + (1.0 - beta_val) * old_v[j]

            # state_remove[h, k] = sum_j k[t, h, j] * old_v[h, j]
            state_remove = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_elt = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)
                old_j = old_v[j]
                state_remove += k_elt * old_j

            # state_update[h, k] = sum_j k[t, h, j] * new_v[h, j]
            state_update = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_elt = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)
                new_j = new_v[j]
                state_update += k_elt * new_j

            # state_new[h, v_i, k] = g[h, v_i] * state_old[h, v_i, k] - state_remove + state_update
            for j in range(0, K):
                state_old_elt = tl.load(state_old_ptr + h * (V * K) + v_i * K + j).to(tl.float32)
                new_state_elt = g_val * state_old_elt - state_remove[j] + state_update[j]
                tl.store(new_state_ptr + h * (V * K) + v_i * K + j, new_state_elt)


@triton.jit
def compute_output_kernel(q_exp_ptr, new_state_ptr, out_ptr, scale, T, H, V, K):
    """
    Compute out[t, v, k] = scale * dot(q_exp[t, v, :], new_state[t, v, :, :]) for all t, v.
    q_exp_ptr: [T*V*K] float32
    new_state_ptr: [T*V*K*K] float32 (but we only use new_state[t, v, :, :] per t, v)
    out_ptr: [T*V*K] float32
    """
    t = tl.program_id(0)  # one program per token t
    for v in range(0, V):
        acc = tl.zeros((K,), dtype=tl.float32)
        # For each k, compute dot over K dimension using scalar loads for simplicity
        for k in range(0, K):
            q_vec_ptr = q_exp_ptr + t * (V * K) + v * K  # [K]
            for kk in range(0, K):
                q_elt = tl.load(q_vec_ptr + kk).to(tl.float32)
                # new_state[t, v, kk, :] is a K-vector starting at offset t*(V*K*K) + v*(K*K) + kk*K
                ns_base = t * (V * K * K) + v * (K * K) + kk * K
                ns_vec_ptr = new_state_ptr + ns_base  # [K]
                ns_vec = tl.zeros((K,), dtype=tl.float32)
                for jj in range(0, K):
                    ns_vec[jj] = tl.load(ns_vec_ptr + jj).to(tl.float32)
                acc[k] += q_elt * tl.sum(ns_vec)
        # scale * acc
        acc = acc * scale
        out_base = t * (V * K)
        out_vec_ptr = out_ptr + out_base + v * K
        for kk in range(0, K):
            tl.store(out_vec_ptr + kk, acc[kk])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Shapes based on original assertions:
        total_seq_len, num_q_heads, head_size = q.shape  # T, H=4, K=128
        _, num_v_heads, _ = v.shape  # V=8
        num_k_heads = k.shape[1]
        device = q.device

        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8 and head_size == 128
        H = 4
        V = 8
        K = 128

        # Ensure contiguity and float32 compute
        q = q.contiguous().float()
        k = k.contiguous().float()
        v = v.contiguous().float()
        if state is not None:
            state = state.contiguous().float()
        a = a.contiguous().float()  # [T, V]
        dt_bias = dt_bias.contiguous().float()  # [V]
        b = b.contiguous().float()  # [T, V]
        A_log = A_log.contiguous().float()  # [V]

        # Compute g and beta (T*V)
        T = total_seq_len
        g_flat = torch.empty((T * V), dtype=torch.float32, device=device)
        beta_flat = torch.empty((T * V), dtype=torch.float32, device=device)

        grid_g = (T,)
        compute_g_kernel[grid_g](a, dt_bias, A_log, g_flat, T, V, num_warps=1)
        grid_beta = (T,)
        compute_beta_kernel[b](b, beta_flat, T, V, num_warps=1)

        # Prepare expanded q and k for output: q_exp, k_exp are [T, V, K]
        # The original code performs q.repeat_interleave(num_v_heads//num_q_heads, dim=1)
        # Since num_v_heads//num_q_heads == 2, we don't have to repeat; q and k are already [T,4,128].
        # But to match the original procedure, we can explicitly expand: create q_exp as q with v-dimension V by indexing.
        # Here, we assume q,k are already [T,4,128]. We will launch update_state and output kernels directly.

        # Allocate new_state [num_seqs, V, K, K]
        num_seqs = cu_seqlens.size(0) - 1
        new_state = torch.zeros((num_seqs, V, K, K), dtype=torch.float32, device=device)

        # Perform state updates per sequence block
        for seq_idx in range(0, num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Initialize state for this block: take state if provided, else zeros; layout expected is [H,V,K] in original code,
            # but here we use [H,V,K] for compute and return [num_seqs,V,K,K]. We'll assume state is [H,V,K] and treat it as such.
            # However, original state is [H,V,K] per block. We need to use it in update. To be safe, we set state_old from state
            # if state is not None for this seq_idx. Otherwise, zeros.
            if state is not None and seq_idx < state.shape[0]:
                # state[seq_idx] has shape [H,V,K]; flatten to [H*V*K] for kernel (we'll pass pointers directly with strides)
                # But Triton expects contiguous. Convert to [H*V*K] contiguous
                state_old = state[seq_idx].contiguous().view(H * V * K)
            else:
                state_old = torch.zeros((H * V * K,), dtype=torch.float32, device=device)

            # Update state for all tokens in this block
            for t in range(seq_start, seq_end):
                # We need to pass pointers for q row t and k row t as [H,K] contiguous vectors
                q_t = q[t].contiguous().view(H * K)  # [H*K] contiguous
                k_t = k[t].contiguous().view(H * K)  # [H*K] contiguous
                v_t = v[t].contiguous().view(V * K)  # [V*K] contiguous

                # Launch update kernel
                update_state_kernel[(1,)](q_t, k_t, v_t, state_old, g_flat, beta_flat, new_state[seq_idx].view(-1), T, H, V, K, t, seq_idx, num_warps=1)

                # Update state_old for next iteration (in-place update)
                # state_old is same as new_state at the end for next token within the same block
                state_old = new_state[seq_idx].contiguous().view(H * V * K)

        # Compute output: out[t, v, k] = scale * q_exp[t, v, k] @ new_state[t, v, :, :]
        # Create q_exp and k_exp explicitly as [T,V,K] using indexing from q and k (original repeats along dim=1 with ratio V/H).
        # Since original q,k are [T,4,128], and V=8, we need to expand along dim=1. We can construct q_exp by taking q[:, :, :] and
        # using v dimension indices. However, since original q_exp is repeat_interleave by factor 2, we can simply index q[:, v_h, :]
        # by mapping v in {0,1,2,3} -> (v//2, ...). To be precise, we create q_exp by expanding q along dim=1 with 2 repeats, but here
        # we don't have repeat. Given the original uses num_v_heads//num_q_heads == 2, and H=4,V=8, we can reconstruct q_exp as:
        # q_exp[v] = q[:, v//2, :] for v in [0..7], which is exactly repeating q's 4 heads into 8. We'll implement this in Triton by
        # building q_exp_ptr as [T*V*K] and k_exp analogously.

        # We'll build q_exp and k_exp in host for correctness (small tensors), then use Triton to compute output.
        q_exp = torch.empty((T, V, K), dtype=torch.float32, device=device)
        k_exp = torch.empty((T, V, K), dtype=torch.float32, device=device)
        # Map: for v in 0..7, use q[:, v//2, :]
        for t in range(T):
            for v_i in range(V):
                base = q[t, v_i // 2, :].contiguous()  # [K]
                q_exp[t, v_i, :] = base
                base_k = k[t, v_i // 2, :].contiguous()  # [K]
                k_exp[t, v_i, :] = base_k

        # Allocate output [T, V, K]
        out = torch.empty((T, V, K), dtype=torch.float32, device=device)
        out_flat = out.view(-1)  # [T*V*K]
        grid_out = (T,)
        compute_output_kernel[grid_out](q_exp.view(-1), new_state[-1].view(-1), out_flat, float(scale), T, H, V, K, num_warps=1)

        output = out.to(torch.bfloat16)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
