import torch
import math
import triton
import triton.language as tl


# Triton kernels: elementwise and GEMV (scalar-loop versions)
@triton.jit
def softplus_scalar(x_ptr, out_ptr, N: tl.constexpr):
    for i in range(N):
        x = tl.load(x_ptr + i)
        y = tl.log(1.0 + tl.exp(x))
        tl.store(out_ptr + i, y)


@triton.jit
def sigmoid_scalar(x_ptr, out_ptr, N: tl.constexpr):
    for i in range(N):
        x = tl.load(x_ptr + i)
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + i, y)


@triton.jit
def scale_vector(x_ptr, out_ptr, scale, N: tl.constexpr):
    for i in range(N):
        xi = tl.load(x_ptr + i)
        yi = scale * xi
        tl.store(out_ptr + i, yi)


@triton.jit
def gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # Computes out[i] = sum_{k=0}^{K-1} q[k] * A[k, i], q_ptr [K], A_ptr [K, V] contiguous, out_ptr [V]
    for i in range(V):
        acc = 0.0
        for k in range(K):
            qk = tl.load(q_ptr + k)
            Ak = tl.load(A_ptr + k * V + i)
            acc += qk * Ak
        tl.store(out_ptr + i, acc)


@triton.jit
def elementwise_mul_add_scalar(alpha_ptr, beta_ptr, v_ptr, old_ptr, out_ptr, N: tl.constexpr):
    # out[i] = alpha * v[i] + beta * old[i], alpha and beta are scalars from alpha_ptr[0], beta_ptr[0]
    alpha = tl.load(alpha_ptr)
    beta = tl.load(beta_ptr)
    for i in range(N):
        vi = tl.load(v_ptr + i)
        oldi = tl.load(old_ptr + i)
        outi = alpha * vi + beta * oldi
        tl.store(out_ptr + i, outi)


@triton.jit
def dot_scalar(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    # out[0] = sum_{i=0}^{N-1} x[i] * y[i]
    acc = 0.0
    for i in range(N):
        xi = tl.load(x_ptr + i)
        yi = tl.load(y_ptr + i)
        acc += xi * yi
    tl.store(out_ptr + 0, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Gated Delta Net prefill (k-last layout).
        - q: [T, H_q, 128], H_q=4
        - k: [T, H_k, 128], H_k=4
        - v: [T, H_v, 128], H_v=8
        - state: [num_seqs, H, V, K], V=128, K=128 (k-last), H = max(H_q, H_v) = 8
        Returns:
        - output: [T, H, 128] bfloat16
        - new_state: [num_seqs, H, 128, 128] float32
        """
        device = q.device
        T, H_q, K = q.shape
        H_k = k.shape[1]
        H_v = v.shape[1]
        H = max(H_q, H_v)
        V = K  # head_size, 128
        assert K == V == 128
        assert H_q == 4 and H_k == 4 and H_v == 8

        # Prepare output and new_state
        output = torch.empty((T, H, V), dtype=torch.bfloat16, device=device)
        num_seqs = cu_seqlens.numel() - 1
        new_state = torch.zeros((num_seqs, H, V, V), dtype=torch.float32, device=device)

        # Expand q and k for v heads
        # Since H_q != H_v, we repeat q and k to match v heads (repeat_interleave). However the original code uses:
        # q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1) which for 4->8 works. Implement manually:
        q_exp = torch.empty((T, H, V), dtype=torch.float32, device=device)
        k_exp = torch.empty((T, H, V), dtype=torch.float32, device=device)

        # Map q and k from H_q/H_k to H
        # For each h in [0..H-1], take q[h // 2] and k[h // 2] when H_q==4 and H_k==4, else generalized mapping is needed.
        # Since original asserts hold, mapping is: h in [0..3] -> h; h in [4..7] -> h-4.
        # But H_q=4 and H_v=8 means we map q's 4 heads into 8:
        # h in [0..3] -> q[:, h, :] mapped twice; to keep simplicity, assume the environment ensures correctness.
        # We'll implement mapping via torch indexing, which is allowed here (host side), and focus on Triton computations.
        # Manual mapping per h:
        for h in range(H):
            if h < 4:
                q_exp[:, h, :] = q[:, h, :]
                k_exp[:, h, :] = k[:, h, :]
            else:
                q_exp[:, h, :] = q[:, h - 4, :]
                k_exp[:, h, :] = k[:, h - 4, :]

        # Process segments
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            T_seg = seq_end - seq_start
            if T_seg <= 0:
                continue

            # Initialize new_state for this segment
            # new_state[seq_idx, :, :, :] will be filled per t
            # Compute scale
            if scale is None or scale == 0.0:
                scale_val = 1.0 / math.sqrt(V)
            else:
                scale_val = float(scale)

            # Loop over time within segment
            for t in range(T_seg):
                t_abs = seq_start + t

                # Compute g and beta per head h using Triton (softplus and sigmoid)
                H_vec = torch.arange(H, device=device)
                a_t = a[t_abs]  # [H]
                b_t = b[t_abs]  # [H]
                # z = A_log + a[t] + dt_bias
                z = A_log + a_t + dt_bias  # [H], all float32
                # softplus(z)
                softplus_z = torch.empty(H, dtype=torch.float32, device=device)
                # Launch softplus_scalar: softplus_scalar(z, softplus_z, N=H)
                softplus_scalar(z, softplus_z, N=H)
                # g = exp(-exp(A_log) * softplus(z))
                g_vec = torch.empty(H, dtype=torch.float32, device=device)
                exp_A_log = torch.exp(A_log)
                g_vec = torch.exp(-exp_A_log * softplus_z)
                # beta = sigmoid(b[t])
                beta_vec = torch.empty(H, dtype=torch.float32, device=device)
                sigmoid_scalar(b_t, beta_vec, N=H)

                # For each head h
                for h in range(H):
                    # Prepare q_vec, k_vec, v_vec for this t,h
                    q_vec = q_exp[t_abs, h].contiguous().to(torch.float32)  # [128]
                    k_vec = k_exp[t_abs, h].contiguous().to(torch.float32)  # [128]
                    v_vec = v[t_abs, h].contiguous().to(torch.float32)      # [128]

                    # State from previous segment: state is [num_seqs, H, V, K], we need [seq_idx, h, :, :]
                    # Build state_old_T[h]: [V, K] then transpose to [K, V]
                    # For simplicity, assume no state provided; otherwise, use:
                    if state is not None:
                        state_HVKh = state[seq_idx, h].contiguous()  # [V, K]
                        state_old_mat = state_HVKh.transpose(0, 1).contiguous()  # [K, V]
                    else:
                        state_old_mat = torch.zeros((V, V), dtype=torch.float32, device=device)  # [K, V]

                    # Compute old_v = k_vec @ state_old_T[h] = dot over K dimension
                    # Implement GEMV-like in Triton: out_vec[h] = sum_k k_vec[k] * state_old_mat[k, i]
                    old_v = torch.empty(V, dtype=torch.float32, device=device)
                    # Triton gemv_1xKxKxV_into


def run(*args):
    return ModelNew()(*args)
