import torch
import math
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernels: simplified and robust (scalar-accumulation or explicit vectorized over 128)
@triton.jit
def _scale_vector(v_ptr, out_ptr, alpha, N: tl.constexpr):
    # out[i] = alpha * v[i], elementwise over N (N=128 here)
    for i in range(0, N):
        vi = tl.load(v_ptr + i)
        tl.store(out_ptr + i, vi * alpha)


@triton.jit
def _elementwise_add(v_ptr, old_ptr, out_ptr, alpha, beta, N: tl.constexpr):
    # out[i] = alpha * v[i] + beta * old[i], elementwise over N (N=128)
    for i in range(0, N):
        vi = tl.load(v_ptr + i)
        oldi = tl.load(old_ptr + i)
        tl.store(out_ptr + i, vi * alpha + oldi * beta)


@triton.jit
def _gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # Computes out[i] = sum_k q[k] * A[k, i] for i in 0..V-1
    # A is laid out as [K, V] contiguous: row-major, stride(0)=V, stride(1)=1
    # q is [K] contiguous
    # out is [V]
    for i in range(0, V):
        acc = tl.zeros((), dtype=tl.float32)
        for k in range(0, K):
            qk = tl.load(q_ptr + k)
            aki = tl.load(A_ptr + k * V + i)
            acc += qk * aki
        tl.store(out_ptr + i, acc)


@triton.jit
def _gemv_1xVxK_into_1xK(q_ptr, A_ptr, out_ptr, V: tl.constexpr, K: tl.constexpr):
    # Computes out[k] = sum_i q[i] * A[i, k] for k in 0..K-1
    # A is laid out as [V, K] contiguous: row-major, stride(0)=K, stride(1)=1
    # q is [V] contiguous
    for k in range(0, K):
        acc = tl.zeros((), dtype=tl.float32)
        for i in range(0, V):
            qi = tl.load(q_ptr + i)
            aik = tl.load(A_ptr + i * K + k)
            acc += qi * aik
        tl.store(out_ptr + k, acc)


@triton.jit
def _dot_scalar(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    # Computes scalar = sum(x[i] * y[i]) for i in 0..N-1
    acc = tl.zeros((), dtype=tl.float32)
    for i in range(0, N):
        xi = tl.load(x_ptr + i)
        yi = tl.load(y_ptr + i)
        acc += xi * yi
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        q: [T, 4, 128], bfloat16
        k: [T, 4, 128], bfloat16
        v: [T, 8, 128], bfloat16
        state: [1, 8, 128, 128], float32 (can be None)
        A_log: [8], float32
        a: [T, 8], bfloat16
        dt_bias: [8], float32
        b: [T, 8], bfloat16
        cu_seqlens: [num_seqs+1], int64 (cumsum of sequence lengths)
        scale: float32 scalar (if None or 0, we use 1/sqrt(head_size)=1/sqrt(128))
        Returns: (output [T, 8, 128], new_state [num_seqs, 8, 128, 128])
        """
        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        num_sab_heads = max(num_q_heads, num_v_heads)
        num_seqs = cu_seqlens.size(0) - 1
        device = q.device

        # Compute scale once (avoid torch.sqrt in hot path)
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(head_size)

        # Repeat q,k along the head dimension (num_v_heads // num_q_heads == 2)
        q_rep = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [T, 8, 128]
        k_rep = k.repeat_interleave(num_k_heads // num_q_heads, dim=1)  # [T, 8, 128]

        # Prepare output [T, 8, 128] bfloat16
        output = torch.empty((total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device)

        # Prepare new_state [num_seqs, 8, 128, 128] float32
        new_state = torch.empty((num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device)

        # Compute per-segment g and beta using PyTorch (one-time per segment), Triton will use them
        # softplus(x) = log(1 + exp(x)), sigmoid(x) = 1 / (1 + exp(-x))
        # We need to build x = a + dt_bias per (t, h), with dt_bias broadcast over T.
        # However, to keep Triton for the hot path, we compute g and beta on host once per segment.
        # Note: This is allowed and not in the inner loop per (t, h).
        g_list = []
        beta_list = []
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            a_seg = a[seq_start:seq_start + seq_len, :]  # [seq_len, 8]
            b_seg = b[seq_start:seq_start + seq_len, :]  # [seq_len, 8]
            x = (a_seg.float() + dt_bias.float().unsqueeze(0)).to(torch.float32)  # [seq_len, 8]
            g_seg = torch.exp(-torch.exp(A_log.float().unsqueeze(0).expand(seq_len, -1)) * torch.nn.functional.softplus(x))  # [seq_len, 8]
            beta_seg = torch.sigmoid(b_seg.float())  # [seq_len, 8]
            g_list.append(g_seg)
            beta_list.append(beta_seg)

        # For each sequence segment
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start

            if seq_len <= 0:
                continue

            # Initialize state_HKV for this segment for all heads
            state_HKV = torch.zeros((num_sab_heads, head_size, head_size), dtype=torch.float32, device=device)

            for t in range(seq_len):
                t_abs = seq_start + t

                # Gather q_vec, k_vec, v_vec
                q_vec = q_rep[t_abs].contiguous().to(torch.float32)  # [128]
                k_vec = k_rep[t_abs].contiguous().to(torch.float32)  # [128]
                v_vec = v[t_abs].contiguous().to(torch.float32)      # [128]

                # For each head h
                for h in range(num_sab_heads):
                    # Compute old_v = k @ state_old_T where state_old_T is [128, 128]
                    state_T = state_HKV[h]  # [128, 128]
                    out_v = torch.empty((head_size,), dtype=torch.float32, device=device)

                    # GEMV: out_v[i] = sum_k q_vec[k] * state_T[k, i]
                    _gemv_1xKxKxV_into_1xV(q_vec, state_T, out_v, 128, 128)
                    old_v = out_v  # [128]

                    # new_v = beta[t, h] * v_vec + (1 - beta[t, h]) * old_v
                    beta_t_h = beta_list[seq_idx][t, h].item()
                    g_t_h = g_list[seq_idx][t, h].item()
                    one_minus_beta = 1.0 - beta_t_h

                    # Compute new_v using Triton elementwise add (alpha = beta, beta = 1-beta)
                    new_v = torch.empty((head_size,), dtype=torch.float32, device=device)
                    _elementwise_add(v_vec, old_v, new_v, beta_t_h, one_minus_beta, 128)

                    # Compute state_remove = dot(k_vec, old_v) and state_update = dot(k_vec, new_v)
                    state_remove = torch.empty((), dtype=torch.float32, device=device)
                    _dot_scalar(k_vec, old_v, state_remove, 128)
                    state_update = torch.empty((), dtype=torch.float32, device=device)
                    _dot_scalar(k_vec, new_v, state_update, 128)
                    delta = state_update - state_remove  # scalar

                    # Update state_T in-place: state_new_T = g[t, h] * state_T + delta broadcasted
                    state_new_T = state_T * g_t_h + delta  # [128, 128]
                    state_HKV[h] = state_new_T

                    # Output: output[t, h, :] = scale * (q_vec @ state_new_T)
                    out_out = torch.empty((head_size,), dtype=torch.float32, device=device)
                    _gemv_1xKxKxV_into_1xV(q_vec, state_new_T, out_out, 128, 128)
                    out_vec = (out_out * float(scale)).to(torch.bfloat16)
                    output[t_abs, h, :] = out_vec

            # Store new_state for this segment
            for h in range(num_sab_heads):
                new_state[seq_idx, h, :, :] = state_HKV[h].transpose(0, 1)  # [128, 128] -> [128, 128]

        return output, new_state


def run(*args):
    return ModelNew()(*args)
