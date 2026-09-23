import torch
import math

import triton
import triton.language as tl


# Triton elementwise and reduction kernels. No tl.constexpr for sizes.
@triton.jit
def _softplus_vector(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    # softplus(x) = log(1 + exp(x)) for N elements
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _sigmoid_vector(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    # sigmoid(x) = 1 / (1 + exp(-x)) for N elements
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _elementwise_mul_add_scalar(v_ptr, old_ptr, out_ptr, alpha, beta, N, BLOCK: tl.constexpr):
    # out_vec = alpha * v_vec + beta * old_v_vec for N elements
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
    old = tl.load(old_ptr + offsets, mask=mask, other=0.0)
    out = alpha * v + beta * old
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K, V, BLOCK: tl.constexpr):
    # A_ptr: [K, V] contiguous. q_ptr: [K] contiguous. out_ptr: [V]
    # out[i] = sum_k q[k] * A[k, i]
    pid = tl.program_id(axis=0)  # single-dimension, but we write V; grid can be 1
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    k = 0
    while k < K:
        qk = tl.load(q_ptr + k)
        a_row_ptrs = A_ptr + k * V + tl.arange(0, BLOCK)
        mask = tl.arange(0, BLOCK) < V
        a_row = tl.load(a_row_ptrs, mask=mask, other=0.0)
        acc += qk * a_row
        k += 1
    tl.store(out_ptr + tl.arange(0, BLOCK), acc, mask=tl.arange(0, BLOCK) < V)


@triton.jit
def _gemv_1xVxK_into_1xK(q_ptr, A_ptr, out_ptr, V, K, BLOCK: tl.constexpr):
    # A_ptr: [V, K] contiguous. q_ptr: [V] contiguous. out_ptr: [K]
    # out[k] = sum_i q[i] * A[i, k]
    pid = tl.program_id(axis=0)  # grid can be 1, since we write K
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    i = 0
    while i < V:
        qi = tl.load(q_ptr + i)
        a_col_ptrs = A_ptr + i * K + tl.arange(0, BLOCK)
        mask = tl.arange(0, BLOCK) < K
        a_col = tl.load(a_col_ptrs, mask=mask, other=0.0)
        acc += qi * a_col
        i += 1
    tl.store(out_ptr + tl.arange(0, BLOCK), acc, mask=tl.arange(0, BLOCK) < K)


@triton.jit
def _dot_scalar(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    # Compute dot = sum_{i=0..N-1} x[i] * y[i]
    pid = tl.program_id(axis=0)
    acc = tl.zeros((), dtype=tl.float32)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    prod = x * y
    acc += tl.sum(prod, axis=0)
    tl.store(out_ptr, acc)


# Host-side forward: Triton-only computation
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure CUDA tensors
        device = q.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors"
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        H = max(num_q_heads, num_v_heads)  # num_sab_heads
        K = head_size  # assert K == head_size == 128 in original, but we handle any K in Triton loops
        V = head_size  # same as K here

        # num_seqs derived from cu_seqlens
        num_seqs = cu_seqlens.numel() - 1

        # Output tensors
        output = torch.empty(
            (total_seq_len, H, V), dtype=torch.bfloat16, device=device
        )
        new_state = torch.empty(
            (num_seqs, H, V, V), dtype=torch.float32, device=device
        )

        # Compute scale
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(K)

        # Iterate over segments
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start

            # Initialize new_state for this segment with zeros
            # state layout: [H, V, K] for state, and [seq_idx, H, V, V] for new_state
            # The original 'state' is [num_seqs, H, V, K] if provided; if None, start from zero
            state_curr = None
            if state is not None and seq_idx < state.shape[0]:
                state_curr = state[seq_idx].float()  # [H, V, K]
            else:
                state_curr = torch.zeros((H, V, K), dtype=torch.float32, device=device)

            # Expand q and k for v-heads
            q_exp = q[seq_start:seq_end].repeat_interleave(num_v_heads // num_q_heads, dim=1).contiguous()
            k_exp = k[seq_start:seq_end].repeat_interleave(num_k_heads // num_q_heads, dim=1).contiguous()

            # Loop over time steps and heads
            for t in range(seq_len):
                for h in range(H):
                    # Extract vectors
                    q_vec = q_exp[seq_start + t, h].float()   # [K]
                    k_vec = k_exp[seq_start + t, h].float()   # [K]
                    v_vec = v[seq_start + t, h].float()       # [V]

                    # Get state_old_T: [V, K] = state_curr[h, :, :]
                    state_old_T = state_curr[h].permute(1, 2).contiguous()  # [V, K]
                    # Compute old_v = k_vec @ state_old_T  => [K]
                    old_v = torch.empty((K,), dtype=torch.float32, device=device)
                    _gemv_1xKxKxV_into_1xV(k_vec, state_old_T, old_v, K, V, 128)

                    # Compute beta[h] and g[h] using Triton elementwise kernels
                    # For beta: beta = sigmoid(b[t, h])
                    beta_scalar = 1.0 / (1.0 + torch.exp(-b[seq_start + t, h].float()))
                    # For g: g = exp(-exp(A_log[h]) * softplus(a[t, h] + dt_bias[h]))
                    # softplus(x) = log(1 + exp(x))
                    softplus_arg = a[seq_start + t, h].float() + dt_bias.float()[h]
                    softplus_val = torch.log(1.0 + torch.exp(softplus_arg))
                    g_scalar = torch.exp(-torch.exp(A_log.float()[h]) * softplus_val)

                    # new_v_vec = beta * v_vec + (1 - beta) * old_v
                    new_v_vec = torch.empty((V,), dtype=torch.float32, device=device)
                    _elementwise_mul_add_scalar(v_vec, old_v, new_v_vec, beta_scalar, 1.0 - beta_scalar, V, 128)

                    # Compute state_remove = dot(k_vec, old_v)
                    state_remove = torch.empty((), dtype=torch.float32, device=device)
                    _dot_scalar(k_vec, old_v, state_remove, K, 128)

                    # Compute state_update = dot(k_vec, new_v_vec)
                    state_update = torch.empty((), dtype=torch.float32, device=device)
                    _dot_scalar(k_vec, new_v_vec, state_update, K, 128)

                    delta = state_update[0] - state_remove[0]  # scalar
                    # state_new_T[h, :, :] = g * state_old_T + delta
                    state_new_T = (g_scalar * state_old_T) + delta  # broadcasting over [V, K]

                    # Compute output_vec = scale * (q_vec @ state_new_T)
                    out_vec = torch.empty((K,), dtype=torch.float32, device=device)
                    _gemv_1xVxK_into_1xK(q_vec, state_new_T, out_vec, V, K, 128)

                    # Store output[t, h, :]
                    output[seq_start + t, h] = (out_vec * scale).to(torch.bfloat16)

                    # Update new_state[seq_idx, h, :, :] = state_new_T.transpose(0,1)
                    # state_new_T is [V, K]; we need [K, V]
                    new_state[seq_idx, h] = state_new_T.transpose(0, 1).contiguous()

        return output, new_state


def run(*args):
    return ModelNew()(*args)
