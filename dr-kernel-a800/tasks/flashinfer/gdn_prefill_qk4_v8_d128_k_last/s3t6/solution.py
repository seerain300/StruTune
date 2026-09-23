import torch
import math
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernels for elementwise ops (to be launched from host)
@triton.jit
def _softplus_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes softplus(x) = log1p(exp(x)) elementwise, vectorized
    i = tl.arange(0, 128)  # N can be passed as constexpr; here we assume N=128 for head_size
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def _sigmoid_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Sigmoid(x) = 1 / (1 + exp(-x))
    i = tl.arange(0, 128)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def _scale_vector(x_ptr, out_ptr, scale, N: tl.constexpr):
    # out = scale * x
    i = tl.arange(0, 128)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = scale * x
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def _subtract_scalar_from_matrix(A_ptr, alpha, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # For each element (k, j) in matrix A of shape [K, V], out[k, j] = A[k, j] - alpha
    # We launch a 2D grid over (k, j) tiles.
    pid_k = tl.program_id(0)
    pid_j = tl.program_id(1)
    kk = pid_k * 64 + tl.arange(0, 64)  # tile along K
    jj = pid_j * 64 + tl.arange(0, 64)  # tile along V
    mask_k = kk < K
    mask_j = jj < V
    # Create 2D indices
    rows = kk[:, None]  # [64, 1]
    cols = jj[None, :]  # [1, 64]
    mask = mask_k[:, None] & mask_j[None, :]
    A = tl.load(A_ptr + rows * V + cols, mask=mask, other=0.0)
    A = A - alpha
    tl.store(out_ptr + rows * V + cols, A, mask=mask)


# Triton kernel for GEMV: computes out_vec[i] = sum_k q_vec[k] * A_mat[k, i]
@triton.jit
def _gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    i = tl.arange(0, 128)
    acc = tl.zeros([128], dtype=tl.float32)
    # Loop over K dimension (128)
    for k in range(0, 128):
        # Load q[k]
        qk = tl.load(q_ptr + k)
        # Load A[k, i]
        a = tl.load(A_ptr + k * V + i, mask=i < V, other=0.0)
        acc += qk * a
    tl.store(out_ptr + i, acc, mask=i < V)


# Triton kernel: elementwise vector op out_vec = beta * v_vec + (1 - beta) * old_v_vec
@triton.jit
def _elementwise_scalar_mul_add(v_ptr, old_ptr, out_ptr, beta, N: tl.constexpr):
    i = tl.arange(0, 128)
    v = tl.load(v_ptr + i, mask=i < N, other=0.0)
    old = tl.load(old_ptr + i, mask=i < N, other=0.0)
    out = beta * v + (1.0 - beta) * old
    tl.store(out_ptr + i, out, mask=i < N)


# Triton kernel: dot product of two 1xK vectors (k_vec and x_vec), returns scalar
@triton.jit
def _dot_scalar_1xK(k_ptr, x_ptr, out_ptr, K: tl.constexpr):
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, 128):
        vk = tl.load(x_ptr + k)
        kk = tl.load(k_ptr + k)
        acc += vk * kk
    tl.store(out_ptr, acc)


# ModelNew: Triton-orchestrated forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Shapes and constraints (same as original)
        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        assert num_q_heads == 4
        assert num_k_heads == 4
        assert num_v_heads == 8
        assert head_size == 128

        # Compute scale (float32 scalar)
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(head_size)

        # Repeat q and k along heads (data movement, not computation)
        q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
        k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

        # Allocate outputs
        output = torch.empty(
            (total_seq_len, num_v_heads, head_size), dtype=torch.float32, device=q.device
        )
        new_state = torch.empty(
            (cu_seqlens.shape[0] - 1, num_v_heads, head_size, head_size), dtype=torch.float32, device=q.device
        )

        # Process each segment defined by cu_seqlens
        for seq_idx in range(cu_seqlens.shape[0] - 1):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # For each time step t in the segment
            for t in range(seq_len):
                # Build base tensors for this t
                t_idx = seq_start + t
                # Build q_vec, k_vec, v_vec (length 128)
                q_vec = q_exp[t_idx, :].contiguous()  # [128]
                k_vec = k_exp[t_idx, :].contiguous()  # [128]
                v_vec = v[t_idx, :].contiguous()      # [128]

                # Compute per-head g and beta for this segment
                # We use A_log[h], a[t, h], dt_bias[h], b[t, h]
                # Note: A_log is length 8 (num_v_heads), a and b are [seq_len, 8]
                h = 0
                # 1) Compute softplus(a[t,h] + dt_bias[h])
                a_t_h = a[t_idx, h].float()  # scalar
                dt_bias_h = dt_bias[h].float()  # scalar
                softplus_in = a_t_h + dt_bias_h
                softplus_out = torch.empty((), dtype=torch.float32, device=q.device)
                _softplus_vector[(1,)](softplus_in, softplus_out, N=128)
                softplus_val = softplus_out[0]  # scalar

                # 2) g = exp(-exp(A_log[h]) * softplus(softplus_in))
                A_log_h = A_log[h].float()  # scalar
                eA = torch.exp(A_log_h)     # scalar
                g_val = torch.exp(-eA * softplus_val)  # scalar
                # Store g into a 1-element tensor to pass to Triton
                g_scalar_t = torch.empty((), dtype=torch.float32, device=q.device)
                g_scalar_t.fill_(g_val)

                # 3) beta = sigmoid(b[t,h])
                b_t_h = b[t_idx, h].float()  # scalar
                beta_scalar_t = torch.empty((), dtype=torch.float32, device=q.device)
                _sigmoid_vector[(1,)](b_t_h, beta_scalar_t, N=128)
                beta_scalar = beta_scalar_t[0]  # scalar

                # 4) Load state_old_T for head h: state_curr has shape [seq_len, 8, 128, 128]
                #    For this segment, state_curr is just the original 'state' (not updated across t).
                #    We assume state corresponds to the last sequence end (this matches typical usage).
                #    We can take state_old = state[0] since cu_seqlens defines segments; for simplicity,
                #    we assume state_curr is provided for the segment. However, original code uses 'state'
                #    from module scope; here we use the input 'state' which is [1, 8, 128, 128].
                #    To get per-segment state, we should index by seq_idx. Since 'state' is provided as [1,8,128,128],
                #    we infer that it's the state at the start of the segment. We'll use state[0] as the
                #    state for this segment.
                #    In the original run, 'state' is provided as the initial state. We’ll use it as-is.
                #    So state_old_T is state[0, h] transposed to [K, V].
                #    Given 'state' has shape [1, 8, 128, 128], state_old_T is [128, 128].
                #    To make this robust, we’ll use the provided 'state' tensor directly.
                #    Here, 'state' is [1, 8, 128, 128], so state_old_T = state[0, h].transpose(2, 3).
                state_old = state[0, h].transpose(2, 3).contiguous()  # [128, 128]
                state_old_T = state_old  # already [128, 128]

                # 5) old_v = k_vec @ state_old_T via GEMV
                old_v = torch.empty(128, dtype=torch.float32, device=q.device)
                _gemv_1xKxKxV_into_1xV[(1,)](k_vec, state_old_T, old_v, K=128, V=128)

                # 6) new_v_vec = beta * v_vec + (1 - beta) * old_v
                new_v_vec = torch.empty(128, dtype=torch.float32, device=q.device)
                _elementwise_scalar_mul_add[(1,)](v_vec, old_v, new_v_vec, beta_scalar, N=128)

                # 7) Compute state_remove = dot(k_vec, old_v) (scalar)
                state_remove = torch.empty((), dtype=torch.float32, device=q.device)
                _dot_scalar_1xK[(1,)](k_vec, old_v, state_remove, K=128)

                # 8) state_update = dot(k_vec, new_v_vec) (scalar)
                state_update = torch.empty((), dtype=torch.float32, device=q.device)
                _dot_scalar_1xK[(1,)](k_vec, new_v_vec, state_update, K=128)

                # 9) state_new_mat = g * state_old_T + (state_update - state_remove)[None, :]
                #    First, subtract scalar from matrix
                diff = state_update - state_remove  # scalar
                state_new_tmp = torch.empty((128, 128), dtype=torch.float32, device=q.device)
                _subtract_scalar_from_matrix[(2, 2)](state_old_T, diff, state_new_tmp, K=128, V=128)
                # Then scale by g
                _scale_vector[(1,)](state_new_tmp, state_new_tmp, g_scalar_t[0], N=128 * 128)
                # Note: We scaled the whole matrix by scalar g (incorrect logic-wise). We need to scale
                # each row by g. The previous _scale_vector call incorrectly scaled elements elementwise.
                # Fix: perform row-wise scaling in Triton. We implement a kernel that multiplies each row by g.
                # We’ll do this by tiling over rows and multiplying by g.
                # Triton kernel to multiply each row of a [K,V] matrix by a scalar g:
                # (We need to pass g as a 1-element tensor; Triton can load it.)

                # Triton row-scale kernel: scale each row of A by g
                @triton.jit
                def _row_scale_matrix(A_ptr, out_ptr, g_scalar, K: tl.constexpr, V: tl.constexpr):
                    pid_k = tl.program_id(0)
                    row = pid_k * 64 + tl.arange(0, 64)
                    mask = row < K
                    # Load the row
                    a = tl.load(A_ptr + row[:, None] * V + tl.arange(0, V)[None, :], mask=mask[:, None], other=0.0)
                    # Multiply by g
                    a = a * g_scalar
                    tl.store(out_ptr + row[:, None] * V + tl.arange(0, V)[None, :], a, mask=mask[:, None])

                state_new_mat = torch.empty((128, 128), dtype=torch.float32, device=q.device)
                _row_scale_matrix[(2,)](state_new_tmp, state_new_mat, g_scalar_t[0], K=128, V=128)
                # Add the diff to each row (since state_new_tmp was initialized as state_old_T minus diff)
                # We need to add diff to each element; instead, we can add diff to state_new_mat directly by
                # using _subtract_scalar_from_matrix with alpha=-diff. But that changes values.
                # Correction: state_new_mat should be g * (state_old_T + (state_update - state_remove)).
                # We previously computed state_new_tmp = state_old_T - (state_update - state_remove),
                # and then scaled. That is incorrect. The correct steps:
                # tmp = state_old_T - (state_update - state_remove)  -> already computed
                # state_new_mat = g * tmp + (state_update - state_remove). We can implement this in Triton:
                # out = tmp + alpha, where alpha = (state_update - state_remove), and then scale by g.

                # Let’s fix this by recomputing:
                # tmp = state_old_T - diff
                # state_new_mat = g * tmp + diff
                # Implement a Triton kernel to add a scalar to each element:
                @triton.jit
                def _add_scalar_to_matrix_elements(A_ptr, out_ptr, alpha, K: tl.constexpr, V: tl.constexpr):
                    pid_k = tl.program_id(0)
                    pid_j = tl.program_id(1)
                    kk = pid_k * 64 + tl.arange(0, 64)
                    jj = pid_j * 64 + tl.arange(0, 64)
                    mask_k = kk < K
                    mask_j = jj < V
                    rows = kk[:, None]
                    cols = jj[None, :]
                    mask = mask_k[:, None] & mask_j[None, :]
                    a = tl.load(A_ptr + rows * V + cols, mask=mask, other=0.0)
                    a = a + alpha
                    tl.store(out_ptr + rows * V + cols, a, mask=mask)

                tmp = state_old_T  # start from state_old_T
                state_new_mat = torch.empty((128, 128), dtype=torch.float32, device=q.device)
                # tmp - diff
                _subtract_scalar_from_matrix[(2, 2)](tmp, diff, state_new_mat, K=128, V=128)
                # scale by g
                _row_scale_matrix[(2,)](state_new_mat, state_new_mat, g_scalar_t[0], K=128, V=128)
                # add diff back
                _add_scalar_to_matrix_elements[(2, 2)](state_new_mat, state_new_mat, diff, K=128, V=128)

                # 10) output_vec = scale * (q_vec @ state_new_mat)
                output_vec = torch.empty(128, dtype=torch.float32, device=q.device)
                _gemv_1xKxKxV_into_1xV[(1,)](q_vec, state_new_mat, output_vec, K=128, V=128)

                # Store output[t, h, :]
                output[t_idx, h, :] = output_vec

                # 11) Update new_state[seq_idx, h, :, :] = state_new_mat.transpose(0,1)
                new_state[seq_idx, h, :, :] = state_new_mat.transpose(0, 1)

        # Cast output to bfloat16 as required by original model
        output = output.to(torch.bfloat16)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
