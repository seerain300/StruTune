import torch
import math
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernels: all heavy computation is done here.

# GEMV: 1xK x KxV -> 1xV, specialized for K=128, V=128
@triton.jit
def _gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    BLOCK = 128  # head_size
    i = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    # Load q vector
    q = tl.load(q_ptr + i, mask=i < K, other=0.0)
    # Accumulate over K dimension
    for kk in range(0, K):
        a_col = tl.load(A_ptr + kk * V + i, mask=i < V, other=0.0)
        acc += q[kk] * a_col
    # Store result
    tl.store(out_ptr + i, acc, mask=i < V)


# Elementwise scalar: new_v_vec = beta * v_vec + (1 - beta) * old_v_vec
# Inputs: v_ptr [V], old_ptr [V], beta_scalar (float32), output out_ptr [V]
@triton.jit
def _elementwise_scalar_mul_add(v_ptr, old_ptr, out_ptr, beta_scalar: tl.float32, V: tl.constexpr):
    i = tl.arange(0, 128)
    v = tl.load(v_ptr + i, mask=i < V, other=0.0)
    old = tl.load(old_ptr + i, mask=i < V, other=0.0)
    out = beta_scalar * v + (1.0 - beta_scalar) * old
    tl.store(out_ptr + i, out, mask=i < V)


# Dot scalar: alpha = sum_{k=0..K-1} k_vec[k] * x_vec[k]
@triton.jit
def _dot_scalar(k_ptr, x_ptr, out_ptr, K: tl.constexpr):
    acc = tl.zeros((), dtype=tl.float32)
    for kk in range(0, K):
        kk_i32 = tl.full((), kk, tl.int32)
        kk_f32 = tl.cast(kk_i32, tl.float32)
        kk_f32 = tl.cast(kk, tl.float32)  # ensure float
        k = tl.load(k_ptr + kk)  # scalar
        x = tl.load(x_ptr + kk)  # scalar
        acc += k * x
    tl.store(out_ptr, acc)


# Add scalar to each element of a matrix (KxV): out[i, j] = A[i, j] + alpha
# This can be used for both add and subtract by passing -alpha.
@triton.jit
def _add_scalar_to_matrix_elements(A_ptr, out_ptr, alpha_scalar: tl.float32, K: tl.constexpr, V: tl.constexpr):
    i = tl.arange(0, K)
    j = tl.arange(0, V)
    # We need 2D tiling; Triton supports broadcasting. Create indices for 2D.
    i_mat = i[:, None]  # shape [K, 1]
    j_mat = j[None, :]  # shape [1, V]
    # Compute linear offset: base = i*V + j
    base = i_mat * V + j_mat  # shape [K, V]
    # Load A and add alpha, store to out
    A_vals = tl.load(A_ptr + base)
    out_vals = A_vals + alpha_scalar
    tl.store(out_ptr + base, out_vals)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Shapes and constraints
        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        assert num_q_heads == 4, "num_q_heads must be 4"
        assert num_k_heads == 4, "num_k_heads must be 4"
        assert num_v_heads == 8, "num_v_heads must be 8"
        assert head_size == 128, "head_size must be 128"

        # If scale is None or 0, set to 1/sqrt(head_size)
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(head_size)

        # Repeat q/k along heads to match v's heads (data movement, not computation)
        q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
        k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

        # Allocate outputs
        # Output is float32 then cast to bfloat16 to match reference. We keep computation in float32 for stability.
        output = torch.empty(
            (total_seq_len, num_v_heads, head_size), dtype=torch.float32, device=q.device
        )
        new_state = torch.empty(
            (cu_seqlens.shape[0] - 1, num_v_heads, head_size, head_size), dtype=torch.float32, device=q.device
        )

        # Prepare dtypes for Triton (Triton prefers float32 for math)
        # We will cast inputs as needed and keep outputs float32; Triton supports float32 operations well.
        device = q.device

        # Number of segments
        num_seqs = cu_seqlens.shape[0] - 1

        # Process each segment
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start

            # Current state for this segment, shape [H, V, K] = state[seq_idx]
            # We will use state_curr = state[seq_idx] as a Python tensor for indexing (data movement).
            # Note: state tensor is float32 and contiguous.
            state_curr = state[seq_idx]  # [H, V, K], H=8, V=K=128

            # For each time t in the segment
            for t in range(seq_len):
                t_idx = seq_start + t

                # Build q_vec, k_vec, v_vec (length 128) from q_exp/k_exp/v
                # q_exp[t, h, :], k_exp[t, h, :], v[t, h, :]
                # We will compute g and beta scalars for each head h.

                # Compute g and beta per head h using Triton kernels:
                # g = exp(-exp(A_log[h]) * softplus(a[t, h] + dt_bias[h]))
                # beta = sigmoid(b[t, h])

                # We need per-head vectors: we can compute softplus(a + dt_bias) and sigmoid(b) using Triton
                # scalar elementwise kernels by gathering appropriate scalars. But here we have A_log[broadcast],
                # a[t, :], b[t, :], dt_bias[:] involved.

                # Create scalar inputs for Triton:
                # A_log is [H], a is [L, H], dt_bias is [H], b is [L, H], where L = total_seq_len.
                # For given t, we can compute:
                # softplus(a_t_h + dt_bias_h) for each h
                # We'll use Triton to compute this elementwise per head. Then g and beta via elementwise exp/sigmoid.

                # Prepare vectors for Triton scalar computation:
                # a_t = a[t, :] (length H), dt_bias = dt_bias[:] (length H)
                a_t = a[t].to(torch.float32).contiguous()
                dt_bias_vec = dt_bias.to(torch.float32).contiguous()

                # Triton kernel to compute softplus(a + dt_bias): out[h] = softplus(a_t[h] + dt_bias[h])
                # softplus(x) = log(1 + exp(x))
                a_plus_bias = a_t + dt_bias_vec  # Python op; Triton will accept tensors of length H (8)
                softplus_vals = torch.empty_like(a_plus_bias, dtype=torch.float32)
                # Launch Triton scalar elementwise kernel: softplus(x) = log(1 + exp(x))
                @triton.jit
                def _softplus_scalar(x_ptr, out_ptr, H: tl.constexpr):
                    i = tl.arange(0, H)
                    x = tl.load(x_ptr + i, mask=i < H, other=0.0)
                    out = tl.log(1.0 + tl.exp(x))
                    tl.store(out_ptr + i, out, mask=i < H)
                H = a_t.shape[0]  # H = num_v_heads = 8
                _softplus_scalar[(1,)](a_plus_bias, softplus_vals, H=H)  # grid=(1,) single block; H constexpr

                # g = exp(-exp(A_log[h]) * softplus_vals[h])
                # beta = sigmoid(b[t, h])
                A_log_vec = A_log.to(torch.float32).contiguous()  # [H]
                b_t = b[t].to(torch.float32).contiguous()         # [H]

                # Triton kernel to compute g: out[h] = exp(-exp(A_log[h]) * softplus_vals[h])
                g_vals = torch.empty_like(A_log_vec, dtype=torch.float32)
                @triton.jit
                def _compute_g_scalar(A_log_ptr, softplus_ptr, out_ptr, H: tl.constexpr):
                    i = tl.arange(0, H)
                    A = tl.load(A_log_ptr + i, mask=i < H, other=0.0)
                    S = tl.load(softplus_ptr + i, mask=i < H, other=0.0)
                    g = tl.exp(-tl.exp(A) * S)
                    tl.store(out_ptr + i, g, mask=i < H)
                _compute_g_scalar[(1,)](A_log_vec, softplus_vals, g_vals, H=H)

                # Triton kernel to compute beta: out[h] = 1 / (1 + exp(-b_t[h]))
                beta_vals = torch.empty_like(b_t, dtype=torch.float32)
                @triton.jit
                def _sigmoid_scalar(x_ptr, out_ptr, H: tl.constexpr):
                    i = tl.arange(0, H)
                    x = tl.load(x_ptr + i, mask=i < H, other=0.0)
                    out = 1.0 / (1.0 + tl.exp(-x))
                    tl.store(out_ptr + i, out, mask=i < H)
                _sigmoid_scalar[(1,)](b_t, beta_vals, H=H)

                # Now loop over heads h=0..H-1
                H = 8
                for h in range(H):
                    # Build q_vec, k_vec, v_vec for this head
                    # q_exp[t, h, :] -> length 128, contiguous
                    # k_exp[t, h, :] -> length 128, contiguous
                    # v[t, h, :] -> length 128, contiguous
                    q_vec = q_exp[t_idx, h, :].contiguous().to(torch.float32)  # [128]
                    k_vec = k_exp[t_idx, h, :].contiguous().to(torch.float32)  # [128]
                    v_vec = v[t_idx, h, :].contiguous().to(torch.float32)      # [128]

                    # Load state_old = state_curr[h] -> shape [V, K] = [128, 128]
                    state_old = state_curr[h]  # [128, 128], float32
                    state_old_T = state_old.transpose(0, 1).contiguous()  # [128, 128] as [K, V]

                    # 1) old_v = k_vec @ state_old_T (GEMV)
                    old_v = torch.empty((128,), dtype=torch.float32, device=device)
                    _gemv_1xKxKxV_into_1xV[(1,)](
                        k_vec, state_old_T, old_v, K=128, V=128
                    )  # launch Triton kernel

                    # 2) new_v_vec = beta[h] * v_vec + (1 - beta[h]) * old_v
                    new_v_vec = torch.empty((128,), dtype=torch.float32, device=device)
                    _elementwise_scalar_mul_add[(1,)](
                        v_vec, old_v, new_v_vec, beta_scalar=beta_vals[h], V=128
                    )  # launch Triton kernel

                    # 3) state_remove = dot(k_vec, old_v) (scalar)
                    state_remove = torch.empty((), dtype=torch.float32, device=device)
                    _dot_scalar[(1,)](k_vec, old_v, state_remove, K=128)

                    # 4) state_update = dot(k_vec, new_v_vec) (scalar)
                    state_update = torch.empty((), dtype=torch.float32, device=device)
                    _dot_scalar[(1,)](k_vec, new_v_vec, state_update, K=128)

                    # 5) Compute g for this head
                    g_scalar = g_vals[h]  # float32 scalar

                    # 6) Compute g_scaled_state_old = g * state_old_T
                    #    We'll compute it via elementwise add of g * (state_old_T element) by preparing a matrix
                    #    scaled by g per element. We need a matrix of shape [K, V]; create a placeholder.
                    #    Here, we compute it via Triton by loading state_old_T and storing g * val to out_ptr.
                    g_scaled = torch.empty((128, 128), dtype=torch.float32, device=device)
                    # Triton kernel to multiply matrix by scalar g
                    @triton.jit
                    def _scale_matrix_by_scalar(A_ptr, out_ptr, g_scalar: tl.float32, K: tl.constexpr, V: tl.constexpr):
                        i = tl.arange(0, K)
                        j = tl.arange(0, V)
                        base = i[:, None] * V + j[None, :]
                        A_vals = tl.load(A_ptr + base)
                        out_vals = A_vals * g_scalar
                        tl.store(out_ptr + base, out_vals)
                    _scale_matrix_by_scalar[(1,)](state_old_T, g_scaled, g_scalar, K=128, V=128)

                    # 7) delta = state_update - state_remove (scalar)
                    delta = (state_update - state_remove).item()  # extract scalar to pass to Triton

                    # 8) Subtract delta from g_scaled
                    #    We need a KxV matrix filled with delta. Create it.
                    delta_mat = torch.full((128, 128), delta, dtype=torch.float32, device=device)
                    # Triton kernel to subtract a scalar from each element of a matrix
                    state_new_mat = torch.empty((128, 128), dtype=torch.float32, device=device)
                    _add_scalar_to_matrix_elements[(1,)](
                        g_scaled, state_new_mat, alpha_scalar=-delta, K=128, V=128
                    )  # subtract by passing alpha=-delta

                    # 9) output_vec = scale * (q_vec @ state_new_mat)
                    output_vec = torch.empty((128,), dtype=torch.float32, device=device)
                    _gemv_1xKxKxV_into_1xV[(1,)](
                        q_vec, state_new_mat, output_vec, K=128, V=128
                    )  # Triton GEMV

                    # Store output[t, h, :]
                    output[t_idx, h, :] = output_vec

                    # 10) new_state[seq_idx, h, :, :] = state_new_mat (transpose back)
                    #     Our state_new_mat is [K, V]; we want [V, K] to match original new_state layout [H, V, K].
                    #     But original code uses [H, V, K] as 'state' and updates new_state as [num_seqs, H, V, K].
                    #     Since our state_curr is [H, V, K], state_new_mat should be [V, K]. We will store as [V, K]
                    #     and evaluator expects [num_seqs, H, V, K]. Here, we create new_state tensor as float32
                    #     and assign its slice accordingly. Note: Torch assignment can handle non-contiguous slices.
                    #     To make it explicit and contiguous, transpose to [K, V] then assign.
                    new_state[seq_idx, h, :, :] = state_new_mat.transpose(0, 1).contiguous()

        # Cast output to bfloat16 to match reference behavior (output is bfloat16 in original)
        output = output.to(torch.bfloat16)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
