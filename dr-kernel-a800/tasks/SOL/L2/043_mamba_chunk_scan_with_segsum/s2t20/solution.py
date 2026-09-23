import torch
import torch.nn.functional as F

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: elementwise multiply Y = D[h, d] * X[b, s, h, d]
# Shapes:
#   X: [B, S, H, D] (float32)
#   D: [H, D] (float32)
#   Y: [B, S, H, D] (float32), output of Y = D * X
@triton.jit
def d_residual_mul_kernel(X_ptr, D_ptr, Y_ptr,
                          B, S, H, D,
                          stride_x_b, stride_x_s, stride_x_h, stride_x_d,
                          stride_d_h, stride_d_d,
                          stride_y_b, stride_y_s, stride_y_h, stride_y_d):
    total = B * S * H * D
    pid = tl.program_id(axis=0)
    if pid >= total:
        return
    b = pid // (S * H * D)
    rem = pid % (S * H * D)
    s = rem // (H * D)
    rem2 = rem % (H * D)
    h = rem2 // D
    d = rem2 % D

    x_off = b * stride_x_b + s * stride_x_s + h * stride_x_h + d * stride_x_d
    d_off = h * stride_d_h + d * stride_d_d
    y_off = b * stride_y_b + s * stride_y_s + h * stride_y_h + d * stride_y_d

    x_val = tl.load(X_ptr + x_off)
    d_val = tl.load(D_ptr + d_off)
    y_val = x_val * d_val
    tl.store(Y_ptr + y_off, y_val)


# Triton kernel: segment sum lower-triangular (diagonal=-1) with exp, along last dim
# Input: A: [B, T, I] (float32), Output: L: [B, T, I] = exp(tril(A, -1).cumsum along last dim per (b,t))
# We compute cumsum across j from 0..I-1, and mask j <= i-1
@triton.jit
def segment_sum_lower_tri_exp_kernel(A_ptr, L_ptr,
                                     B, T, I,
                                     stride_a_b, stride_a_t, stride_a_i,
                                     stride_l_b, stride_l_t, stride_l_i):
    b = tl.program_id(0)
    t = tl.program_id(1)
    i = tl.program_id(2)

    # Initialize cumsum
    cumsum = 0.0

    if i == 0:
        tl.store(L_ptr + b * stride_l_b + t * stride_l_t + i * stride_l_i, 0.0)
        return

    for j in range(0, i):
        a_off = b * stride_a_b + t * stride_a_t + j * stride_a_i
        a_val = tl.load(A_ptr + a_off)
        cumsum += a_val
    exp_val = tl.exp(cumsum)
    l_off = b * stride_l_b + t * stride_l_t + i * stride_l_i
    tl.store(L_ptr + l_off, exp_val)


# Triton kernel: reshape and cast (not doing heavy math, just moving and casting)
# We take a flattened source tensor and write to output with a given shape and dtype bfloat16
# This is a very simple copy+cast kernel. Forward will use it to produce final output.
@triton.jit
def reshape_and_cast_kernel(src_ptr, out_ptr,
                            num_elems,
                            out_b, out_s, out_hd,
                            stride_out_b, stride_out_s, stride_out_hd):
    pid = tl.program_id(axis=0)
    if pid >= num_elems:
        return
    # Write to out[b, s, hd]
    # Compute b, s, hd via division/modulo
    b = pid // (out_s * out_hd)
    rem = pid % (out_s * out_hd)
    s = rem // out_hd
    hd = rem % out_hd

    out_off = b * stride_out_b + s * stride_out_s + hd * stride_out_hd
    val = tl.load(src_ptr + pid)  # src_ptr is linearized
    # Cast to bfloat16 (Triton will cast if out_ptr is bfloat16)
    tl.store(out_ptr + out_off, val)


def _run_d_residual_mul(x, d):
    """
    x: [B, S, H, D] float32 tensor
    d: [H, D] float32 tensor
    return: y: [B, S, H, D] float32, y = d * x
    """
    B, S, H, D = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    stride_x_b, stride_x_s, stride_x_h, stride_x_d = x.stride()
    stride_d_h, stride_d_d = d.stride()
    stride_y_b, stride_y_s, stride_y_h, stride_y_d = y.stride()
    total = B * S * H * D
    grid = (total,)
    d_residual_mul_kernel[grid](
        x, d, y,
        B, S, H, D,
        stride_x_b, stride_x_s, stride_x_h, stride_x_d,
        stride_d_h, stride_d_d,
        stride_y_b, stride_y_s, stride_y_h, stride_y_d,
        num_warps=1,
        num_stages=1,
    )
    return y


def _run_segment_sum_lower_tri_exp(a):
    """
    a: [B, T, I] float32 tensor
    returns: l: [B, T, I] float32 tensor with l[b, t, i] = exp(tril(a[b, t, :i], diagonal=-1).sum())
    """
    assert a.ndim == 3, "a must be 3D [B, T, I]"
    B, T, I = a.shape
    l = torch.empty_like(a, dtype=torch.float32)
    stride_a_b, stride_a_t, stride_a_i = a.stride()
    stride_l_b, stride_l_t, stride_l_i = l.stride()
    grid = (B, T, I)
    segment_sum_lower_tri_exp_kernel[grid](
        a, l,
        B, T, I,
        stride_a_b, stride_a_t, stride_a_i,
        stride_l_b, stride_l_t, stride_l_i,
        num_warps=1,
        num_stages=1,
    )
    return l


def _run_reshape_and_cast(src_linear, out_shape, out_dtype):
    """
    src_linear: 1D tensor (linearized), out_shape: (B, S, H*D), out_dtype: torch dtype
    returns: out tensor with shape out_shape and dtype out_dtype (bfloat16)
    """
    B, S, H_D = out_shape
    out = torch.empty((B, S, H_D), device=src_linear.device, dtype=out_dtype)
    # We need strides for out
    stride_out_b, stride_out_s, stride_out_hd = out.stride()
    num_elems = B * S * H_D
    grid = (num_elems,)
    reshape_and_cast_kernel[grid](
        src_linear, out,
        num_elems,
        B, S, H_D,
        stride_out_b, stride_out_s, stride_out_hd,
        num_warps=1,
        num_stages=1,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, initial_states: torch.Tensor):
        """
        Return: (output, final_state)
        output: [batch_size, seq_len, num_heads * head_dim], dtype bfloat16
        final_state: [batch_size, num_heads, head_dim, state_size], dtype bfloat16 (dummy, not used in original computation)
        """

        # Shapes from original
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        state_size = 256
        n_groups = 1
        chunk_size = 256

        # Compute padding size to make seq_len multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

        # Convert to float32 for numerical stability (match original behavior)
        hidden_states_f = hidden_states.to(torch.float32)
        D = D.to(torch.float32)  # [H, D]

        # We need to produce output = [B, S, H*D] and cast to bfloat16.
        # Since


def run(*args):
    return ModelNew()(*args)
