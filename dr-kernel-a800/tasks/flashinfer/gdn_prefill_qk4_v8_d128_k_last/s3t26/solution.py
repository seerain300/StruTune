import torch
import math
import triton
import triton.language as tl


# Triton GEMV: computes out_vec[i] = sum_k B[k] * A[k, i], for i in 0..V-1
@triton.jit
def _gemv_1xKxKxV_into_1xV(B_ptr, A_ptr, out_ptr,
                            K: tl.constexpr, V: tl.constexpr,
                            stride_Ak: tl.constexpr, stride_Av: tl.constexpr):
    """
    B_ptr: [K] contiguous, A_ptr: [K, V] row-major, out_ptr: [V] contiguous.
    Computes out_vec = B @ A, where A is [K, V].
    """
    i = tl.arange(0, V)
    acc = tl.zeros([V], dtype=tl.float32)
    for k in range(0, K):
        b_k = tl.load(B_ptr + k)  # scalar
        a_row = tl.load(A_ptr + k * stride_Ak + i * stride_Av)  # [V]
        acc += b_k * a_row
    tl.store(out_ptr + i, acc)


# Triton GEMV_T: computes out_vec[k] = sum_v B[v] * A_T[v, k], for k in 0..K-1
@triton.jit
def _gemv_1xVxK_into_1xK(B_ptr, A_T_ptr, out_ptr,
                         V: tl.constexpr, K: tl.constexpr,
                         stride_AT_v: tl.constexpr, stride_AT_k: tl.constexpr):
    """
    B_ptr: [V] contiguous, A_T_ptr: [V, K] row-major (A^T), out_ptr: [K] contiguous.
    Computes out_vec = B @ A_T, where A_T is [V, K].
    """
    k = tl.arange(0, K)
    acc = tl.zeros([K], dtype=tl.float32)
    for v in range(0, V):
        b_v = tl.load(B_ptr + v)  # scalar
        a_row = tl.load(A_T_ptr + v * stride_AT_v + k * stride_AT_k)  # [K]
        acc += b_v * a_row
    tl.store(out_ptr + k, acc)


# Triton dot product: dot = sum_i x[i] * y[i]
@triton.jit
def _dot_scalar(x_ptr, y_ptr, out_ptr,
                N: tl.constexpr):
    acc = tl.zeros([1], dtype=tl.float32)
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.load(y_ptr + i, mask=i < N, other=0.0)
    acc = tl.sum(x * y, axis=0)
    tl.store(out_ptr, acc)


# Triton elementwise: out[i] = alpha * x[i] + y[i]
@triton.jit
def _elementwise_mul_add_scalar(alpha, x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.load(y_ptr + i, mask=i < N, other=0.0)
    out = alpha * x + y
    tl.store(out_ptr + i, out, mask=i < N)


# Helper functions to launch Triton kernels
def _gemv_triton(B, A):
    """
    B: [K] tensor, A: [K, V] tensor (row-major), returns out: [V] tensor (float32).
    """
    assert B.is_cuda and A.is_cuda
    K = B.numel()
    V = A.shape[1]
    out = torch.empty((V,), dtype=torch.float32, device=B.device)
    # Single block since V is small and constexpr; K,V=128
    _gemv_1xKxKxV_into_1xV[(1,)](B, A, out, K=K, V=V, stride_Ak=V, stride_Av=1, num_warps=4)
    return out


def _gemv_triton_T(B, A_T):
    """
    B: [V] tensor, A_T: [V, K] tensor (row-major), returns out: [K] tensor (float32).
    """
    assert B.is_cuda and A_T.is_cuda
    V = B.numel()
    K = A_T.shape[1]
    out = torch.empty((K,), dtype=torch.float32, device=B.device)
    _gemv_1xVxK_into_1xK[(1,)](B, A_T, out, V=V, K=K, stride_AT_v=K, stride_AT_k=1, num_warps=4)
    return out


def _dot_triton(x, y):
    """
    x, y: [N] tensors, returns dot product as a 1-element tensor (float32).
    """
    assert x.is_cuda and y.is_cuda
    N = x.numel()
    out = torch.empty((1,), dtype=torch.float32, device=x.device)
    _dot_scalar[(1,)](x, y, out, N=N, num_warps=1)
    return out


def _elementwise_mul_add_scalar_triton(alpha, x, y, out):
    """
    out = alpha * x + y, elementwise. x,y,out: [N] contiguous tensors.
    """
    assert x.is_cuda and y.is_cuda and out.is_cuda
    N = x.numel()
    _elementwise_mul_add_scalar[(1,)](alpha, x, y, out, N=N, num_warps=1)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        q: [T, num_q_heads, 128] bfloat16
        k: [T, num_k_heads, 128] bfloat16
        v: [T, num_v_heads, 128] bfloat16
        state: [num_seqs, num_v_heads, 128, 128] float32 (k-last: [H,V,K])
        A_log: [num_v_heads] float32
        a: [T, num_v_heads] bfloat16
        dt_bias: [num_v_heads] float32
        b: [T, num_v_heads] bfloat16
        cu_seqlens: [num_seqs+1] int64
        scale: float32 scalar
        Returns (output, new_state), output bfloat16 [T,num_v_heads,128], new_state float32 [num_seqs,num_v_heads,128,128]
        """
        device = q.device
        T, num_q_heads, _ = q.shape
        _, num_k_heads, _ = k.shape
        _, num_v_heads, _ = v.shape
        num_seqs = cu_seqlens.size(0) - 1

        # Expanded q,k for heads
        q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1).contiguous()
        k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1).contiguous()

        output = torch.empty((T, num_v_heads, 128), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((num_seqs, num_v_heads, 128, 128), dtype=torch.float32, device=device)

        # Compute g and beta per (t,h) on host for correctness (small vectors)
        # softplus(x) = log(1 + exp(x))
        # g = exp(-exp(A_log[h]) * softplus(a[t,h] + dt_bias[h]))
        # beta = sigmoid(b[t,h])
        # We'll compute them per head for each t using torch, since Triton cannot index by t.
        # Note: num_q_heads, num_k_heads, num_v_heads may vary; we compute per head and use in Triton for elementwise math.
        # However, since Triton kernels do not accept dynamic indexing, we keep these as host-side tensors.
        # Create g and beta tensors [T, H].
        a_f = a.float()
        b_f = b.float()
        g = torch.exp(-torch.exp(A_log.float()) * F.softplus(a_f + dt_bias.float()))  # [T, H]
        beta = torch.sigmoid(b_f)  # [T, H]

        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # state is [num_seqs, H, V, K] which we index per head h -> [V,K]
            state_curr = state[seq_idx].contiguous()  # [H, V, K] here H=V=K=128

            # scale
            scale_val = float(scale) if scale is not None else 1.0 / math.sqrt(128)

            for t in range(seq_len):
                abs_t = seq_start + t

                # For each head h
                for h in range(num_v_heads):
                    # q_vec, k_vec, v_vec for this t,h
                    q_vec = q_exp[abs_t, h].contiguous().float()  # [128]
                    k_vec = k_exp[abs_t, h].contiguous().float()  # [128]
                    v_vec = v[abs_t, h].contiguous().float()      # [128]

                    # Load state_old_T[h] = state_curr[h] as [V, K]
                    state_h_KV = state_curr[h].contiguous()  # [V, K]
                    V, K = 128, 128

                    # 1) old_v = k_vec @ state_old_T
                    old_v = _gemv_triton(k_vec, state_h_KV)  # [V]

                    # 2) new_v_vec = beta[t,h] * v_vec + (1 - beta) * old_v
                    beta_h = float(beta[abs_t, h].item())
                    new_v_vec = torch.empty((V,), dtype=torch.float32, device=device)
                    _elementwise_mul_add_scalar_triton(beta_h, v_vec, old_v, new_v_vec, num_warps=1)

                    # 3) Compute state_remove = dot(k_vec, old_v) and state_update = dot(k_vec, new_v_vec)
                    state_remove = _dot_triton(k_vec, old_v)  # [1]
                    state_update = _dot_triton(k_vec, new_v_vec)  # [1]
                    alpha = (float(state_update.item()) - float(state_remove.item()))  # scalar

                    # 4) state_new_T[h] = g[t,h] * state_old_T + alpha broadcast
                    g_h = float(g[abs_t, h].item())
                    state_new_T = (g_h * state_h_KV.float()) + alpha  # [V, K]

                    # Update new_state[seq_idx, h] = state_new_T
                    new_state[seq_idx, h] = state_new_T  # [V, K] -> [H,V,K] layout

                    # 5) output_vec = scale * (q_vec @ state_new_T)
                    # state_new_T is [V, K], q_vec is [K], output is [V]
                    out_vec = _gemv_triton_T(q_vec, state_new_T)  # [K]
                    # output is [T, H, V], but


def run(*args):
    return ModelNew()(*args)
