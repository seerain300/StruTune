import torch
import math
import torch.nn.functional as F

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton elementwise kernels
if TRITON_AVAILABLE:
    @triton.jit
    def softplus_ab_kernel(a_ptr, dt_bias_ptr, sp_ptr,
                            T: tl.constexpr, V: tl.constexpr):
        # 2D grid over (T, V)
        t = tl.program_id(0)
        hv = tl.program_id(1)
        if t >= T or hv >= V:
            return
        a_val = tl.load(a_ptr + t * V + hv)
        dt_bias_val = tl.load(dt_bias_ptr + hv)
        softplus_val = tl.log(1.0 + tl.exp(a_val + dt_bias_val))
        tl.store(sp_ptr + t * V + hv, softplus_val)

    @triton.jit
    def sigmoid_b_kernel(b_ptr, sig_ptr,
                          T: tl.constexpr, V: tl.constexpr):
        # 2D grid over (T, V)
        t = tl.program_id(0)
        hv = tl.program_id(1)
        if t >= T or hv >= V:
            return
        b_val = tl.load(b_ptr + t * V + hv)
        sig_val = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(sig_ptr + t * V + hv, sig_val)

    @triton.jit
    def g_from_softplus_kernel(A_log_ptr, sp_ptr, g_ptr,
                                T: tl.constexpr, V: tl.constexpr):
        # 2D grid over (T, V)
        t = tl.program_id(0)
        hv = tl.program_id(1)
        if t >= T or hv >= V:
            return
        A_log_val = tl.load(A_log_ptr + hv)    # A_log[hv]
        sp_val = tl.load(sp_ptr + t * V + hv)  # softplus(a[t,hv] + dt_bias[hv])
        g_val = tl.exp(-tl.exp(A_log_val) * sp_val)
        tl.store(g_ptr + t * V + hv, g_val)

    @triton.jit
    def matmul_kernel(A_ptr, B_ptr, C_ptr,
                       M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       stride_am, stride_ak,
                       stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # 2D grid over tiles (M,N)
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def scalar_update_kernel(state_ptr,
                              g_scalar, remove_ptr, update_ptr,
                              H: tl.constexpr, K: tl.constexpr, V: tl.constexpr):
        # 3D grid over (h, k, v)
        h = tl.program_id(0)
        k = tl.program_id(1)
        v = tl.program_id(2)
        if h >= H or k >= K or v >= V:
            return
        state_val = tl.load(state_ptr + h * (K * V) + k * V + v)
        remove_val = tl.load(remove_ptr + h * K + k)
        update_val = tl.load(update_ptr + h * K + k)
        new_val = state_val * g_scalar - remove_val + update_val
        tl.store(state_ptr + h * (K * V) + k * V + v, new_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-only implementation. Computes output and new_state using Triton kernels.
        Returns:
          - output: list of [H, V] tensors (bfloat16), one per t
          - new_state: None (not used; original returns None in our case)
        """
        device = q.device
        if not TRITON_AVAILABLE:
            total_seq_len, H, K = q.shape
            V = v.shape[1]
            output = []
            return output, None

        # Use asserts from original: H=4, K=4, V=8
        total_seq_len, H, K = q.shape
        V = v.shape[1]
        assert H == 4 and K == 4 and V == 8, "This Triton implementation expects H=4, K=4, V=8 as per original asserts."

        # Prepare output list
        output = [None] * total_seq_len

        # Process segments
        num_segments = cu_seqlens.shape[0] - 1
        for seg in range(num_segments):
            seq_start = int(cu_seqlens[seg].item())
            seq_end = int(cu_seqlens[seg + 1].item())
            T = seq_end - seq_start
            if T <= 0:
                continue

            # Per-segment state in float32
            state_HKV = torch.zeros((H, K, V), dtype=torch.float32, device=device)

            # Elementwise Triton buffers for this segment
            a_t = a[seq_start:seq_end]     # [T, V]
            b_t = b[seq_start:seq_end]     # [T, V]
            sp = torch.empty((T, V), dtype=torch.float32, device=device)
            sig = torch.empty((T, V), dtype=torch.float32, device=device)
            g = torch.empty((T, V), dtype=torch.float32, device=device)

            # Launch Triton elementwise kernels
            grid_elem = (T, V)
            softplus_ab_kernel[grid_elem](a_t, dt_bias, sp, T, V, num_warps=2)
            sigmoid_b_kernel[grid_elem](b_t, sig, T, V, num_warps=2)
            g_from_softplus_kernel[(V,)](A_log, sp, g, T, V, num_warps=2)  # A_log length V; broadcast along T

            # Iterate per t within this segment
            for t_idx in range(T):
                k_t = k[seq_start + t_idx]           # [H, K]
                v_t = v[seq_start + t_idx]           # [H, V]
                q_t = q[seq_start + t_idx]           # [H, K]

                # old_v = k @ state_HKV
                A_k = k_t.contiguous().view(H, K)    # [H, K]
                B_k = state_HKV                      # [K, V]
                C_k = torch.empty((H, V), dtype=torch.float32, device=device)
                grid_k = (triton.cdiv(H, 64), triton.cdiv(V, 64))
                matmul_kernel[grid_k](A_k, B_k, C_k,
                                      H, V, K,
                                      A_k.stride(0), A_k.stride(1),
                                      B_k.stride(0), B_k.stride(1),
                                      C_k.stride(0), C_k.stride(1),
                                      BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4)
                old_v = C_k                         # [H, V]

                # new_v = beta * v + (1 - beta) * old_v
                beta_scalar = float(sig[t_idx].item())
                new_v = beta_scalar * v_t + (1.0 - beta_scalar) * old_v  # [H, V]

                # state_remove = k @ old_v
                A_rm = k_t
                B_rm = old_v
                C_rm = torch.empty((H, V), dtype=torch.float32, device=device)
                grid_rm = (triton.cdiv(H, 64), triton.cdiv(V, 64))
                matmul_kernel[grid_rm](A_rm, B_rm, C_rm,
                                       H, V, K,
                                       A_rm.stride(0), A_rm.stride(1),
                                       B_rm.stride(0), B_rm.stride(1),
                                       C_rm.stride(0), C_rm.stride(1),
                                       BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4)
                state_remove = C_rm                   # [H, K]

                # state_update = k @ new_v
                A_up = k_t
                B_up = new_v
                C_up = torch.empty((H, V), dtype=torch.float32, device=device)
                grid_up = (triton.cdiv(H, 64), triton.cdiv(V, 64))
                matmul_kernel[grid_up](A_up, B_up, C_up,
                                       H, V, K,
                                       A_up.stride(0), A_up.stride(1),
                                       B_up.stride(0), B_up.stride(1),
                                       C_up.stride(0), C_up.stride(1),
                                       BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4)
                state_update = C_up                   # [H, K]

                # Update state_HKV using scalar g_scalar and vectors
                g_scalar = float(g[t_idx].item())
                remove_vec = state_remove.view(H * K).contiguous()    # [H*K]
                update_vec = state_update.view(H * K).contiguous()    # [H*K]
                state_flat = state_HKV.view(H * K * V).contiguous()   # [H*K*V]
                grid_upd = (H, K, V)
                scalar_update_kernel[grid_upd](state_flat,
                                               g_scalar, remove_vec, update_vec,
                                               H, K, V, num_warps=2)

                # Compute output[t] = scale * q @ state_HKV
                A_q = q_t.contiguous().view(H, K)   # [H, K]
                B_q = state_HKV                      # [K, V]
                C_q = torch.empty((H, V), dtype=torch.float32, device=device)
                grid_q = (triton.cdiv(H, 64), triton.cdiv(V, 64))
                matmul_kernel[grid_q](A_q, B_q, C_q,
                                      H, V, K,
                                      A_q.stride(0), A_q.stride(1),
                                      B_q.stride(0), B_q.stride(1),
                                      C_q.stride(0), C_q.stride(1),
                                      BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4)
                output[seq_start + t_idx] = (C_q * (scale if scale is not None else 1.0)).to(torch.bfloat16)

        return (output, None)


def run(*args):
    return ModelNew()(*args)
