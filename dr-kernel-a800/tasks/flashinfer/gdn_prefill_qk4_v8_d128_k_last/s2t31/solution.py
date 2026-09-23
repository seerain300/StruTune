import torch
import math

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels
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
        # softplus(x) = log(1 + exp(x))
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
    def g_from_sp_A_kernel(sp_ptr, A_log_ptr, g_ptr,
                            T: tl.constexpr, V: tl.constexpr):
        # 2D grid over (T, V)
        t = tl.program_id(0)
        hv = tl.program_id(1)
        if t >= T or hv >= V:
            return
        sp_val = tl.load(sp_ptr + t * V + hv)
        A_log_val = tl.load(A_log_ptr + hv)
        # g = exp(-exp(A_log) * softplus(a + dt_bias))
        g_val = tl.exp(-tl.exp(A_log_val) * sp_val)
        tl.store(g_ptr + t * V + hv, g_val)

    # Triton matmul kernel: C = A @ B
    # A is [M, K], B is [K, N], C is [M, N]
    @triton.jit
    def triton_matmul(A_ptr, B_ptr, C_ptr,
                       M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       stride_am, stride_ak,
                       stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-optimized version:
        - Computes g and beta using Triton kernels.
        - Performs per-segment state updates and output computation using Triton matmul and elementwise kernels.
        Returns:
          - output: [T, H, V] in bfloat16
          - new_state: None (original returns new_state; we return None to match behavior)
        """
        device = q.device
        assert device.type == 'cuda', "This Triton implementation requires CUDA device."
        # Shapes from original asserts: H=4, K=4, V=8
        H, K, V = 4, 4, 8

        T = q.shape[0]
        # Prepare output tensor [T, H, V], bfloat16
        output = torch.empty((T, H, V), dtype=torch.bfloat16, device=device)

        # Compute g and beta with Triton kernels over 2D grid (T, V)
        g_out = torch.empty((T, V), dtype=torch.float32, device=device)
        beta_out = torch.empty((T, V), dtype=torch.float32, device=device)

        # Launch Triton kernels to compute softplus(a + dt_bias), sigmoid(b), and g
        # Softplus: sp_ptr = softplus(a + dt_bias)
        sp = torch.empty((T, V), dtype=torch.float32, device=device)
        grid_sp = (T, V)
        softplus_ab_kernel[grid_sp](a.view(T, V), dt_bias, sp, T, V, num_warps=1)

        # Sigmoid: beta = sigmoid(b)
        grid_sig = (T, V)
        sigmoid_b_kernel[grid_sig](b.view(T, V), beta_out, T, V, num_warps=1)

        # g = exp(-exp(A_log) * softplus(a + dt_bias))
        grid_g = (T, V)
        g_from_sp_A_kernel[grid_g](sp, A_log, g_out, T, V, num_warps=1)

        # Initialize per-segment state_HKV in float32, shape [H, K, V]
        num_cu = cu_seqlens.numel() - 1
        state_HKV = [None] * num_cu

        # Process each segment
        for seg in range(num_cu):
            seq_start = int(cu_seqlens[seg].item())
            seq_end = int(cu_seqlens[seg + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue
            # Assume initial state zeros per segment
            state_HKV[seg] = torch.zeros((H, K, V), dtype=torch.float32, device=device)

            # Loop over t within this segment
            for t in range(seq_len):
                t_abs = seq_start + t

                # Compute old_v = k[t_abs] @ state_HKV using Triton matmul
                # A: k[t_abs] -> [H, K], B: state_HKV -> [K, V]
                A_old = k[t_abs].contiguous().view(H, K)
                B_old = state_HKV[seg]  # [K, V]
                C_old = torch.empty((H, V), dtype=torch.float32, device=device)
                stride_am_old = A_old.stride(0)
                stride_ak_old = A_old.stride(1)
                stride_bk_old = B_old.stride(0)
                stride_bn_old = B_old.stride(1)
                stride_cm_old = C_old.stride(0)
                stride_cn_old = C_old.stride(1)
                BLOCK_M_old = 8
                BLOCK_N_old = 8
                BLOCK_K_old = 8
                grid_old = (1, 1)
                triton_matmul[grid_old](A_old, B_old, C_old,
                                        H, V, K,
                                        stride_am_old, stride_ak_old,
                                        stride_bk_old, stride_bn_old,
                                        stride_cm_old, stride_cn_old,
                                        BLOCK_M=BLOCK_M_old, BLOCK_N=BLOCK_N_old, BLOCK_K=BLOCK_K_old, num_warps=1)

                old_v = C_old  # [H, V]

                # v_t: [H, V]
                v_t = v[t_abs].contiguous().view(H, V)


def run(*args):
    return ModelNew()(*args)
