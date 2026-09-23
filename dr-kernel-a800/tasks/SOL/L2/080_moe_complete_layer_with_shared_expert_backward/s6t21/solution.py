import torch
import triton
import triton.language as tl


# Triton GEMM-style row-tiled kernel: computes Y = X @ W^T
# X: [M, K] bfloat16, W: [K, N] bfloat16, Y: [M, N] float32
@triton.jit
def matmul_rowtile_bf16_to_f32(
    X_ptr, W_ptr, Y_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids for 2D launch grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K dimension in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # pointers for X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)  # bfloat16

        # pointers for W tile: [BLOCK_K, BLOCK_N] (load along N)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # bfloat16

        # accumulate into fp32
        # x: [BM, BK], w: [BK, BN] -> tl.dot(x, w) = [BM, BN]
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    # store result tile
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


# Triton elementwise kernel: y = gate_out * sigmoid(gate_out) * up_out
# gate_out, up_out: [M, N] float32, y: [M, N] float32
@triton.jit
def silu_mul_elementwise_f32(
    Gate_ptr, Up_ptr, Out_ptr,
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
):
    pid = tl.program_id(0)
    total = M * N
    # simple 1D grid; each program handles multiple elements
    idx = pid * 128 + tl.arange(0, 128)
    mask = idx < total

    # compute m, n from linear index
    n = idx % N
    m = idx // N

    g_ptrs = Gate_ptr + m * stride_gm + n * stride_gn
    u_ptrs = Up_ptr + m * stride_um + n * stride_un
    o_ptrs = Out_ptr + m * stride_om + n * stride_on

    gate = tl.load(g_ptrs, mask=mask, other=0.0)
    up = tl.load(u_ptrs, mask=mask, other=0.0)

    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-gate))
    y = gate * sig * up
    tl.store(o_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight):
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda, "All tensors must be on CUDA device"
        hidden_states = hidden_states.contiguous()
        gate_w = shared_expert_gate_weight.contiguous()  # [K, N]
        up_w = shared_expert_up_weight.contiguous()     # [K, N]

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = gate_w.shape[1]  # 1408

        # Outputs as float32 (for numerical stability), Triton-only compute
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch matmul row-tile kernels
        # X: [M, K] bfloat16, W: [K, N] bfloat16
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_rowtile_bf16_to_f32[grid](
            hidden_states, gate_w, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3
        )

        matmul_rowtile_bf16_to_f32[grid](
            hidden_states, up_w, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_w.stride(0), up_w.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3
        )

        # Elementwise: shared_activated = SiLU(gate_out) * up_out
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid2 = (triton.cdiv(M * N, 128),)
        silu_mul_elementwise_f32[grid2](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return as bfloat16 (final cast constructor, not elementwise op)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
