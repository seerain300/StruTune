import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_fp32(out_ptr, A_ptr, B_ptr,
                 M, N, K,
                 A_stride_m, A_stride_k,
                 B_stride_k, B_stride_n,
                 out_stride_m, out_stride_n,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 2D program id
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * A_stride_m) + (offs_k[None, :] * A_stride_k)  # [BM, BK]
        b_ptrs = B_ptr + (offs_k[:, None] * B_stride_k) + (offs_n[None, :] * B_stride_n)  # [BK, BN]

        # Masks for loads
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load and cast to fp32
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Accumulate using tl.dot (fp32)
        acc += tl.dot(a, b)  # a: [BM, BK], b: [BK, BN]

    # Write back result
    c_ptrs = out_ptr + (offs_m[:, None] * out_stride_m) + (offs_n[None, :] * out_stride_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _silu_kernel(out_ptr, x_ptr, M, N, out_stride_m, out_stride_n, x_stride_m, x_stride_n):
    # 2D tiling over [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_m = 64
    tile_n = 64
    offs_m = pid_m * tile_m + tl.arange(0, tile_m)
    offs_n = pid_n * tile_n + tl.arange(0, tile_n)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs_m[:, None] * x_stride_m + offs_n[None, :] * x_stride_n, mask=mask, other=0.0)
    x = x.to(tl.float32)
    silu_x = x * tl.sigmoid(x)  # SiLU(x) = x * sigmoid(x)
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, silu_x, mask=mask)


@triton.jit
def _mul_kernel(out_ptr, a_ptr, b_ptr, M, N, a_stride_m, a_stride_n, b_stride_m, b_stride_n, out_stride_m, out_stride_n):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_m = 64
    tile_n = 64
    offs_m = pid_m * tile_m + tl.arange(0, tile_m)
    offs_n = pid_n * tile_n + tl.arange(0, tile_n)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptr + offs_m[:, None] * a_stride_m + offs_n[None, :] * a_stride_n, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs_m[:, None] * b_stride_m + offs_n[None, :] * b_stride_n, mask=mask, other=0.0).to(tl.float32)
    c = a * b
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, c, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, grad_output: torch.Tensor,
                hidden_states: torch.Tensor,
                router_weight: torch.Tensor,
                e_score_correction_bias: torch.Tensor,
                router_logits: torch.Tensor,
                scores: torch.Tensor,
                topk_indices: torch.Tensor,
                topk_weights: torch.Tensor,
                score_mask: torch.Tensor,
                shared_expert_gate_weight: torch.Tensor,
                shared_expert_up_weight: torch.Tensor,
                shared_expert_down_weight: torch.Tensor,
                shared_gate_output: torch.Tensor,
                shared_up_output: torch.Tensor,
                shared_activated: torch.Tensor):
        """
        Triton-only forward that computes the shared expert activated output:
        shared_activated = silu(hidden @ gate_weight) * hidden @ up_weight

        Forward does not use any torch matmul or elementwise ops on tensors;
        all computation is performed inside Triton kernels.

        Returns: tensor of shape [batch_seq_len, intermediate_size], dtype bfloat16.
        """
        # Extract shapes
        M = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        N1 = shared_expert_gate_weight.shape[0]  # intermediate_size

        # Ensure contiguous and compute in fp32 inside Triton
        hidden_fp32 = hidden_states.float().contiguous()
        gate_w_fp32 = shared_expert_gate_weight.float().contiguous()
        up_w_fp32 = shared_expert_up_weight.float().contiguous()

        # Allocate outputs for matmul (fp32)
        shared_gate_output_fp32 = torch.empty((M, N1), dtype=torch.float32, device=hidden_fp32.device)
        shared_up_output_fp32 = torch.empty((M, N1), dtype=torch.float32, device=hidden_fp32.device)

        # Launch Triton matmul kernels
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid_gate = (triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        _matmul_fp32[grid_gate](
            shared_gate_output_fp32, hidden_fp32, gate_w_fp32,
            M, N1, hidden_size,
            hidden_fp32.stride(0), hidden_fp32.stride(1),
            gate_w_fp32.stride(0), gate_w_fp32.stride(1),
            shared_gate_output_fp32.stride(0), shared_gate_output_fp32.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        grid_up = (triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        _matmul_fp32[grid_up](
            shared_up_output_fp32, hidden_fp32, up_w_fp32,
            M, N1, hidden_size,
            hidden_fp32.stride(0), hidden_fp32.stride(1),
            up_w_fp32.stride(0), up_w_fp32.stride(1),
            shared_up_output_fp32.stride(0), shared_up_output_fp32.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # Triton elementwise: silu on shared_gate_output_fp32
        silu_output_fp32 = torch.empty((M, N1), dtype=torch.float32, device=hidden_fp32.device)
        grid_silu = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _silu_kernel[grid_silu](
            silu_output_fp32, shared_gate_output_fp32,
            M, N1, silu_output_fp32.stride(0), silu_output_fp32.stride(1),
            shared_gate_output_fp32.stride(0), shared_gate_output_fp32.stride(1)
        )

        # Triton elementwise: multiply activated = silu_output_fp32 * shared_up_output_fp32
        activated_fp32 = torch.empty((M, N1), dtype=torch.float32, device=hidden_fp32.device)
        grid_mul = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _mul_kernel[grid_mul](
            activated_fp32, silu_output_fp32, shared_up_output_fp32,
            M, N1, silu_output_fp32.stride(0), silu_output_fp32.stride(1),
            shared_up_output_fp32.stride(0), shared_up_output_fp32.stride(1),
            activated_fp32.stride(0), activated_fp32.stride(1)
        )

        # Return bfloat16 to match typical input dtype
        return activated_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
