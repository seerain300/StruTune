import torch
import triton
import triton.language as tl

@triton.jit
def batched_matmul_kernel(
    A_ptr,  # *const float, shape [S, H], S = M
    B_ptr,  # *const float, shape [H, H], i.e., weight.T
    C_ptr,  # *float, shape [S, H]
    S: tl.constexpr, H: tl.constexpr,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (batch, tiles over M=S, tiles over N=H)
    pid_b = tl.program_id(0)  # not used directly in indexing here, but indicates per-batch behavior
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for boundary
    mask_m = offs_m < S
    mask_n = offs_n < H

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K (hidden_dim)
    for k in range(0, H, BLOCK_K):
        k_idx = k + offs_k
        mask_k = k_idx < H

        # A is [S, H] with strides (A_stride_m, A_stride_k)
        a_ptrs = A_ptr + (offs_m[:, None] * A_stride_m + k_idx[None, :] * A_stride_k)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # B is [H, H] with strides (B_stride_k, B_stride_n)
        b_ptrs = B_ptr + (k_idx[:, None] * B_stride_k + offs_n[None, :] * B_stride_n)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a, b)

    # Store result to C
    c_ptrs = C_ptr + (offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def stack_copy_kernel(
    src_ptr,  # *const float, shape [S, H], S can be T or I
    dst_ptr,  # *float, shape [B, T or I, H] (depending on stream)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    src_stride_m, src_stride_n,
    dst_stride_b, dst_stride_m, dst_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STREAM: tl.constexpr,  # 0 for encoder, 1 for hidden
):
    # Grid: (1, tiles over M=S, tiles over N=H). We index src for (m,n) and write to dst[b, m, n]
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < S
    mask_n = offs_n < H

    # We assume dst has leading batch dimension B. We iterate over b to place into correct output.
    for b in range(0, B):
        src_ptrs = src_ptr + (offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n)
        vals = tl.load(src_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        dst_ptrs = dst_ptr + (b * dst_stride_b + offs_m[:, None] * dst_stride_m + offs_n[None, :] * dst_stride_n)
        tl.store(dst_ptrs, vals, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of:
          1) Concatenate along sequence dim: [B, T+I, H]
          2) Linear projection: @ process_weight.T
          3) Split back into [B, T, H] and [B, I, H]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors."
        assert hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16), "Expected floating dtype."
        assert encoder_hidden_states.dtype == hidden_states.dtype and process_weight.dtype == hidden_states.dtype, "All tensors should share dtype."

        # Ensure contiguous and float32 for stable Triton kernels (no torch operations)
        device = hidden_states.device
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        # Process weight should be [H, H], as in the original code.
        assert process_weight.shape == (H, H), "process_weight must have shape [hidden_dim, hidden_dim]"

        # Cast to float32 for computation; keep a reference of original dtype for potential casting back if needed
        hidden_states_f = hidden_states.contiguous().to(torch.float32)
        encoder_hidden_states_f = encoder_hidden_states.contiguous().to(torch.float32)
        process_weight_f = process_weight.contiguous().to(torch.float32)

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=torch.float32)

        # 1) Compute C_e[b] = encoder_hidden_states[b] @ process_weight.T, result [B, T, H]
        # A_e shape: [B*T, H] -> we iterate per batch, so A_e[b, :, :] = encoder_hidden_states[b, :, :]
        # Build A_e contiguous [B, T, H] then treat as [S_e=M, K=H], but simpler: per-batch GEMM below
        # We launch one GEMM per batch for the encoder slice
        # We need A_e pointer of shape [T, H]; we can simply do:
        # A_e[b] = encoder_hidden_states_f[b] (already [T, H])
        # For Triton, pass pointer directly.
        # We'll do a grid over (B, tiles over T, tiles over H)
        BLOCK_T = 128
        BLOCK_H = 128
        BLOCK_K = 64
        grid_e = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))
        batched_matmul_kernel[grid_e](
            encoder_hidden_states_f,  # A_ptr: [B, T, H] but we index per b; kernel assumes A is [M, K], here M=T
            process_weight_f.t(),     # B_ptr: [H, H]
            processed_encoder,        # C_ptr: [B, T, H]
            T, H,
            encoder_hidden_states_f.stride(1), encoder_hidden_states_f.stride(2),
            process_weight_f.t().stride(0), process_weight_f.t().stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1),
            BLOCK_M=BLOCK_T, BLOCK_N=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 2) Compute C_i[b] = hidden_states[b] @ process_weight.T, result [B, I, H]
        BLOCK_I = 128
        BLOCK_H2 = 128
        BLOCK_K2 = 64
        grid_i = (B, triton.cdiv(I, BLOCK_I), triton.cdiv(H, BLOCK_H2))
        batched_matmul_kernel[grid_i](
            hidden_states_f,  # A_ptr: [B, I, H] → for each b, A is [I, H]
            process_weight_f.t(),  # B_ptr: [H, H]
            processed_hidden,      # C_ptr: [B, I, H]
            I, H,
            hidden_states_f.stride(1), hidden_states_f.stride(2),
            process_weight_f.t().stride(0), process_weight_f.t().stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1),
            BLOCK_M=BLOCK_I, BLOCK_N=BLOCK_H2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=3,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
