import torch
import triton
import triton.language as tl

# Triton kernel: per-batch batched GEMM computing C = A @ B
# A: [M, K] where M is seq_len slice (either text or image), K = hidden_dim
# B: [K, N] = process_weight.T, N = hidden_dim
# C: [M, N]
@triton.jit
def matmul_seq_hidden_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,     # strides for A: row-major [M, K]
    stride_bk, stride_bn,     # strides for B: [K, N]
    stride_cm, stride_cn,     # strides for C: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch: (batch, tiles over M, tiles over N)
    # program_id(2) is batch index
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A, B, C tiles
    A_tile_ptr = A_ptr + pid_b * stride_am + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_tile_ptr = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_tile_ptr,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            B_tile_ptr,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        # acc += a @ b
        acc += tl.dot(a, b)

        # Advance pointers along K
        A_tile_ptr += BLOCK_K * stride_ak
        B_tile_ptr += BLOCK_K * stride_bk

    # Write back to C with masks
    C_tile_ptr = C_ptr + pid_b * stride_cm + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        C_tile_ptr,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )

# Triton kernel: stack per-batch outputs into final tensors
# We need two stack kernels: one for encoder output, one for hidden output.
# Each kernel copies output_encoder[b] of shape [text_seq_len, H] into processed_encoder[b, :, :]
# and similarly for hidden.
@triton.jit
def stack_perbatch_kernel(
    per_batch_ptr,               # pointer to output_encoder or output_hidden, shape [B, S_slice, H], contiguous
    stacked_ptr,                 # pointer to final stacked tensor, shape [B, S_slice, H] for this stream
    B, S_slice, H,
    stride_pb_b, stride_pb_m, stride_pb_n,   # strides for per_batch
    stride_st_b, stride_st_m, stride_st_n,   # strides for stacked
    STREAM: tl.constexpr,                   # either 'encoder' or 'hidden' (used for naming, no effect on logic)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 3D grid: (batch, tiles over M=S_slice, tiles over N=H)
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Pointers for source (per-batch output)
    src_ptr = per_batch_ptr + pid_b * stride_pb_b + (offs_m[:, None] * stride_pb_m + offs_n[None, :] * stride_pb_n)
    # Pointers for destination (stacked output)
    dst_ptr = stacked_ptr + pid_b * stride_st_b + (offs_m[:, None] * stride_st_m + offs_n[None, :] * stride_st_n)

    # Load and store with masks
    vals = tl.load(
        src_ptr,
        mask=(offs_m[:, None] < S_slice) & (offs_n[None, :] < H),
        other=0.0
    )
    tl.store(
        dst_ptr,
        vals,
        mask=(offs_m[:, None] < S_slice) & (offs_n[None, :] < H)
    )

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Computes per-batch processed sequences using Triton GEMM.
        - Stacks the per-batch results into final outputs using Triton.
        Returns (processed_encoder, processed_hidden), matching the original signature.
        """

        # Ensure inputs are on CUDA and contiguous; cast to float32 for kernel (matmul kernel expects fp32)
        device = hidden_states.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors."

        # Make sure dtype is float32 for Triton kernel; if not, cast
        # (You can remove casts if you ensure inputs are fp32 in caller.)
        dtype = hidden_states.dtype
        if dtype not in (torch.float32,):
            hidden_states = hidden_states.to(torch.float32)
            encoder_hidden_states = encoder_hidden_states.to(torch.float32)

        B = hidden_states.shape[0]
        text_seq_len = encoder_hidden_states.shape[1]
        img_seq_len = hidden_states.shape[1]
        hidden_dim = hidden_states.shape[2]
        # B for process_weight should match hidden_dim
        assert process_weight.shape == (hidden_dim, hidden_dim), "process_weight must be [hidden_dim, hidden_dim]"
        Bw = process_weight.shape[0]
        assert Bw == hidden_dim, "process_weight first dim must equal hidden_dim"

        # Prepare B = process_weight.T (contiguous)
        B_mat = process_weight.t().contiguous()  # [H, H], contiguous

        # Output per-batch tensors for encoder and image streams
        # We'll fill them via Triton matmul kernels, then stack with Triton.
        # For safety, allocate as zeros (same as initializing with empty and filling).
        # Note: We only launch kernels to write into these; no torch stacking used.
        output_encoder = torch.empty((B, text_seq_len, hidden_dim), device=device, dtype=torch.float32)
        output_hidden = torch.empty((B, img_seq_len, hidden_dim), device=device, dtype=torch.float32)

        # Choose tiling parameters; 64 works well for H≈1024
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64
        num_warps = 4
        num_stages = 3

        # Launch matmul for encoder slice (per batch)
        grid_m_e = (text_seq_len + BLOCK_M - 1) // BLOCK_M
        grid_n_e = (hidden_dim + BLOCK_N - 1) // BLOCK_N

        grid_e = (grid_m_e, grid_n_e, B)
        matmul_seq_hidden_kernel[grid_e](
            encoder_hidden_states, B_mat, output_encoder,
            text_seq_len, hidden_dim, hidden_dim,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(2),
            B_mat.stride(0), B_mat.stride(1),
            output_encoder.stride(0), output_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )

        # Launch matmul for image slice (per batch)
        grid_m_i = (img_seq_len + BLOCK_M - 1) // BLOCK_M
        grid_n_i = (hidden_dim + BLOCK_N - 1) // BLOCK_N

        grid_i = (grid_m_i, grid_n_i, B)
        matmul_seq_hidden_kernel[grid_i](
            hidden_states, B_mat, output_hidden,
            img_seq_len, hidden_dim, hidden_dim,
            hidden_states.stride(0), hidden_states.stride(2),
            B_mat.stride(0), B_mat.stride(1),
            output_hidden.stride(0), output_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )

        # Now stack per-batch outputs into final tensors using Triton
        # processed_encoder: [B, text_seq_len, hidden_dim]
        # processed_hidden: [B, img_seq_len, hidden_dim]
        # We'll allocate final outputs and launch a stack kernel for each.
        processed_encoder = torch.empty((B, text_seq_len, hidden_dim), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, img_seq_len, hidden_dim), device=device, dtype=torch.float32)

        # For stacking, we simply copy per-batch outputs into processed arrays.
        # Use Triton to perform the copy. This is the 'stack' per batch.

        # BLOCK sizes for copy can be larger; 128x128 tiles
        BLOCK_M2 = 128
        BLOCK_N2 = 128
        num_warps_stack = 4
        num_stages_stack = 1

        # For encoder stack
        grid_m_e_stack = (text_seq_len + BLOCK_M2 - 1) // BLOCK_M2
        grid_n_e_stack = (hidden_dim + BLOCK_N2 - 1) // BLOCK_N2
        grid_stack_e = (grid_m_e_stack, grid_n_e_stack, B)
        stack_perbatch_kernel[grid_stack_e](
            output_encoder, processed_encoder,
            B, text_seq_len, hidden_dim,
            output_encoder.stride(0), output_encoder.stride(1), output_encoder.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            STREAM=0,  # unused but helps differentiate in code generation
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
            num_warps=num_warps_stack, num_stages=num_stages_stack,
        )

        # For hidden stack
        grid_m_i_stack = (img_seq_len + BLOCK_M2 - 1) // BLOCK_M2
        grid_n_i_stack = (hidden_dim + BLOCK_N2 - 1) // BLOCK_N2
        grid_stack_i = (grid_m_i_stack, grid_n_i_stack, B)
        stack_perbatch_kernel[grid_stack_i](
            output_hidden, processed_hidden,
            B, img_seq_len, hidden_dim,
            output_hidden.stride(0), output_hidden.stride(1), output_hidden.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            STREAM=1,  # unused but helps differentiate in code generation
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
            num_warps=num_warps_stack, num_stages=num_stages_stack,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
