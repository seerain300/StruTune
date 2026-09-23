import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    in_e_ptr, in_i_ptr, out_ptr,
    B, T, I, H,
    in_e_stride_b, in_e_stride_s, in_e_stride_h,
    in_i_stride_b, in_i_stride_s, in_i_stride_h,
    out_stride_b, out_stride_s, out_stride_h,
):
    # Grid: (B, tiles over rows of concatenated sequence)
    # Each program handles one batch b and a block of concatenated rows.
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    S = T + I
    BLOCK_M = 256
    row_start = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = row_start < S

    # Compute source indices
    is_encoder = row_start < T
    src_row = tl.where(is_encoder, row_start, row_start - T)
    src_b = tl.full((BLOCK_M,), b, tl.int32)

    # Compute pointers for encoder and image parts
    e_ptrs = in_e_ptr + src_b * in_e_stride_b + src_row * in_e_stride_s + tl.arange(0, H) * in_e_stride_h
    i_ptrs = in_i_ptr + src_b * in_i_stride_b + src_row * in_i_stride_s + tl.arange(0, H) * in_i_stride_s

    # Load and store into out
    out_row = row_start
    out_ptrs = out_ptr + src_b * out_stride_b + out_row * out_stride_s + tl.arange(0, H) * out_stride_h
    # Mask for e: only rows < T
    e_vals = tl.load(e_ptrs, mask=mask & is_encoder, other=0.0)
    i_vals = tl.load(i_ptrs, mask=mask & (~is_encoder), other=0.0)
    vals = e_vals + i_vals  # non-encoder rows will have i_vals; encoder rows have e_vals and i_vals=0
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles over M, tiles over N)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        mask_k = k_ids < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * A_stride_m + offs_m[:, None] * A_stride_m + k_ids[None, :] * A_stride_k
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_ids[:, None] * B_stride_k + offs_n[None, :] * B_stride_n
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, b_tile)

    # Store results
    c_ptrs = C_ptr + b * C_stride_m + offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def split_kernel(
    C_ptr, out_e_ptr, out_i_ptr,
    B, T, I, H,
    C_stride_b, C_stride_s, C_stride_h,
    out_e_stride_b, out_e_stride_s, out_e_stride_h,
    out_i_stride_b, out_i_stride_s, out_i_stride_h,
    num_tiles_m: tl.constexpr,
    num_tiles_n: tl.constexpr,
):
    # Grid: (B, tiles over T, tiles over H) for encoder, and (B, tiles over I, tiles over H) for image.
    # Here we assume num_tiles_m covers S = T+I and num_tiles_n covers H.
    # We compute b, tile for T/H, and write into out_e/out_i.
    b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_nh = tl.program_id(2)

    BLOCK_T = 64
    BLOCK_H = 128
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_nh * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_t = offs_t < T
    mask_h = offs_h < H

    # For encoder: rows [0, T)
    e_ptrs = C_ptr + (b * (T + I) + offs_t) * C_stride_s + offs_h * C_stride_h
    e_vals = tl.load(e_ptrs, mask=mask_t[:, None] & mask_h[None, :], other=0.0)
    tl.store(out_e_ptr + b * out_e_stride_b + offs_t[:, None] * out_e_stride_s + offs_h[None, :] * out_e_stride_h, e_vals, mask=mask_t[:, None] & mask_h[None, :])

    # For image: rows [T, T+I)
    i_ptrs = C_ptr + (b * (T + I) + T + offs_t) * C_stride_s + offs_h * C_stride_h
    i_vals = tl.load(i_ptrs, mask=mask_t[:, None] & mask_h[None, :], other=0.0)
    tl.store(out_i_ptr + b * out_i_stride_b + offs_t[:, None] * out_i_stride_s + offs_h[None, :] * out_i_stride_h, i_vals, mask=mask_t[:, None] & mask_h[None, :])


@triton.jit
def stack_copy_kernel(
    src_ptr, dst_ptr,
    B, T, H,
    src_stride_b, src_stride_s, src_stride_h,
    dst_stride_b, dst_stride_s, dst_stride_h,
):
    # Grid: (B, tiles over T, tiles over H) — simply copy src[b, t, h] to dst[b, t, h]
    b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    BLOCK_T = 64
    BLOCK_H = 128
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_t = offs_t < T
    mask_h = offs_h < H

    src_ptrs = src_ptr + b * src_stride_b + offs_t[:, None] * src_stride_s + offs_h[None, :] * src_stride_h
    dst_ptrs = dst_ptr + b * dst_stride_b + offs_t[:, None] * dst_stride_s + offs_h[None, :] * dst_stride_h

    vals = tl.load(src_ptrs, mask=mask_t[:, None] & mask_h[None, :], other=0.0)
    tl.store(dst_ptrs, vals, mask=mask_t[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA and contiguous
        device = hidden_states.device
        assert device.type == "cuda", "Inputs must be on CUDA for Triton kernels."
        B, T, H = hidden_states.shape
        _, I, H2 = encoder_hidden_states.shape
        assert H == H2, "hidden_dim must match for hidden states and encoder hidden states"
        assert process_weight.shape == (H, H), "process_weight must be [hidden_dim, hidden_dim]"

        # Cast to float32 for kernel math
        hidden_states_f = hidden_states.contiguous().to(torch.float32)
        encoder_hidden_states_f = encoder_hidden_states.contiguous().to(torch.float32)
        process_weight_f = process_weight.contiguous().to(torch.float32)

        # 1) Concatenate sequences along sequence dimension into A_concat: [B*S, H], S = T + I
        S = T + I
        A_concat = torch.empty((B * S, H), device=device, dtype=torch.float32)

        grid_concat = (B, triton.cdiv(S, 256))
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states_f, hidden_states_f, A_concat,
            B, T, I, H,
            encoder_hidden_states_f.stride(0), encoder_hidden_states_f.stride(1), encoder_hidden_states_f.stride(2),
            hidden_states_f.stride(0), hidden_states_f.stride(1), hidden_states_f.stride(2),
            A_concat.stride(0), A_concat.stride(1), A_concat.stride(2),
        )

        # 2) Batched GEMM: for each batch b, A = A_concat[b*S:(b+1)*S, :], B = process_weight.T
        #    Output C per batch: [S, H]
        C_perbatch = torch.empty((B, S, H), device=device, dtype=torch.float32)

        grid_gemm = (B, triton.cdiv(S, 64), triton.cdiv(H, 128))
        batched_matmul_kernel[grid_gemm](
            A_concat, process_weight_f.t(), C_perbatch,
            S, H, H,
            A_concat.stride(0), A_concat.stride(1),
            process_weight_f.t().stride(0), process_weight_f.t().stride(1),
            C_perbatch.stride(0), C_perbatch.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64,
        )

        # 3) Split per-batch C into encoder [T, H] and image [I, H]
        C_split = torch.empty((B, T, H), device=device, dtype=torch.float32)
        C_split_img = torch.empty((B, I, H), device=device, dtype=torch.float32)

        # Use the number of tiles along M and N for this split; they match the GEMM tiling
        split_kernel[(B, triton.cdiv(T, 64), triton.cdiv(H, 128))](
            C_perbatch, C_split, C_split_img,
            B, T, I, H,
            C_perbatch.stride(0), C_perbatch.stride(1), C_perbatch.stride(2),
            C_split.stride(0), C_split.stride(1), C_split.stride(2),
            C_split_img.stride(0), C_split_img.stride(1), C_split_img.stride(2),
            num_tiles_m=triton.cdiv(S, 64), num_tiles_n=triton.cdiv(H, 128),
        )

        # 4) Stack per-batch outputs into final [B, T, H] and [B, I, H] using Triton kernel
        processed_encoder = torch.empty((B, T, H), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=torch.float32)

        grid_stack = (B, triton.cdiv(T, 64), triton.cdiv(H, 128))
        stack_copy_kernel[grid_stack](
            C_split, processed_encoder,
            B, T, H,
            C_split.stride(0), C_split.stride(1), C_split.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        )

        grid_stack_img = (B, triton.cdiv(I, 64), triton.cdiv(H, 128))
        stack_copy_kernel[grid_stack_img](
            C_split_img, processed_hidden,
            B, I, H,
            C_split_img.stride(0), C_split_img.stride(1), C_split_img.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
