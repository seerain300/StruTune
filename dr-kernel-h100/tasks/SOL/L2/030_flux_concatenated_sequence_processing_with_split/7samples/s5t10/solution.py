import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,            # *const T: [B, T, H]
    i_ptr,            # *const T: [B, I, H]
    out_ptr,          # *T: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2, # strides for e_ptr
    i_s0, i_s1, i_s2, # strides for i_ptr
    o_s0, o_s1, o_s2, # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Program IDs: batch, sequence-l tiles, hidden-h tiles
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Compute index vectors
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # sequence positions in [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden dim indices

    # Create 2D indices for the tile
    L = l[:, None]  # shape [BLOCK_l, 1]
    H2 = h[None, :]  # shape [1, BLOCK_h]
    # Valid masks
    mask_l = L < (T + I)
    mask_h = H2 < H
    mask = mask_l & mask_h

    # Base pointers for output tile
    out_base = out_ptr + pid_b * o_s0 + L * o_s1 + H2 * o_s2

    # Determine source and offset
    is_encoder = L < T  # boolean [BLOCK_l, 1]
    # Pointers for encoder and image source
    e_base = e_ptr + pid_b * e_s0 + L * e_s1 + H2 * e_s2
    i_base = i_ptr + pid_b * i_s0 + (L - T) * i_s1 + H2 * i_s2  # L - T is always < I where is_encoder is False

    # Select source: where is_encoder, use e_base, else use i_base
    # Triton uses tl.where for elementwise selection
    src_ptr = tl.where(is_encoder, e_base, i_base)

    # Load and store (masked)
    val = tl.load(src_ptr, mask=mask, other=0.0)
    tl.store(out_base, val, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,    # *const T: [B*(T+I), H]
    W_ptr,    # *const T: [H, H] (process_weight.T)
    C_ptr,    # *T: [B*(T+I), H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    A_s0, A_s1,  # strides for A_ptr
    W_s0, W_s1,  # strides for W_ptr
    C_s0, C_s1,  # strides for C_ptr
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid over (M, N), M = B*(T+I), N = H
    pid_m = tl.program_id(0)  # tile along rows
    pid_n = tl.program_id(1)  # tile along columns

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    M = m_offsets[:, None]  # [BLOCK_M, 1]
    N = n_offsets[None, :]  # [1, BLOCK_N]

    # mask for valid rows/cols
    mask_m = M < (B * (T + I))
    mask_n = N < H
    mask_out = mask_m & mask_n

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K = H
    for k0 in range(0, H, BLOCK_K):
        K = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = K < H

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + M * A_s0 + K[None, :] * A_s1
        A_mask = mask_m[:, None] & mask_k[None, :]
        A = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # W^T tile: we want W[:, K] -> shape [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + K[:, None] * W_s0 + N * W_s1
        W_mask = mask_k[:, None] & mask_n[None, :]
        Wt = tl.load(W_ptrs, mask=W_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A.to(tl.float32), Wt.to(tl.float32))

    # Store result to C with mask
    C_ptrs = C_ptr + M * C_s0 + N * C_s1
    tl.store(C_ptrs, acc, mask=mask_out)


@triton.jit
def copy_rows_kernel(
    src_ptr, dest_ptr,
    B: tl.int32, L: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dest_s0, dest_s1, dest_s2,
    ROW_START: tl.constexpr,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Copies rows [ROW_START, ROW_START+L) from src to dest, both [B, ? , H]
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)

    # Valid masks
    mask_l = (l + ROW_START) < (ROW_START + L)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Build pointers
    src_rows = ROW_START + l  # [BLOCK_l]
    src_base = src_ptr + pid_b * src_s0 + src_rows[:, None] * src_s1 + h[None, :] * src_s2
    dest_base = dest_ptr + pid_b * dest_s0 + l[:, None] * dest_s1 + h[None, :] * dest_s2

    val = tl.load(src_base, mask=mask, other=0.0)
    tl.store(dest_base, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        Bsz = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate using Triton kernel: out [B, T+I, H]
        out = torch.empty((Bsz, T + I, H), device=device, dtype=dtype)

        # Launch concat kernel
        BLOCK_l = 64
        BLOCK_h = 64
        grid = (Bsz, triton.cdiv(T + I, BLOCK_l), triton.cdiv(H, BLOCK_h))
        concat_encoder_image_kernel[grid](
            encoder_hidden_states, hidden_states, out,
            Bsz, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(), *out.stride(),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = out @ process_weight.T, result [B, T+I, H]
        # Ensure process_weight is [H, H], matching the original code.
        W_t = process_weight.transpose(0, 1).contiguous()  # [H, H]
        # Allocate output tensor
        processed = torch.empty((Bsz, T + I, H), device=device, dtype=dtype)

        # Flatten M = B*(T+I)
        M = Bsz * (T + I)
        N = H
        # Strides for A [M, H], C [M, H], W [H, H]
        A = out  # shape [B, T+I, H]
        A_s0 = A.stride(0); A_s1 = A.stride(2)  # A is [M, H] logically; stride(0)=B*(T+I)*H, stride(2)=1
        W = W_t  # shape [H, H]
        C = processed

        # Choose tiling; H is typically <= 1024 in these workloads
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_kernel[grid_matmul](
            A, W, C,
            Bsz, T, I, H,
            A_s0, A_s1,
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split: copy first T rows to processed_encoder, next I rows to processed_hidden
        processed_encoder = torch.empty((Bsz, T, H), device=device, dtype=dtype)
        processed_hidden = torch.empty((Bsz, I, H), device=device, dtype=dtype)

        # Copy first T rows: [0, T)
        grid_first = (Bsz, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_first](
            processed, processed_encoder,
            Bsz, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows: [T, T+I)
        grid_second = (Bsz, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_second](
            processed, processed_hidden,
            Bsz, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
