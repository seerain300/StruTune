import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # *ptr to [B, T, H]
    img_ptr,       # *ptr to [B, I, H]
    A_ptr,         # *ptr to [M, H]
    B, T, I, H,    # sizes
    stride_b_e, stride_t_e, stride_h_e,  # strides for enc
    stride_b_i, stride_i_i, stride_h_i,  # strides for img
    stride_m_a, stride_h_a,               # strides for A (row-major: [M, H])
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program ids for tiling over rows M
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # row indices into A
    M_total = B * (T + I)
    mask_m = offs_m < M_total

    # compute batch and sequence index for each row
    b = offs_m // (T + I)  # vector of batch indices
    seq = offs_m % (T + I)  # vector of sequence indices

    # decide source tensor: seq < T -> encoder, else -> image
    is_encoder = seq < T
    # compute source row index in the source tensor: for encoder it's seq, for image it's seq - T
    src_row_e = seq  # for encoder
    src_row_i = seq - T  # for image
    # build pointers for loads (masked)
    ptr_e = enc_ptr + b * stride_b_e + src_row_e * stride_t_e + tl.arange(0, BLOCK_N) * stride_h_e
    ptr_i = img_ptr + b * stride_b_i + src_row_i * stride_i_i + tl.arange(0, BLOCK_N) * stride_h_i

    # broadcast masks: we need to load only valid m and valid n
    # We will perform a safe load for both enc and img using masks and then choose based on is_encoder.
    # However, Triton doesn't allow branching on a tensor mask for pointer selection. We therefore
    # set up two load operations guarded by scalar condition and select using tl.where,
    # but since Triton requires pointer-based load, we do two masked loads and combine via tl.where after.
    # A robust approach is to use two separate tiles (one for enc, one for img) but Triton allows only one pointer per vector.
    # So we perform two masked loads and then use tl.where to select per lane; but tl.where works on values, not pointers.
    # Instead, we perform one masked load per lane based on scalar condition by structuring loads this way:
    # We'll load from enc_ptr with mask mask_m & is_encoder, and from img_ptr with mask mask_m & ~is_encoder,
    # but Triton doesn't support conditional on pointer in tl.load. Therefore, we use tl.load with masks and then select:
    # Create a temporary dtype 0 vector for those lanes where we won't load.
    zeros = tl.zeros([BLOCK_N], dtype=tl.float32)
    vals_e = tl.load(ptr_e, mask=mask_m & is_encoder, other=0.0)
    vals_i = tl.load(ptr_i, mask=mask_m & (~is_encoder), other=0.0)
    vals = tl.where(is_encoder, vals_e, vals_i)

    # store to A: A is row-major [M, H], strides (stride_m_a, stride_h_a)
    A_ptrs = A_ptr + offs_m[:, None] * stride_m_a + tl.arange(0, BLOCK_N)[None, :] * stride_h_a
    store_mask = mask_m[:, None]
    tl.store(A_ptrs, vals[None, :], mask=store_mask)


@triton.jit
def batched_matmul_kernel(
    A_ptr,     # *ptr to [M, H], row-major
    B_ptr,     # *ptr to [K, H], here K=H and process_weight.T
    C_ptr,     # *ptr to [M, H], row-major
    M, H,      # sizes
    stride_m_a, stride_k_a,          # strides for A
    stride_k_b, stride_n_b,          # strides for B (process_weight.T with shape [H, H])
    stride_m_c, stride_n_c,          # strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # tile over M
    pid_n = tl.program_id(axis=1)  # tile over N=H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension (hidden_dim, here H)
    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_m_a + offs_k[None, :] * stride_k_a
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < H), other=0.0)

        # load B tile: shape [BLOCK_K, BLOCK_N], B is [H, H]
        b_ptrs = B_ptr + offs_k[:, None] * stride_k_b + offs_n[None, :] * stride_n_b
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < H) & (offs_n[None, :] < H), other=0.0)

        # accumulate
        acc += tl.dot(a, b)

    # store C tile
    c_ptrs = C_ptr + offs_m[:, None] * stride_m_c + offs_n[None, :] * stride_n_c
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < H))


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,           # *ptr to [M, H], source
    out_ptr,         # *ptr to [B, T, H], destination
    M, T, H,
    stride_m_c, stride_n_c,              # C strides
    out_stride_b, out_stride_t, out_stride_h,  # destination strides
    start_row,  # starting row in C for encoder: 0
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs < (B * T)

    # compute b and t for each row in C
    b = offs // T
    t = offs % T

    # load from C at row offs and columns [0:H)
    c_ptrs = C_ptr + offs[:, None] * stride_m_c + tl.arange(0, BLOCK_N)[None, :] * stride_n_c
    vals = tl.load(c_ptrs, mask=mask_m[:, None] & (tl.arange(0, BLOCK_N)[None, :] < H), other=0.0)

    # store to out[b, t, :]
    out_ptrs = out_ptr + b[:, None] * out_stride_b + t[:, None] * out_stride_t + tl.arange(0, BLOCK_N)[None, :] * out_stride_h
    tl.store(out_ptrs, vals, mask=mask_m[:, None])


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,           # *ptr to [M, H], source
    out_ptr,         # *ptr to [B, I, H], destination
    M, I, H,
    stride_m_c, stride_n_c,              # C strides
    out_stride_b, out_stride_i, out_stride_h,  # destination strides
    start_row,  # starting row in C for hidden: B*T
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs < (B * I)

    # compute b and i for each row in C
    b = (offs - start_row) // I
    i = (offs - start_row) % I

    # load from C at row offs and columns [0:H)
    c_ptrs = C_ptr + offs[:, None] * stride_m_c + tl.arange(0, BLOCK_N)[None, :] * stride_n_c
    vals = tl.load(c_ptrs, mask=mask_m[:, None] & (tl.arange(0, BLOCK_N)[None, :] < H), other=0.0)

    # store to out[b, i, :]
    out_ptrs = out_ptr + b[:, None] * out_stride_b + i[:, None] * out_stride_i + tl.arange(0, BLOCK_N)[None, :] * out_stride_h
    tl.store(out_ptrs, vals, mask=mask_m[:, None])


def _choose_block_n(H: int) -> int:
    # choose BLOCK_N as a multiple of H to ensure full column coverage and good performance
    # prefer 128, then 64, then 32 if needed
    if H >= 128 and H % 128 == 0:
        return 128
    if H >= 64 and H % 64 == 0:
        return 64
    if H >= 32 and H % 32 == 0:
        return 32
    # fallback: use largest among {128, 64, 32} that is <= H
    if H >= 128:
        return 128
    if H >= 64:
        return 64
    if H >= 32:
        return 32
    return 32


def _choose_block_sizes(H: int):
    # BLOCK_M and BLOCK_K choices
    # Keep BLOCK_M moderate to balance occupancy and register pressure
    if H >= 128:
        BLOCK_M = 64
    elif H >= 64:
        BLOCK_M = 64
    else:
        BLOCK_M = 32

    if H >= 128:
        BLOCK_K = 64
    elif H >= 64:
        BLOCK_K = 64
    else:
        BLOCK_K = 32

    BLOCK_N = _choose_block_n(H)
    return BLOCK_M, BLOCK_K, BLOCK_N


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension using a Triton kernel.
        - Applies a batched matmul with process_weight.T using a Triton kernel.
        - Splits the result back into encoder and hidden outputs using Triton copy kernels.
        All heavy computation is done by Triton kernels; no torch.cat or torch.matmul in forward.
        """
        # Ensure contiguous inputs
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight = process_weight.contiguous()  # [H, H]

        B, T, H = enc.shape
        I = img.shape[1]

        # 1) Build A [M, H] via Triton concatenation kernel, M = B * (T + I)
        M = B * (T + I)
        A = torch.empty((M, H), dtype=enc.dtype, device=enc.device)

        # Select block sizes for concat kernel
        BLOCK_M_A = 128
        BLOCK_N_A = 128  # large enough to cover H=128; for other H, we rely on mask
        grid_concat = (triton.cdiv(M, BLOCK_M_A),)

        # Launch concat kernel
        concat_rows_to_A_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M_A, BLOCK_N=BLOCK_N_A,
            num_warps=4, num_stages=2,
        )

        # 2) Compute C = A @ process_weight.T using Triton matmul kernel
        # Weight_T: [H, H]
        weight_T = weight.transpose(0, 1).contiguous()  # [H, H]

        C = torch.empty((M, H), dtype=enc.dtype, device=enc.device)

        H_dim = H
        BLOCK_M, BLOCK_K, BLOCK_N = _choose_block_sizes(H_dim)

        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(H_dim, BLOCK_N))
        batched_matmul_kernel[grid_matmul](
            A, weight_T, C,
            M, H_dim,
            A.stride(0), A.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 3) Split C into processed_encoder [B, T, H] and processed_hidden [B, I, H] using Triton copy kernels
        processed_encoder = torch.empty((B, T, H), dtype=enc.dtype, device=enc.device)
        processed_hidden = torch.empty((B, I, H), dtype=enc.dtype, device=enc.device)

        # Copy first B*T rows of C into processed_encoder
        BLOCK_M_CP = 128
        BLOCK_N_CP = 128
        grid_copy_e = (triton.cdiv(B * T, BLOCK_M_CP),)
        copy_rows_encoder_kernel[grid_copy_e](
            C, processed_encoder,
            M, T, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row=0,
            BLOCK_M=BLOCK_M_CP, BLOCK_N=BLOCK_N_CP,
            num_warps=4, num_stages=2,
        )

        # Copy remaining B*I rows of C into processed_hidden
        grid_copy_h = (triton.cdiv(B * I, BLOCK_M_CP),)
        copy_rows_hidden_kernel[grid_copy_h](
            C, processed_hidden,
            M, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_row=B * T,
            BLOCK_M=BLOCK_M_CP, BLOCK_N=BLOCK_N_CP,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
