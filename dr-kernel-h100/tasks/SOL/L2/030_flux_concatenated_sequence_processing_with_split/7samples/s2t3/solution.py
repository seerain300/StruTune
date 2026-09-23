import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    src_e_ptr, src_i_ptr, dst_ptr,
    B, T, I, H,
    src_e_stride_b, src_e_stride_t, src_e_stride_h,
    src_i_stride_b, src_i_stride_i, src_i_stride_h,
    dst_stride_b, dst_stride_t, dst_stride_h,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr,
):
    # grid = (B, ceil(T/BLOCK_T), ceil(I/BLOCK_I))
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    mask_t = offs_t < T
    mask_i = offs_i < I

    # For each batch, copy encoder slice into dst[b, :T, :]
    base_e = pid_b * src_e_stride_b
    e_ptrs = src_e_ptr + base_e + offs_t[:, None] * src_e_stride_t + tl.arange(0, H)[None, :] * src_e_stride_h
    e_vals = tl.load(e_ptrs, mask=mask_t[:, None], other=0.0)  # [BLOCK_T, H]
    dst_base_e = pid_b * dst_stride_b
    dst_e_ptrs = dst_ptr + dst_base_e + offs_t[:, None] * dst_stride_t + tl.arange(0, H)[None, :] * dst_stride_h
    tl.store(dst_e_ptrs, e_vals, mask=mask_t[:, None])

    # Copy image slice into dst[b, T:, :]
    base_i = pid_b * src_i_stride_b
    i_ptrs = src_i_ptr + base_i + offs_i[:, None] * src_i_stride_i + tl.arange(0, H)[None, :] * src_i_stride_h
    i_vals = tl.load(i_ptrs, mask=mask_i[:, None], other=0.0)  # [BLOCK_I, H]
    dst_base_i = dst_base_e + T * dst_stride_t
    dst_i_ptrs = dst_ptr + dst_base_i + offs_i[:, None] * dst_stride_t + tl.arange(0, H)[None, :] * dst_stride_h
    tl.store(dst_i_ptrs, i_vals, mask=mask_i[:, None])


@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    S, H,  # A is [S, H], B is [H, H], C is [S, H]
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # grid = (BATCH, ceil(S/BLOCK_M), ceil(H/BLOCK_N))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A/C
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C/N
    k_offsets = tl.arange(0, BLOCK_K)

    # Pointers to tiles
    A_tile_ptrs = A_ptr + pid_b * A_stride_m + m_offsets[:, None] * A_stride_k + k_offsets[None, :] * 0  # k will be broadcast
    B_tile_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, H, BLOCK_K):
        k = k0 + k_offsets
        a = tl.load(
            A_ptr + pid_b * A_stride_m + m_offsets[:, None] * A_stride_k + k[None, :] * 1,
            mask=(m_offsets[:, None] < S) & (k[None, :] < H),
            other=0.0,
        )
        b = tl.load(
            B_ptr + k[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n,
            mask=(k[:, None] < H) & (n_offsets[None, :] < H),
            other=0.0,
        )
        acc += tl.dot(a, b)

    # Store
    C_tile_ptrs = C_ptr + pid_b * C_stride_m + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    tl.store(
        C_tile_ptrs,
        acc,
        mask=(m_offsets[:, None] < S) & (n_offsets[None, :] < H),
    )


@triton.jit
def split_kernel(
    src_ptr, dst_encoder_ptr, dst_hidden_ptr,
    B, T, I, H,
    src_stride_b, src_stride_t, src_stride_h,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr,
):
    # grid = (B, ceil(T/BLOCK_T), ceil(I/BLOCK_I))
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    mask_t = offs_t < T
    mask_i = offs_i < I

    # Copy encoder part: src[b, :T, :] -> dst_encoder[b, :T, :]
    base_src = pid_b * src_stride_b
    src_ptrs_e = src_ptr + base_src + offs_t[:, None] * src_stride_t + tl.arange(0, H)[None, :] * src_stride_h
    vals_e = tl.load(src_ptrs_e, mask=mask_t[:, None], other=0.0)
    base_enc = pid_b * enc_stride_b
    enc_ptrs = dst_encoder_ptr + base_enc + offs_t[:, None] * enc_stride_t + tl.arange(0, H)[None, :] * enc_stride_h
    tl.store(enc_ptrs, vals_e, mask=mask_t[:, None])

    # Copy hidden part: src[b, T:, :] -> dst_hidden[b, :I, :]
    base_src_i = base_src + T * src_stride_t
    src_ptrs_h = src_ptr + base_src_i + offs_i[:, None] * src_stride_t + tl.arange(0, H)[None, :] * src_stride_h
    vals_h = tl.load(src_ptrs_h, mask=mask_i[:, None], other=0.0)
    base_hid = pid_b * hid_stride_b
    hid_ptrs = dst_hidden_ptr + base_hid + offs_i[:, None] * hid_stride_i + tl.arange(0, H)[None, :] * hid_stride_h
    tl.store(hid_ptrs, vals_h, mask=mask_i[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function:
        - Concatenates [B, T, H] and [B, I, H] along sequence dim in Triton.
        - Applies GEMM A @ B where A is concatenated [B*(T+I), H], B is process_weight.T [H, H].
        - Splits the result back into [B, T, H] and [B, I, H] in Triton.
        All computation is done in Triton; forward only manages allocations and launches.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        assert hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16), "Unsupported dtype"
        # For simplicity and correctness, compute in float32; cast inputs if necessary
        # Note: Triton kernels below assume float32; we enforce this to avoid dtype issues.
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H
        assert process_weight.shape[0] == H and process_weight.shape[1] == H

        # 1) Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton
        concatenated = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

        BLOCK_T = 128
        BLOCK_I = 128
        grid_concat = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(I, BLOCK_I))
        concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_I=BLOCK_I,
        )

        # 2) Prepare A and B for GEMM
        # A: concatenated [B*(T+I), H], row-major contiguous
        A = concatenated.reshape(B * (T + I), H).contiguous()
        # B: process_weight.T [H, H], contiguous
        B_weight = process_weight.t().contiguous()

        # Allocate C: [B*(T+I), H]
        C = torch.empty((B * (T + I), H), device=hidden_states.device, dtype=torch.float32)

        # 3) Batched GEMM in Triton: one kernel per batch (we use grid over batch, M-tiles, N-tiles)
        # Note: we use BLOCK_M = 64, BLOCK_N = 128, BLOCK_K = 64 for robustness across H sizes.
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(B * (T + I), BLOCK_M), triton.cdiv(H, BLOCK_N))
        batched_matmul_kernel[grid_gemm](
            A, B_weight, C,
            B * (T + I), H,
            A.stride(0), A.stride(1),
            B_weight.stride(0), B_weight.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 4) Split C back into encoder and hidden parts using Triton
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=torch.float32)

        BLOCK_T_split = 128
        BLOCK_I_split = 128
        grid_split = (B, triton.cdiv(T, BLOCK_T_split), triton.cdiv(I, BLOCK_I_split))
        split_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_T=BLOCK_T_split, BLOCK_I=BLOCK_I_split,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
