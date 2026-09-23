import torch
import triton
import triton.language as tl


# Kernel 1: Concatenate along sequence dimension.
# A: hidden_states [B, I, H], starting at concatenated[:, T:, :]
# B: encoder_hidden_states [B, T, H], starting at concatenated[:, :T, :]
# C: concatenated [B, T+I, H]
@triton.jit
def concatenate_along_seq_kernel(
    A_ptr,  # hidden_states
    B_ptr,  # encoder_hidden_states
    C_ptr,  # concatenated
    B: tl.constexpr,  # batch size
    I: tl.constexpr,  # img_seq_len
    T: tl.constexpr,  # text_seq_len
    H: tl.constexpr,  # hidden_dim
    strideA_b, strideA_i, strideA_h,
    strideB_b, strideB_t, strideB_h,
    strideC_b, strideC_seq, strideC_h,
    BLOCK_seq: tl.constexpr,  # tile size over sequence length
    BLOCK_h: tl.constexpr,    # tile size over hidden dim
):
    pid_b = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Compute tile ranges
    seq_start = pid_seq * BLOCK_seq
    h_start = pid_h * BLOCK_h

    seq_idx = seq_start + tl.arange(0, BLOCK_seq)
    h_idx = h_start + tl.arange(0, BLOCK_h)

    M = T + I

    # Masks for bounds
    mask_seq = seq_idx < M
    mask_h = h_idx < H

    # Base pointers for batch
    A_base = A_ptr + pid_b * strideA_b
    B_base = B_ptr + pid_b * strideB_b
    C_base = C_ptr + pid_b * strideC_b

    # For each seq position, determine which source to read from:
    # seq < T -> from B (encoder), else -> from A (hidden) at index seq - T
    # We will load two candidates and then select. Since Triton doesn't support tl.where on pointers, we compute the index for each via masks.
    # Create masks: from_encoder = (seq < T), from_hidden = (~from_encoder)
    from_encoder = seq_idx < T

    # Compute indices for hidden: since hidden index for those seq >= T is (seq - T), but we only need where from_hidden is true, use masked select for value. We'll load via masks by constructing index vectors.
    # Construct two index vectors: idx_b for encoder and idx_i for hidden.
    idx_b = seq_idx  # encoder indices
    idx_i = seq_idx - T  # hidden indices when from_hidden

    # Build full 2D pointers for loads: [BLOCK_seq, BLOCK_h]
    # For encoder: ptrs_b = B_base + idx_b[:, None] * strideB_t + h_idx[None, :] * strideB_h
    ptrs_b = B_base + idx_b[:, None] * strideB_t + h_idx[None, :] * strideB_h
    # For hidden: ptrs_a = A_base + idx_i[:, None] * strideA_i + h_idx[None, :] * strideA_h
    ptrs_a = A_base + idx_i[:, None] * strideA_i + h_idx[None, :] * strideA_h

    # Masks for loads
    mask_b = mask_seq[:, None] & mask_h[None, :] & from_encoder[:, None]
    mask_i = mask_seq[:, None] & mask_h[None, :] & (~from_encoder)[:, None]

    # Load values, defaulting masked elements to 0
    vals_b = tl.load(ptrs_b, mask=mask_b, other=0.0)
    vals_i = tl.load(ptrs_a, mask=mask_i, other=0.0)

    # Select: for positions where from_encoder is true, vals_b; else vals_i
    # Using tl.where on 2D: where mask_b -> vals_b, where mask_i -> vals_i
    # Since at each (seq,h), exactly one mask is true, this is fine.
    vals = tl.where(from_encoder[:, None], vals_b, vals_i)

    # Store to C at sequence index = seq_idx
    C_ptrs = C_base + seq_idx[:, None] * strideC_seq + h_idx[None, :] * strideC_h
    tl.store(C_ptrs, vals, mask=(mask_seq[:, None] & mask_h[None, :]))


# Kernel 2: Matmul for concatenated [B, L, H] and W [H, H], output [B, L, H].
# We flatten M = B*L in one grid dim and N = H in another. Reduction is over K = H.
@triton.jit
def matmul_cat_wT_kernel(
    A_ptr,  # concatenated [B, L, H]
    W_ptr,  # process_weight [H, H]
    C_ptr,  # output [B, L, H]
    B: tl.constexpr,  # batch
    L: tl.constexpr,  # total sequence length = T + I
    H: tl.constexpr,  # hidden dim
    strideA_b, strideA_l, strideA_h,
    strideW_h, strideW_k,
    strideC_b, strideC_l, strideC_h,
    BLOCK_M: tl.constexpr,  # tile size over M = B*L
    BLOCK_N: tl.constexpr,  # tile size over N = H
    BLOCK_K: tl.constexpr,  # tile size over K = H
):
    pid_m = tl.program_id(0)  # over M tiles
    pid_n = tl.program_id(1)  # over N tiles

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_idx = m_start + tl.arange(0, BLOCK_M)  # over B*L
    n_idx = n_start + tl.arange(0, BLOCK_N)  # over H

    # Bounds masks
    mask_m = m_idx < (B * L)
    mask_n = n_idx < H

    # Compute b and l from m_idx
    b_idx = m_idx // L
    l_idx = m_idx % L

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K (hidden dimension)
    for k_start in range(0, H, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_idx < H

        # Load A tile: A[b, l, k] -> shape [BLOCK_M, BLOCK_K]
        # Pointer arithmetic: A_ptr + b_idx[:, None]*strideA_b + l_idx[:, None]*strideA_l + k_idx[None, :]*strideA_h
        A_ptrs = A_ptr + b_idx[:, None] * strideA_b + l_idx[:, None] * strideA_l + k_idx[None, :] * strideA_h
        mask_A = mask_m[:, None] & mask_k[None, :]
        A_tile = tl.load(A_ptrs, mask=mask_A, other=0.0)

        # Load W tile: W[k, n] -> shape [BLOCK_K, BLOCK_N]
        # Pointer arithmetic: W_ptr + k_idx[:, None]*strideW_k + n_idx[None, :]*strideW_h
        W_ptrs = W_ptr + k_idx[:, None] * strideW_k + n_idx[None, :] * strideW_h
        mask_W = mask_k[:, None] & mask_n[None, :]
        W_tile = tl.load(W_ptrs, mask=mask_W, other=0.0)

        # acc += A_tile @ W_tile
        acc += tl.dot(A_tile, W_tile)

    # Store results to C[b, l, n]
    # Compute output pointers: C_ptr + b_idx[:, None]*strideC_b + l_idx[:, None]*strideC_l + n_idx[None, :]*strideC_h
    C_ptrs = C_ptr + b_idx[:, None] * strideC_b + l_idx[:, None] * strideC_l + n_idx[None, :] * strideC_h
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=mask_out)


# Kernel 3: Slice the processed [B, L, H] into processed_encoder [B, T, H]
@triton.jit
def slice_to_encoder_kernel(
    Input_ptr,  # processed [B, L, H]
    Output_ptr, # processed_encoder [B, T, H]
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    strideI_b, strideI_l, strideI_h,
    strideO_b, strideO_t, strideO_h,
    BLOCK_t: tl.constexpr,
    BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    t_start = pid_t * BLOCK_t
    h_start = pid_h * BLOCK_h

    t_idx = t_start + tl.arange(0, BLOCK_t)
    h_idx = h_start + tl.arange(0, BLOCK_h)

    mask_t = t_idx < T
    mask_h = h_idx < H

    Input_base = Input_ptr + pid_b * strideI_b
    Output_base = Output_ptr + pid_b * strideO_b

    Input_ptrs = Input_base + t_idx[:, None] * strideI_l + h_idx[None, :] * strideI_h
    Output_ptrs = Output_base + t_idx[:, None] * strideO_t + h_idx[None, :] * strideO_h

    vals = tl.load(Input_ptrs, mask=(mask_t[:, None] & mask_h[None, :]), other=0.0)
    tl.store(Output_ptrs, vals, mask=(mask_t[:, None] & mask_h[None, :]))


# Kernel 4: Slice the processed [B, L, H] into processed_hidden [B, I, H]
@triton.jit
def slice_to_hidden_kernel(
    Input_ptr,  # processed [B, L, H]
    Output_ptr, # processed_hidden [B, I, H]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    strideI_b, strideI_l, strideI_h,
    strideO_b, strideO_i, strideO_h,
    BLOCK_i: tl.constexpr,
    BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    i_start = pid_i * BLOCK_i
    h_start = pid_h * BLOCK_h

    i_idx = i_start + tl.arange(0, BLOCK_i)
    h_idx = h_start + tl.arange(0, BLOCK_h)

    mask_i = i_idx < I
    mask_h = h_idx < H

    Input_base = Input_ptr + pid_b * strideI_b
    Output_base = Output_ptr + pid_b * strideO_b

    # Input sequence index for hidden stream is l = T + i
    Input_ptrs = Input_base + (T + i_idx)[:, None] * strideI_l + h_idx[None, :] * strideI_h
    Output_ptrs = Output_base + i_idx[:, None] * strideO_i + h_idx[None, :] * strideO_h

    vals = tl.load(Input_ptrs, mask=(mask_i[:, None] & mask_h[None, :]), other=0.0)
    tl.store(Output_ptrs, vals, mask=(mask_i[:, None] & mask_h[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation of:
            concatenated = cat([encoder_hidden_states, hidden_states], dim=1)
            processed = concatenated @ process_weight.T
            processed_encoder = processed[:, :text_seq_len, :]
            processed_hidden = processed[:, text_seq_len:, :]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Triton kernels require CUDA tensors"

        B = hidden_states.shape[0]
        I = hidden_states.shape[1]  # img_seq_len
        T = encoder_hidden_states.shape[1]  # text_seq_len
        H = hidden_states.shape[2]  # hidden_dim
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H
        assert process_weight.shape[0] == H and process_weight.shape[1] == H

        # Ensure inputs are contiguous (Triton benefits from contiguous memory)
        hidden_states_c = hidden_states.contiguous()
        encoder_hidden_states_c = encoder_hidden_states.contiguous()
        process_weight_c = process_weight.contiguous()

        # 1) Concatenate along sequence dimension using Triton
        L = T + I
        concatenated = torch.empty((B, L, H), dtype=hidden_states_c.dtype, device=hidden_states_c.device)

        # Launch grid for concatenation: (B, ceil((T+I)/BLOCK_seq), ceil(H/BLOCK_h))
        BLOCK_seq = 128
        BLOCK_h = 64
        grid_concat = (B, triton.cdiv(L, BLOCK_seq), triton.cdiv(H, BLOCK_h))
        concatenate_along_seq_kernel[grid_concat](
            hidden_states_c, encoder_hidden_states_c, concatenated,
            B, I, T, H,
            hidden_states_c.stride(0), hidden_states_c.stride(1), hidden_states_c.stride(2),
            encoder_hidden_states_c.stride(0), encoder_hidden_states_c.stride(1), encoder_hidden_states_c.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_seq=BLOCK_seq, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: concatenated [B, L, H] @ process_weight.T [H, H] -> [B, L, H]
        processed = torch.empty((B, L, H), dtype=concatenated.dtype, device=concatenated.device)

        # Triton GEMM grid: flatten M over B*L
        BLOCK_M = 64   # tile size over B*L
        BLOCK_N = 64   # tile size over H
        BLOCK_K = 32   # tile size over H (reduction)
        grid_matmul = (triton.cdiv(B * L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_cat_wT_kernel[grid_matmul](
            concatenated, process_weight_c, processed,
            B, L, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight_c.stride(0), process_weight_c.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split processed back into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, T, H), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, I, H), dtype=processed.dtype, device=processed.device)

        # Launch slice to encoder
        BLOCK_t = 128
        BLOCK_h = 64
        grid_encoder = (B, triton.cdiv(T, BLOCK_t), triton.cdiv(H, BLOCK_h))
        slice_to_encoder_kernel[grid_encoder](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_t=BLOCK_t, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # Launch slice to hidden
        BLOCK_i = 128
        BLOCK_h = 64
        grid_hidden = (B, triton.cdiv(I, BLOCK_i), triton.cdiv(H, BLOCK_h))
        slice_to_hidden_kernel[grid_hidden](
            processed, processed_hidden,
            B, T, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_i=BLOCK_i, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
