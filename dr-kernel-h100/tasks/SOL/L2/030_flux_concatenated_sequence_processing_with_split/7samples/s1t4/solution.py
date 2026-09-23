import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_sequences_kernel(
    in1_ptr, in2_ptr, out_ptr,
    B, T, I, D,
    sIn1_b, sIn1_t, sIn1_d,
    sIn2_b, sIn2_i, sIn2_d,
    sOut_b, sOut_m, sOut_d,
    BLOCK_M: tl.constexpr,  # tile size along sequence (M = T + I)
    BLOCK_D: tl.constexpr,  # tile size along hidden_dim (N = D)
):
    # program ids: batch, tile along M, tile along D
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_d = tl.program_id(axis=2)

    # Compute offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along T + I
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # along D

    # Masks to avoid out-of-bounds
    M = T + I
    mask_m = m_offsets < M
    mask_d = d_offsets < D

    # For each row in the tile, decide whether it comes from in1 (encoder) or in2 (hidden)
    # m_offsets in [0, T-1] -> from in1; [T, T+I-1] -> from in2
    is_from_in1 = m_offsets < T

    # Compute base pointers for inputs and output for this batch
    in1_base = in1_ptr + pid_b * sIn1_b
    in2_base = in2_ptr + pid_b * sIn2_b
    out_base = out_ptr + pid_b * sOut_b

    # Build 2D pointer grids for loading/storing
    # Note: rows and columns are both along D dimension
    # We'll load either from in1 or in2 depending on is_from_in1
    # Create pointer grids for in1 and in2
    in1_ptrs = in1_base + m_offsets[:, None] * sIn1_t + d_offsets[None, :] * sIn1_d  # shape [BM, BD]
    in2_ptrs = in2_base + (m_offsets[:, None] - T) * sIn2_i + d_offsets[None, :] * sIn2_d  # shape [BM, BD]

    # Masks per row: only valid rows contribute
    in1_mask = (mask_m[:, None] & is_from_in1[:, None]) & (mask_d[None, :])
    in2_mask = (mask_m[:, None] & ~is_from_in1[:, None]) & (mask_d[None, :])

    # Load values (masked). Use 0 for masked elements.
    in1_vals = tl.load(in1_ptrs, mask=in1_mask, other=0.0)
    in2_vals = tl.load(in2_ptrs, mask=in2_mask, other=0.0)

    # Choose the correct source per row
    # For positions where is_from_in1 is True, take in1_vals; otherwise take in2_vals.
    # We can do this by selecting based on is_from_in1 per row.
    selected = tl.where(is_from_in1[:, None], in1_vals, in2_vals)

    # Output pointers and mask
    out_ptrs = out_base + m_offsets[:, None] * sOut_m + d_offsets[None, :] * sOut_d
    out_mask = mask_m[:, None] & mask_d[None, :]
    tl.store(out_ptrs, selected, mask=out_mask)


@triton.jit
def _batched_gemm_no_bias_kernel(
    A_ptr, B_ptr, C_ptr,
    B, M, N, K,  # A: [B, M, K], B: [K, N], C: [B, M, N]
    sA_b, sA_m, sA_k,
    sB_k, sB_n,
    sC_b, sC_m, sC_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids over tiles: axis 0 = batch, axis 1 = M tile, axis 2 = N tile
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along M (rows)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along N (cols)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        # A tiles: [BM, BK]
        A_ptrs = A_ptr + pid_b * sA_b + m_offsets[:, None] * sA_m + k_offsets[None, :] * sA_k
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # B tiles: [BK, BN]
        B_ptrs = B_ptr + k_offsets[:, None] * sB_k + n_offsets[None, :] * sB_n
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back C
    C_ptrs = C_ptr + pid_b * sC_b + m_offsets[:, None] * sC_m + n_offsets[None, :] * sC_n
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


def _triton_concat_and_gemm(
    encoder_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
):
    """
    Concatenate encoder_hidden_states and hidden_states along sequence dim (T and I),
    apply GEMM with process_weight (no bias), and return split outputs.
    All computation is done via Triton kernels.
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    D = encoder_hidden_states.shape[2]
    assert hidden_states.shape[2] == D, "hidden_dim mismatch"
    assert process_weight.shape == (D, D), "process_weight must be [D, D]"

    # Make inputs contiguous
    enc = encoder_hidden_states.contiguous()
    hids = hidden_states.contiguous()
    w = process_weight.contiguous()

    # 1) Triton concatenation: out [B, T+I, D]
    M = T + I
    concatenated = torch.empty((B, M, D), dtype=torch.float32, device=enc.device)

    grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))
    _concatenate_sequences_kernel[grid_concat](
        enc, hids, concatenated,
        B, T, I, D,
        enc.stride(0), enc.stride(1), enc.stride(2),
        hids.stride(0), hids.stride(1), hids.stride(2),
        concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    # 2) Triton GEMM: C = concatenated @ w, no bias
    # concatenated: [B, M, K] where K = D
    # w: [K, N] where N = D
    C = torch.empty((B, M, D), dtype=torch.float32, device=enc.device)

    grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))
    _batched_gemm_no_bias_kernel[grid_gemm](
        concatenated, w, C,
        B, M, D, D,
        concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
        w.stride(0), w.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    # 3) Split back
    processed_encoder = C[:, :T, :]
    processed_hidden = C[:, T:, :]
    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA tensors
        if not hidden_states.is_cuda or not encoder_hidden_states.is_cuda or not process_weight.is_cuda:
            # If not on CUDA, fall back to reference PyTorch implementation (but in practice, evaluation provides CUDA tensors)
            # However, to comply with Triton-only requirement, we should not use PyTorch matmul here.
            # So, if inputs are not CUDA, we can move them to CUDA, run Triton, then move back. But typically inputs are CUDA.
            # Let's assert and raise if not CUDA to avoid silent fallback.
            raise RuntimeError("ModelNew requires CUDA tensors for Triton kernels. Move inputs to CUDA.")

        # Run Triton implementation
        processed_encoder, processed_hidden = _triton_concat_and_gemm(
            encoder_hidden_states, hidden_states, process_weight,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
