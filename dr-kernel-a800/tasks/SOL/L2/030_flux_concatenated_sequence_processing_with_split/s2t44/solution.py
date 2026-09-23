import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seqs_kernel(
    enc_ptr,         # *T, [B, T, H]
    hid_ptr,         # *T, [B, I, H]
    out_ptr,         # *T, [B, L, H], L = T + I
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile along L (sequence)
    BLOCK_K: tl.constexpr,  # tile along H (feature)
):
    # Grid: (B, L, H)
    b = tl.program_id(0)
    m = tl.program_id(1)
    k = tl.program_id(2)

    # Bounds
    if b >= B:
        return
    if m >= (T + I) or k >= H:
        return

    # Select source: enc for m < T, hid for m >= T
    use_enc = m < T

    # Compute offsets and pointers
    # out[b, m, k]
    out_off = b * out_ptr.stride(0) + m * out_ptr.stride(1) + k * out_ptr.stride(2)
    # enc[b, m, k]
    enc_off = b * enc_ptr.stride(0) + m * enc_ptr.stride(1) + k * enc_ptr.stride(2)
    # hid[b, m - T, k]
    hid_off = b * hid_ptr.stride(0) + (m - T) * hid_ptr.stride(1) + k * hid_ptr.stride(2)

    val = tl.load(out_ptr + out_off)
    if use_enc:
        val = tl.load(enc_ptr + enc_off)
    else:
        val = tl.load(hid_ptr + hid_off)

    tl.store(out_ptr + out_off, val)


@triton.jit
def _batched_matmul_kernel_fp32(
    A_ptr,            # *T, [B, L, H]
    WT_ptr,           # *T, [H, H]
    C_ptr,            # *float32, [B, L, H]
    B: tl.constexpr,  # int
    M: tl.constexpr,  # int, M = L
    N: tl.constexpr,  # int, N = H (output feature)
    K: tl.constexpr,  # int, K = H (input feature)
    strideA_b, strideA_m, strideA_k,
    strideWT_p, strideWT_q,
    strideC_b, strideC_m, strideC_n,
    BLOCK_M: tl.constexpr,  # tile along M (sequence)
    BLOCK_N: tl.constexpr,  # tile along N (feature)
    BLOCK_K: tl.constexpr,  # tile along K (feature)
):
    # Grid: (B, ceil(M / BLOCK_M), ceil(N / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along M (sequence)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along N (feature)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: A[b, m, k]
        A_ptrs = A_ptr + b * strideA_b + m_offsets[:, None] * strideA_m + k_offsets[None, :] * strideA_k
        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        a = a.to(tl.float32)

        # Load WT tile: WT[k, n]
        WT_ptrs = WT_ptr + k_offsets[:, None] * strideWT_p + n_offsets[None, :] * strideWT_q
        wt = tl.load(WT_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        wt = wt.to(tl.float32)

        # Accumulate
        acc += tl.dot(a, wt)

    # Store result to C[b, m, n] (fp32)
    C_ptrs = C_ptr + b * strideC_b + m_offsets[:, None] * strideC_m + n_offsets[None, :] * strideC_n
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate encoder and hidden sequences into [B, L, H] via Triton
        - Apply linear projection via Triton batched GEMM: [B, L, H] @ [H, H]^T -> [B, L, H]
        - Split back into encoder and hidden streams
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, dim, H]"
        assert process_weight.dim() == 2 and process_weight.shape[1] == process_weight.shape[0], "process_weight must be square [H, H]"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        device = hidden_states.device

        # Ensure contiguous inputs
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()

        # 1) Triton concatenate into out_cat [B, L, H]
        out_cat = torch.empty((B, L, H), device=device, dtype=enc.dtype)

        # Launch Triton kernel: grid (B, L, H)
        BLOCK_M = 1  # one element per program along L for robustness; Triton grid spans all dims
        BLOCK_K = 1
        grid_concat = (B, L, H)
        _concatenate_seqs_kernel[grid_concat](
            enc, hid, out_cat,
            B, T, I, H,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=1, num_stages=1,
        )

        # 2) Prepare process_weight.T as fp32 for stable accumulation
        WT = process_weight.t().contiguous().to(torch.float32)  # [H, H]
        A = out_cat  # [B, L, H], dtype matches enc/hid

        # 3) Triton batched GEMM: C = A @ WT, output fp32 [B, L, H]
        C = torch.empty((B, L, H), device=device, dtype=torch.float32)

        strideA_b, strideA_m, strideA_k = A.stride()
        strideWT_p, strideWT_q = WT.stride()
        strideC_b, strideC_m, strideC_n = C.stride()

        # Tiling params (robust defaults; masks handle edges)
        BLOCK_M_mm = 64
        BLOCK_N_mm = 64
        BLOCK_K_mm = 32
        grid_mm = (B, triton.cdiv(L, BLOCK_M_mm), triton.cdiv(H, BLOCK_N_mm))

        _batched_matmul_kernel_fp32[grid_mm](
            A, WT, C,
            B, L, H, H,
            strideA_b, strideA_m, strideA_k,
            strideWT_p, strideWT_q,
            strideC_b, strideC_m, strideC_n,
            BLOCK_M=BLOCK_M_mm, BLOCK_N=BLOCK_N_mm, BLOCK_K=BLOCK_K_mm,
            num_warps=4, num_stages=3,
        )

        # 4) Split back into encoder and hidden streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
