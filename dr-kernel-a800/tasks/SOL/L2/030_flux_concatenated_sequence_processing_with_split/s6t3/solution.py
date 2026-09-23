import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seq_kernel(
    A_ptr,       # *f32, [B, M, H]
    B_ptr,       # *f32, [B, N, H]
    Out_ptr,     # *f32, [B, C, H], C = M + N
    B: tl.constexpr,        # batch size
    M: tl.constexpr,        # text_seq_len
    N: tl.constexpr,        # img_seq_len
    H: tl.constexpr,        # hidden_dim
    stride_ab,   # int: stride for batch in A
    stride_am,   # int: stride for seq in A
    stride_ak,   # int: stride for hidden in A
    stride_bb,   # int: stride for batch in B
    stride_bn,   # int: stride for seq in B
    stride_bh,   # int: stride for hidden in B
    stride_ob,   # int: stride for batch in Out
    stride_oc,   # int: stride for seq in Out
    stride_oh,   # int: stride for hidden in Out
    BLOCK_M: tl.constexpr,  # tile size along sequence (C)
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    seq_block = tl.program_id(1)

    # sequence offsets this program handles
    m_offsets = seq_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    C = M + N
    mask_m = m_offsets < C

    # Determine which positions come from A (text) and which from B (image)
    from_A = m_offsets < M  # True for indices [0..M-1], False for [M..M+N-1]

    # Compute hidden offsets
    h = tl.arange(0, H)  # [H]

    # Build pointers for Out: Out[b, m, h]
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h[None, :] * stride_oh  # [BLOCK_M, H]
    store_mask = mask_m[:, None]  # mask across sequence rows; hidden dim is full if H <= H

    # Load from A where applicable: A[b, m, h]
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h[None, :] * stride_ak
    a_mask = (mask_m[:, None]) & (h[None, :] < H)
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # Load from B where applicable: B[b, m-M, h]
    b_ptrs = B_ptr + b * stride_bb + (m_offsets - M)[:, None] * stride_bn + h[None, :] * stride_bh
    b_mask = (~from_A[:, None]) & (mask_m[:, None]) & (h[None, :] < H)
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Select based on from_A
    vals = tl.where(from_A[:, None], a_vals, b_vals)  # [BLOCK_M, H]

    # Store to Out
    tl.store(out_ptrs, vals, mask=store_mask)


@triton.jit
def _batched_matmul_kernel(
    X_ptr,  # *f32, [B, C, K], where C = M + N
    W_ptr,  # *f32, [K, K]  (process_weight)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length
    K: tl.constexpr,      # hidden dim (constexpr)
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    stride_wk0, # int: stride along K0 for W (rows)
    stride_wk1, # int: stride along K1 for W (cols)
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile over C (sequences)
    BLOCK_N: tl.constexpr,  # tile over K (hidden dim)
    BLOCK_K: tl.constexpr,  # tile over reduction dim K
):
    # 3D grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # tile offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # masks for boundaries
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load A tile: X[b, m, k] -> [BLOCK_M, BLOCK_K]
        a_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load B tile: W[k, n] -> [BLOCK_K, BLOCK_N]
        b_ptrs = W_ptr + k_offsets[:, None] * stride_wk0 + n_offsets[None, :] * stride_wk1
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    # Store result to P[b, m, n] = acc
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton
        - Performs batched matmul X @ process_weight.T in Triton
        - Splits result back into encoder and image streams
        """
        # Ensure CUDA and dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "This Triton implementation expects float32 tensors"

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H, "Hidden dims must match"

        # Ensure contiguous
        A = encoder_hidden_states.contiguous()   # [B, M, H]
        B_img = hidden_states.contiguous()       # [B, N, H]
        W = process_weight.contiguous()          # [H, H]
        C = M + N

        # Allocate output of concatenation X [B, C, H]
        X = torch.empty((B, C, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch concatenation kernel
        BLOCK_M = 128  # tile size along sequence; grid will be ceil_div(C, BLOCK_M)
        grid = (B, triton.cdiv(C, BLOCK_M))
        _concatenate_seq_kernel[grid](
            A, B_img, X,
            B, M, N, H,
            A.stride(0), A.stride(1), A.stride(2),
            B_img.stride(0), B_img.stride(1), B_img.stride(2),
            X.stride(0), X.stride(1), X.stride(2),
            BLOCK_M=BLOCK_M,
            num_warps=4,
        )

        # Allocate output of GEMM P [B, C, H]
        P = torch.empty((B, C, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch GEMM kernel: P = X @ W^T, where W is [H, H]
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(C, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _batched_matmul_kernel[grid](
            X, W, P,
            B, C, H,  # C is M + N, H is hidden size (constexpr specialization)
            X.stride(0), X.stride(1), X.stride(2),
            W.stride(0), W.stride(1),
            P.stride(0), P.stride(1), P.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # Split back along sequence dimension
        processed_encoder = P[:, :M, :]   # [B, M, H]
        processed_hidden = P[:, M:, :]    # [B, N, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
