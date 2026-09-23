import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,          # *fp32, output A: [B, M, K]
    x1_ptr,           # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,           # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr,  # batch size
    T: tl.constexpr,  # text_seq_len
    I: tl.constexpr,  # img_seq_len
    K: tl.constexpr,  # hidden_dim
    OUT_sB, OUT_sM, OUT_sK,
    X1_sB, X1_sT, X1_sK,
    X2_sB, X2_sI, X2_sK,
):
    # 1D grid across total elements of A: total = B * M * K
    pid = tl.program_id(0)
    total = B * (T + I) * K
    if pid >= total:
        return

    # Compute (b, m, k) via integer arithmetic
    B_times_MK = B * (T + I)
    b = pid // ( (T + I) * K )
    rem = pid % ( (T + I) * K )
    m = rem // K
    k = rem % K

    # Determine source tensor: x1 for m < T, x2 for m >= T
    is_encoder = m < T

    # Load from appropriate input and store to A
    out_offset = b * OUT_sB + m * OUT_sM + k * OUT_sK
    if is_encoder:
        x_offset = b * X1_sB + m * X1_sT + k * X1_sK
        val = tl.load(x1_ptr + x_offset)
    else:
        x_offset = b * X2_sB + (m - T) * X2_sI + k * X2_sK
        val = tl.load(x2_ptr + x_offset)
    tl.store(out_ptr + out_offset, val)


@triton.jit
def batched_matmul_kernel_3d(
    C_ptr,           # *fp32, output [B, M, K]
    A_ptr,           # *fp32, input [B, M, K]
    W_ptr,           # *fp32, weight [K, K]
    B: tl.constexpr, # batch size
    M: tl.constexpr, # M = T + I
    K: tl.constexpr, # hidden_dim
    A_stride_b, A_stride_m, A_stride_k,
    W_stride0, W_stride1,
    C_stride_b, C_stride_m, C_stride_k,
    BLOCK_M: tl.constexpr,  # tile along M (rows of C)
    BLOCK_N: tl.constexpr,  # tile along K (output dim n)
    BLOCK_K: tl.constexpr,  # tile along K (reduction)
):
    # 3D launch grid: (B, tiles over M, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    mask_m = m_offsets < M
    mask_n = n_offsets < K

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + k_offsets
        mask_k = k_idx < K

        # Load A[b, m, k] tile -> shape (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_m + k_idx[None, :] * A_stride_k
        A_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W[k, n] tile -> shape (BLOCK_K, BLOCK_N)
        W_ptrs = W_ptr + k_idx[:, None] * W_stride0 + n_offsets[None, :] * W_stride1
        W_mask = mask_k[:, None] & mask_n[None, :]
        w = tl.load(W_ptrs, mask=W_mask, other=0.0)

        # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, w)

    # Store results
    C_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_k
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized implementation:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim (dim=1) into A[b, M, K] using Triton.
        2) Compute C = A @ process_weight.T using a Triton batched matmul kernel.
        3) Split C back into processed_encoder and processed_hidden.
        Returns: (processed_encoder: [B, T, K], processed_hidden: [B, I, K])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # Prepare inputs
        # Ensure contiguous for predictable strides
        encoder = encoder_hidden_states.contiguous()
        image = hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        # 1) Allocate A of shape [B, M, K], where M = T + I
        M = T + I
        A = torch.empty((B, M, K), dtype=torch.float32, device=hidden_states.device)

        # 2) Launch Triton concat kernel: A = [encoder along seq, image along seq]
        # Grid size: 1D over total elements
        total_elems = B * M * K
        grid_concat = (total_elems,)
        concat_seq_dim1_kernel[grid_concat](
            A, encoder, image,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            image.stride(0), image.stride(1), image.stride(2),
            num_warps=4, num_stages=2,
        )

        # 3) Prepare weight W as [K, K] and cast to fp32 for compute
        W = process_weight  # shape [K, K]
        if W.dtype != torch.float32:
            W = W.float()

        # 4) Allocate C [B, M, K] and run Triton GEMM
        C = torch.empty((B, M, K), dtype=torch.float32, device=hidden_states.device)

        # Tile sizes: tuneable; these are reasonable defaults
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel_3d[grid_gemm](
            C, A, W,
            B, M, K,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 5) Split outputs
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Match original input dtypes
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden