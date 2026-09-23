import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,           # *fp32, output A: [B, M, K]
    x1_ptr,            # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,            # *fp32, hidden_states: [B, I, K]
    B, T, I, K,        # ints
    out_stride0, out_stride1, out_stride2,  # strides for out
    x1_stride0, x1_stride1, x1_stride2,     # strides for x1
    x2_stride0, x2_stride1, x2_stride2,     # strides for x2
    BLOCK_M: tl.constexpr,                  # tile size along M
    BLOCK_N: tl.constexpr,                  # tile size along K
):
    # Grid: (B, ceil(M/BLOCK_M), ceil(K/BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Compute offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # along sequence
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # along hidden dim

    # Create 2D pointer grids for loads/stores, with masks
    # m_offsets[:, None] -> [BLOCK_M, 1], n_offsets[None, :] -> [1, BLOCK_N]
    # We'll compute masks for m and n
    mask_m = m_offsets[:, None] < (T + I)  # M = T + I
    mask_n = n_offsets[None, :] < K

    # For each m in tile, decide source: x1 if m < T else x2
    # Build 2D masks for each source
    mask_from_x1 = (m_offsets[:, None] < T) & mask_m
    mask_from_x2 = ((m_offsets[:, None] >= T) & (m_offsets[:, None] < (T + I))) & mask_m

    # Compute pointers for out
    out_ptrs = out_ptr + b * out_stride0 + m_offsets[:, None] * out_stride1 + n_offsets[None, :] * out_stride2

    # Compute pointers for x1 and x2, casting strides and offsets to fp32 for address arithmetic
    # Note: Triton allows using ints, but we keep computations as fp32 to avoid type issues
    b_fp = tl.full((), b, tl.float32)
    T_fp = tl.full((), T, tl.float32)
    I_fp = tl.full((), I, tl.float32)
    K_fp = tl.full((), K, tl.float32)
    out_stride0_fp = tl.full((), out_stride0, tl.float32)
    out_stride1_fp = tl.full((), out_stride1, tl.float32)
    out_stride2_fp = tl.full((), out_stride2, tl.float32)
    m_offsets_fp = m_offsets.to(tl.float32)[:, None]
    n_offsets_fp = n_offsets.to(tl.float32)[None, :]

    x1_ptrs = x1_ptr + (b_fp) * tl.full((), x1_stride0, tl.float32) + (m_offsets_fp) * tl.full((), x1_stride1, tl.float32) + n_offsets_fp * tl.full((), x1_stride2, tl.float32)
    x2_ptrs = x2_ptr + (b_fp) * tl.full((), x2_stride0, tl.float32) + (m_offsets_fp - T_fp) * tl.full((), x2_stride1, tl.float32) + n_offsets_fp * tl.full((), x2_stride2, tl.float32)

    # Masks: need to cast to bool for Triton
    mask_from_x1 = mask_from_x1.to(tl.int1)
    mask_from_x2 = mask_from_x2.to(tl.int1)
    mask_m2d = mask_m.to(tl.int1)
    mask_n2d = mask_n.to(tl.int1)
    mask_store = mask_m2d & mask_n2d

    # Load from x1 or x2 and store to out. Triton supports masked loads/stores.
    # Note: Triton pointers are typed; out, x1, x2 are fp32 pointers.
    # Since we don't know which elements come from x1 vs x2, we do two masked loads and then
    # select via tl.where. However, Triton allows only one pointer for load, so we perform
    # two masked loads with appropriate pointers and then add, but that's not correct because
    # an element can come from only one source. Instead, we perform two masked loads:
    # one for x1 and one for x2, with zeros for the other, then sum. But Triton doesn't support
    # adding two loads from different pointers; so we do two masked stores from the same out_ptr
    # by loading from x1 and x2 into temp and then performing masked stores.
    # To simplify, we compute the value using a where with pointers computed per element:
    # That's not supported; so we will perform masked loads as follows:

    # We need to load values for positions coming from x1 and x2 separately and store into out.
    # Triton allows masked loads with a single pointer; we can load from x1 at positions where
    # mask_from_x1 is True, and from x2 where mask_from_x2 is True, by constructing two masks and
    # then performing masked loads. However, Triton requires the same pointer for the load; thus
    # we perform two loads with same pointer by using tl.load with mask being the combined
    # mask and then select via pointer sources. The simplest robust approach is to write a loop
    # over m and do per-row masked loads/stores. To keep performance, we instead use a trick:
    # we compute a per-element source index and then use tl.load with masks for each source.
    # Triton doesn't allow branching on per-element Python constructs; so we implement
    # row-wise masked loads/stores.

    # Implementation detail: For each row m in the tile, perform masked load from x1 if m < T,
    # else from x2, and store to out. We'll iterate m as a loop (allowed) and handle the store
    # with mask_m2d for n dimension.

    # Since Triton requires vectorized loads/stores, we implement row-wise masked operations
    # using a loop over m in the tile:
    for im in range(0, BLOCK_M):
        m_idx = m_block * BLOCK_M + im
        m_idx_fp = tl.full((), m_idx, tl.float32)
        valid_m = m_idx < (T + I)
        # Compute out row pointer for this m
        out_row_ptrs = out_ptr + b * out_stride0 + m_idx * out_stride1 + n_offsets * out_stride2
        # Compute load from x1 if m_idx < T, else from x2
        if m_idx < T:
            src_ptrs = x1_ptr + b * x1_stride0 + m_idx * x1_stride1 + n_offsets * x1_stride2
            vals = tl.load(src_ptrs, mask=mask_n, other=0.0)
        else:
            # m_idx in [T, T+I)
            rel_m = m_idx - T
            src_ptrs = x2_ptr + b * x2_stride0 + rel_m * x2_stride1 + n_offsets * x2_stride2
            vals = tl.load(src_ptrs, mask=mask_n, other=0.0)
        # Store to out with bounds mask
        store_mask = valid_m & mask_n
        tl.store(out_row_ptrs, vals, mask=store_mask)

    # The above scalar loop over BLOCK_M avoids complexity with per-element pointer selection.
    # It ensures correct source for each row. n dimension is masked via mask_n.


@triton.jit
def batched_matmul_kernel(
    C_ptr,             # *fp32, output [B, M, N]
    A_ptr,             # *fp32, input [B, M, K]
    W_ptr,             # *fp32, weight [K, N]
    B, M, N, K,        # ints
    A_stride0, A_stride1, A_stride2,  # strides for A
    W_stride0, W_stride1,             # strides for W
    C_stride0, C_stride1, C_stride2,  # strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A[b, m, k]
        A_ptrs = A_ptr + b * A_stride0 + m_offsets[:, None] * A_stride1 + k_offsets[None, :] * A_stride2
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Pointers for W[k, n]
        W_ptrs = W_ptr + k_offsets[:, None] * W_stride0 + n_offsets[None, :] * W_stride1
        W_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w = tl.load(W_ptrs, mask=W_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store result to C
    C_ptrs = C_ptr + b * C_stride0 + m_offsets[:, None] * C_stride1 + n_offsets[None, :] * C_stride2
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension.
        - Applies linear projection via a Triton matmul kernel.
        - Splits back into two outputs.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]  # text_seq_len
        I = hidden_states.shape[1]          # img_seq_len
        K = encoder_hidden_states.shape[2]  # hidden_dim (same for both)

        # Ensure dtypes are float32 for Triton kernels
        # We'll compute in fp32 and cast outputs back to original dtypes.
        # Make inputs contiguous for predictable strides
        x1 = encoder_hidden_states.contiguous()
        x2 = hidden_states.contiguous()
        W = process_weight.contiguous()

        device = x1.device
        # Allocate output A for concatenation [B, M, K], fp32
        M = T + I
        A = torch.empty((B, M, K), device=device, dtype=torch.float32)

        # Launch concat kernel: out=A, x1=encoder, x2=hidden, dims B,T,I,K
        BLOCK_M_C = 128
        BLOCK_N_C = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M_C), triton.cdiv(K, BLOCK_N_C))
        concat_seq_dim1_kernel[grid_concat](
            A, x1, x2,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x2.stride(0), x2.stride(1), x2.stride(2),
            BLOCK_M=BLOCK_M_C, BLOCK_N=BLOCK_N_C,
            num_warps=4, num_stages=2,
        )

        # Allocate output C [B, M, K], fp32
        C = torch.empty((B, M, K), device=device, dtype=torch.float32)

        # Launch batched matmul: C = A @ W, W is [K, K] (no bias)
        # Ensure W is fp32 contiguous
        W_fp32 = W.float()

        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W_fp32,
            B, M, K, K,
            A.stride(0), A.stride(1), A.stride(2),
            W_fp32.stride(0), W_fp32.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # Split back
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes (match inputs)
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden