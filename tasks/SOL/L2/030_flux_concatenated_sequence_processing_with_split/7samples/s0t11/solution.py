import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,        # *float32, output A: [B, M, K]
    x1_ptr,         # *float32, encoder_hidden_states: [B, T, K]
    x2_ptr,         # *float32, hidden_states: [B, I, K]
    B: tl.constexpr,
    T: tl.constexpr,  # text_seq_len
    I: tl.constexpr,  # img_seq_len
    K: tl.constexpr,  # hidden_dim
    BLOCK_M: tl.constexpr,  # tile size along sequence dim (M)
    BLOCK_N: tl.constexpr,  # tile size along hidden_dim (K)
):
    # Grid: (B, tiles over M, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for output positions and hidden-dim
    mask_m = m_offsets < (T + I)
    mask_n = n_offsets < K

    # For each output position m, decide source tensor: x1 if m < T, else x2.
    # We'll handle per-row via masks. We iterate over rows within the tile and store to out.
    # Compute base offsets
    # Out strides: out is [B, M, K] with strides (out_stride_b, out_stride_m, out_stride_k)
    # X1 strides: [B, T, K]
    # X2 strides: [B, I, K]

    # We need to write the tile to out. For each row i in m_offsets:
    # If i < T: read from x1[b, i, n_offsets]; else: read from x2[b, i - T, n_offsets].
    # This loop is over rows in the tile; Triton will vectorize across n_offsets.
    for i in range(BLOCK_M):
        m_i = m_offsets[i]
        valid_m = (m_i < (T + I)) & (i < BLOCK_M)
        # Determine source: encoder or image
        is_encoder = m_i < T
        # Compute per-source base offsets
        # x1 offset for this row: b * x1_stride_b + m_i * x1_stride_m + n_offsets * x1_stride_k
        # x2 offset: b * x2_stride_b + (m_i - T) * x2_stride_m + n_offsets * x2_stride_k
        # We need the out offset: b * out_stride_b + m_i * out_stride_m + n_offsets * out_stride_k
        # Note: We pass out strides as arguments.
        # Construct per-source address vectors; but we need to select based on is_encoder.
        # Triton doesn't support branching on runtime scalars cleanly; instead, we compute both possible offsets and select by masked load/store.
        # However, Triton can't branch on runtime masks inside elementwise expressions directly across vectors, so we compute using scalar logic for source and vector for n_offsets.
        # We'll compute offsets for x1 and x2 and then select based on is_encoder.

        # Compute base offsets for x1 and x2
        # x1: [B, T, K] -> offsets = b*x1_stride_b + m_i*x1_stride_m + n_offsets*x1_stride_k
        x1_base_b = b * x1_stride_b
        x1_row_offset = m_i * x1_stride_m + n_offsets[None, :] * x1_stride_k  # shape [1, BLOCK_N]
        x1_ptrs = x1_ptr + x1_base_b + x1_row_offset

        # x2: [B, I, K] -> offsets = b*x2_stride_b + (m_i - T)*x2_stride_m + n_offsets*x2_stride_k
        x2_row_index = m_i - T
        # x2_row_offset = x2_row_index * x2_stride_m + n_offsets * x2_stride_k
        x2_row_offset = x2_row_index * x2_stride_m + n_offsets[None, :] * x2_stride_k
        x2_ptrs = x2_ptr + b * x2_stride_b + x2_row_offset

        # Out: [B, M, K] -> offsets = b*out_stride_b + m_i*out_stride_m + n_offsets*out_stride_k
        out_row_offset = m_i * out_stride_m + n_offsets[None, :] * out_stride_k
        out_ptrs = out_ptr + b * out_stride_b + out_row_offset

        # Load from the appropriate source. We can't directly branch; use a masked load per source.
        # Compute load masks: for x1, mask_m & is_encoder; for x2, mask_m & (~is_encoder) & (m_i >= T).
        # But since m_i can be < T (encoder) or >= T (image), is_encoder = (m_i < T).
        # We'll compute the load value per source using a masked load and then store with mask_m and mask_n.
        val1 = tl.load(x1_ptrs, mask=mask_n & valid_m & (m_i < T), other=0.0)
        val2 = tl.load(x2_ptrs, mask=mask_n & valid_m & (m_i >= T), other=0.0)

        # Select value: if is_encoder -> val1 else val2
        # Triton supports tl.where. We can compute a boolean per row and select.
        # Note: Triton's boolean is computed elementwise, but here we have a scalar condition per row.
        # We can broadcast is_encoder (computed from m_i) to a vector mask for n_offsets and use tl.where.
        # is_encoder_scalar = (m_i < T)
        # broadcast mask: is_encoder_vec = (m_i < T) is scalar; use it to select.
        # Triton allows scalar conditions; create a vector mask by broadcasting.
        # Construct a vector mask for selection: where(is_encoder, val1, val2).
        # But since Triton operates elementwise, we need to apply selection across n_offsets.
        # Simpler: compute scalar selection and store both vals into out with the correct mask:
        # If is_encoder: store val1; else: store val2.
        # We do that by computing two stores and masking accordingly.
        # Create a selection scalar and broadcast to [BLOCK_N].
        is_encoder_scalar = m_i < T
        # Broadcast to [BLOCK_N] by making it a vector: is_encoder_vec = is_encoder_scalar * tl.ones([BLOCK_N], dtype=tl.int1)
        # However, Triton supports scalar broadcasting in tl.where; we can use is_encoder_scalar directly.
        # Triton doesn't support tl.full, but we can cast is_encoder_scalar to a vector via tl.where using scalar.
        # Instead, compute val_sel = where(is_encoder_scalar, val1, val2)
        val_sel = tl.where(is_encoder_scalar, val1, val2)

        # Store to out with mask for columns and row validity
        tl.store(out_ptrs, val_sel, mask=mask_m[i] & mask_n)


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    A_stride_b, A_stride_m, A_stride_k,
    W_stride_k0, W_stride_k1,  # W is [K, K] => strides (k0, k1)
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, tiles over M, tiles over N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + k_offsets[:, None] * W_stride_k0 + n_offsets[None, :] * W_stride_k1
        W_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w = tl.load(W_ptrs, mask=W_mask, other=0.0)

        # Accumulate: acc += a @ w
        acc += tl.dot(a, w)

    # Store result tile
    C_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Inputs:
        #   hidden_states: [B, I, K]
        #   encoder_hidden_states: [B, T, K]
        #   process_weight: [K, K]
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # Ensure dtypes are float32 for computation; we'll return in original dtype.
        # We'll compute in float32; if original is not float32, cast to float32 for kernels.
        # However, Triton kernels can operate on float16/bfloat16 as well; for safety, use float32.
        # Cast inputs to float32 for kernels
        encoder_hidden_states_f = encoder_hidden_states.contiguous().to(torch.float32)
        hidden_states_f = hidden_states.contiguous().to(torch.float32)
        process_weight_f = process_weight.contiguous().to(torch.float32)

        # 1) Concatenate along sequence dim using Triton
        A = torch.empty((B, M, K), device=encoder_hidden_states.device, dtype=torch.float32)

        BLOCK_M_C = 128
        BLOCK_N_C = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M_C), triton.cdiv(K, BLOCK_N_C))
        concat_seq_dim1_kernel[grid_concat](
            A, encoder_hidden_states_f, hidden_states_f,
            B, T, I, K,
            BLOCK_M=BLOCK_M_C, BLOCK_N=BLOCK_N_C,
            x1_stride_b=encoder_hidden_states_f.stride(0),
            x1_stride_m=encoder_hidden_states_f.stride(1),
            x1_stride_k=encoder_hidden_states_f.stride(2),
            x2_stride_b=hidden_states_f.stride(0),
            x2_stride_m=hidden_states_f.stride(1),
            x2_stride_k=hidden_states_f.stride(2),
            out_stride_b=A.stride(0),
            out_stride_m=A.stride(1),
            out_stride_k=A.stride(2),
            num_warps=4, num_stages=2
        )

        # 2) Batched GEMM: C = A @ process_weight_f
        C = torch.empty((B, M, K), device=encoder_hidden_states.device, dtype=torch.float32)

        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, process_weight_f,
            B, M, K, K,
            A.stride(0), A.stride(1), A.stride(2),
            process_weight_f.stride(0), process_weight_f.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2
        )

        # 3) Split back into encoder and hidden streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
