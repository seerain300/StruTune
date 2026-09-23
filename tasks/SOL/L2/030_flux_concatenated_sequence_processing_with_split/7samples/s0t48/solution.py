import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,        # *fp32, output A: [B, M, K], M = T + I
    x1_ptr,         # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,         # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr, # batch size
    T: tl.constexpr, # text_seq_len
    I: tl.constexpr, # img_seq_len
    K: tl.constexpr, # hidden dim
    stride_ob, stride_om, stride_ok,   # strides for out
    stride_x1b, stride_x1t, stride_x1k, # strides for x1
    stride_x2b, stride_x2i, stride_x2k, # strides for x2
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Tile indices
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Masks for bounds
    m_mask = m_offsets < (T + I)
    k_mask = k_offsets < K

    # Determine source: x1 for m < T, x2 for m >= T
    use_x1 = m_offsets < T

    # Compute pointers for loads
    # out[b, m, k] contiguous addressing
    out_ptrs = out_ptr + pid_b * stride_ob + m_offsets[:, None] * stride_om + k_offsets[None, :] * stride_ok
    # x1[b, t, k]
    x1_ptrs = x1_ptr + pid_b * stride_x1b + m_offsets[:, None] * stride_x1t + k_offsets[None, :] * stride_x1k
    # x2[b, i, k]
    x2_ptrs = x2_ptr + pid_b * stride_x2b + (m_offsets[:, None] - T) * stride_x2i + k_offsets[None, :] * stride_x2k

    # Masked load from x1 where use_x1, else from x2 where use_x1 is False
    # Create 2D mask for x1: m_mask & use_x1 & k_mask; for x2: m_mask & ~use_x1 & k_mask
    mask_x1 = m_mask[:, None] & use_x1[:, None] & k_mask[None, :]
    mask_x2 = m_mask[:, None] & (~use_x1)[:, None] & k_mask[None, :]

    # Load with other=0.0 to avoid OOB and ensure masked elements are zero
    x1_vals = tl.load(x1_ptrs, mask=mask_x1, other=0.0)
    x2_vals = tl.load(x2_ptrs, mask=mask_x2, other=0.0)
    # Select source based on use_x1
    sel = use_x1[:, None]
    # For m >= T, use_x1 is False; for m < T, use_x1 is True.
    # We need a 2D tensor: where sel, take x1, else x2.
    # Triton supports tl.where on tensors.
    # But tl.where expects same shape. We can rely on the fact that use_x1 is 1D and broadcast:
    # Here we construct a valid 2D selection tensor by using sel as a mask and feeding x1/x2 accordingly.
    # Since masked loads set non-selected regions to 0, we can just sum:
    A_tile = x1_vals + tl.zeros_like(x1_vals)  # temporary
    # Instead, we perform the select properly:
    A_tile = tl.where(sel[:, None], x1_vals, tl.zeros_like(x1_vals))
    # For positions where ~sel, x2_vals is already 0 for non-matching m; but we need to inject x2_vals where sel is False.
    # Easiest: compute A_tile as x1 for sel True; else 0; then add x2 where sel is False.
    # However, Triton doesn't allow dynamic selection of pointer results across two masked loads directly.
    # So we compute A_tile by using the fact that masked loads gave zeros for non-selected regions, and we need to force x2 for m >= T.
    # To ensure correctness, we set A_tile = x1_vals where sel is True, else 0; then add x2_vals where sel is False by broadcasting.
    # But since x2_vals is zero where sel is True, we can simply choose the correct one:
    # Construct A_tile as zeros and fill where sel is True with x1_vals, else with x2_vals.
    # Triton allows per-element selection via tl.where with scalars, but not mixing 1D and 2D masks in a single tl.where across two pointers.
    # Therefore, we rely on the masked loads to set non-selected regions to 0 and then select via tl.where using a computed value:
    # We can do:
    # A_tile = tl.where(sel, x1_vals, x2_vals)
    # Note: sel is 1D. Triton requires same shape for tl.where. Use broadcasting: sel[:, None].
    sel2D = sel[:, None]
    A_tile = tl.where(sel2D, x1_vals, x2_vals)

    # Store to out
    store_ptrs = out_ptr + pid_b * stride_ob + m_offsets[:, None] * stride_om + k_offsets[None, :] * stride_ok
    store_mask = m_mask[:, None] & k_mask[None, :]
    tl.store(store_ptrs, A_tile, mask=store_mask)


@triton.jit
def batched_matmul_kernel(
    C_ptr,           # *fp32, output [B, M, N]
    A_ptr,           # *fp32, input [B, M, K]
    W_ptr,           # *fp32, input [K, N]
    B: tl.constexpr, # batch size
    M: tl.constexpr, # total sequence length (T + I)
    K: tl.constexpr, # hidden dim
    N: tl.constexpr, # hidden dim (same as K)
    stride_ab, stride_am, stride_ak,      # strides for A
    stride_wk, stride_wn,                 # strides for W
    stride_cb, stride_cm, stride_cn,      # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid over (B, tiles along M, tiles along N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + pid_b * stride_ab + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        W_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        W_tile = tl.load(W_ptrs, mask=W_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store results
    C_ptrs = C_ptr + pid_b * stride_cb + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension using Triton.
        - Applies linear projection using a Triton batched matmul kernel.
        - Splits outputs back into separate encoder and hidden streams.
        Returns: (processed_encoder_hidden_states, processed_hidden_states)
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # Ensure inputs are contiguous and in fp32 for Triton
        x1 = encoder_hidden_states.contiguous().to(torch.float32)  # [B, T, K]
        x2 = hidden_states.contiguous().to(torch.float32)          # [B, I, K]

        # Allocate output for concatenated A: [B, M, K], fp32
        A = torch.empty((B, M, K), dtype=torch.float32, device=x1.device)

        # Launch concat kernel
        BLOCK_M = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_dim1_kernel[grid_concat](
            A, x1, x2,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x2.stride(0), x2.stride(1), x2.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Prepare weight W = process_weight.T in fp32: [K, K]
        W = process_weight.contiguous().to(torch.float32).t().contiguous()  # [K, K]

        # Allocate output C: [B, M, K], fp32
        C = torch.empty((B, M, K), dtype=torch.float32, device=A.device)

        # Launch batched matmul kernel
        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K, K,  # N = K
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # Split results back along the sequence dimension
        processed_encoder = C[:, :T, :]                     # [B, T, K]
        processed_hidden = C[:, T:, :]                     # [B, I, K]

        # Cast back to original input dtypes to match the original API
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
