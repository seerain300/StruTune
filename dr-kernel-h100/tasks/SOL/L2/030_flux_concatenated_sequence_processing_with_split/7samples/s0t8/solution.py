import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    out_ptr,         # *fp32, output [B, M, K], M = T + I
    x1_ptr,          # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,          # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr, # batch size
    T: tl.constexpr, # text_seq_len
    I: tl.constexpr, # img_seq_len
    K: tl.constexpr, # hidden_dim
):
    # Grid: (B, tiles over M = T+I, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    # Tile sizes
    BLOCK_M = 128
    BLOCK_K = 64

    # Compute offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # along total sequence
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # along hidden_dim

    # Masks
    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # Decide source: encoder (x1) if m < T, else image (x2)
    is_encoder = m_offsets < T

    # For each k in tile, load from x1 or x2 and store into out
    for kk in range(0, BLOCK_K):
        k = k_offsets[kk]
        if not mask_k[kk]:
            break
        # Compute base offsets for the batch
        out_base = b * out_ptr  # element-wise pointer; Triton will increment with strides below
        x1_base = b * x1_ptr
        x2_base = b * x2_ptr

        # Per-row indices
        # If encoder: row = m; else: row = m - T
        row_encoder = tl.where(is_encoder, m_offsets, 0)
        row_image = m_offsets - T

        # Compute addresses
        out_addr = out_base + row_encoder * out_ptr.dtype.element_size + k * out_ptr.dtype.element_size
        # Note: Triton pointer arithmetic expects strides in elements, not bytes. Using .element_size is incorrect.
        # We should use stride values directly. Fix below.

        # Correct address computation using strides: assume row-major [B, M, K] with strides (M*K, K, 1)
        # Strides for out, x1, x2 are passed in elements; here we assume contiguous tensors so stride(0) = M*K, stride(1)=K, stride(2)=1.
        # But better: pass strides as parameters.

        # Fix: we need actual stride parameters for correctness. We will pass strides in the launch.
        # To keep code minimal, we use simple contiguous assumptions for demonstration. In Triton, we can pass strides as runtime args.
        # Since Triton doesn't expose .stride, we assume inputs are made contiguous in host code and use element indexing as above.
        # However, to be correct, we'll re-implement with explicit stride parameters.

        # We need to pass strides for x1, x2, out. Triton supports passing python int arrays as constexpr? No, we should use strides passed via kernel args.
        # Below, we correct by using stride parameters in the launch (see ModelNew.forward). For now, we implement simple indexing assuming contiguity.

        # Since Triton requires explicit strides, we'll re-implement the kernel below with stride parameters.


@triton.jit
def concat_seq_kernel_strided(
    out_ptr, x1_ptr, x2_ptr,
    B, T, I, K,
    out_s0, out_s1, out_s2,   # strides for out: stride over batch, seq, hidden
    x1_s0, x1_s1, x1_s2,      # strides for x1
    x2_s0, x2_s1, x2_s2,      # strides for x2
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # total seq positions
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden dim indices

    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K
    is_encoder = m_offsets < T

    # Loop over K tile
    for kk in range(0, BLOCK_K):
        k = k_offsets[kk]
        if not mask_k[kk]:
            break

        # Compute addresses using strides (in elements)
        # out: [B, M, K]
        # x1: [B, T, K]
        # x2: [B, I, K]
        # Note: Triton pointer arithmetic supports multiplying vector offsets by scalar strides.

        # Out addresses for encoder and image parts
        out_addr_encoder = out_ptr + b * out_s0 + m_offsets * out_s1 + k * out_s2
        out_addr_image = out_ptr + b * out_s0 + (m_offsets - T) * out_s1 + k * out_s2

        # Masks for stores
        store_mask_m = mask_m & is_encoder
        # For image part, only m >= T are valid, but we already masked m_offsets < (T+I) and is_encoder handles selection.

        # Load from x1 for encoder rows
        x1_addr = x1_ptr + b * x1_s0 + m_offsets * x1_s1 + k * x1_s2
        val_encoder = tl.load(x1_addr, mask=store_mask_m, other=0.0)

        # Load from x2 for image rows
        x2_addr = x2_ptr + b * x2_s0 + (m_offsets - T) * x2_s1 + k * x2_s2
        # is_encoder is boolean; for non-encoder rows, x2_addr is valid for m >= T. We need to select based on is_encoder.
        # Use masked load for x2: if is_encoder is False and m < T+I and m >= T, then load x2.
        is_image = ~is_encoder
        valid_image = mask_m & is_image
        val_image = tl.load(x2_addr, mask=valid_image, other=0.0)

        # Select per row: if encoder row -> val_encoder, else -> val_image
        # Build 2D selector with broadcasting
        sel = is_encoder[:, None]  # shape [BLOCK_M, 1]
        # val_image shape [BLOCK_M, BLOCK_K] due to broadcasting over k; we need per-row selection.
        # Better: compute val per m by selecting between val_encoder and val_image. Triton can't directly select; instead,
        # we can assign val per m based on is_encoder. We'll do that by writing val_encoder when is_encoder True; else 0, then add val_image where is_image True.
        # But to keep it simple, compute per-m row value by selecting scalar where applicable.

        # For correctness, we'll store val_encoder to out_addr_encoder where is_encoder True, and val_image to out_addr_image where is_image True.
        # However, Triton doesn't support direct per-element conditional store; we perform two masked stores.

        tl.store(out_addr_encoder, val_encoder, mask=store_mask_m)
        tl.store(out_addr_image, val_image, mask=valid_image)

@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B, M, N, K,
    C_s0, C_s1, C_s2,
    A_s0, A_s1, A_s2,
    W_s0, W_s1, W_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, tiles over M, tiles over N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * W_s1 + n_offsets[None, :] * W_s2
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(a, w)

    # Store result tile to C
    c_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the given PyTorch code:
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension using a Triton kernel.
        - Applies linear projection C = A @ process_weight via a Triton batched GEMM kernel.
        - Returns (processed_encoder_hidden_states, processed_hidden_states) split along the original sequence lengths.
        """
        # Ensure inputs are contiguous and float32 for Triton kernels
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == K, "hidden_dim mismatch"
        assert process_weight.shape[0] == K and process_weight.shape[1] == K, "process_weight must be [K, K]"

        x1 = encoder_hidden_states.contiguous()
        x2 = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate sequences along the sequence dimension into A: [B, M, K], M = T + I
        device = x1.device
        M = T + I
        A = torch.empty((B, M, K), device=device, dtype=torch.float32)

        # Strides (elements, not bytes)
        out_s0 = A.stride(0)  # M*K
        out_s1 = A.stride(1)  # K
        out_s2 = A.stride(2)  # 1
        x1_s0 = x1.stride(0)  # T*K
        x1_s1 = x1.stride(1)  # K
        x1_s2 = x1.stride(2)  # 1
        x2_s0 = x2.stride(0)  # I*K
        x2_s1 = x2.stride(1)  # K
        x2_s2 = x2.stride(2)  # 1

        BLOCK_M = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_kernel_strided[grid_concat](
            A, x1, x2,
            B, T, I, K,
            out_s0, out_s1, out_s2,
            x1_s0, x1_s1, x1_s2,
            x2_s0, x2_s1, x2_s2,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Batched GEMM: C = A @ W, output [B, M, K]
        C = torch.empty((B, M, K), device=device, dtype=torch.float32)
        C_s0 = C.stride(0)  # M*K
        C_s1 = C.stride(1)  # K
        C_s2 = C.stride(2)  # 1
        A_s0 = A.stride(0)  # M*K
        A_s1 = A.stride(1)  # K
        A_s2 = A.stride(2)  # 1
        W_s0 = W.stride(0)  # K*K
        W_s1 = W.stride(1)  # K
        W_s2 = W.stride(2)  # 1

        BLOCK_M_G = 128
        BLOCK_N_G = 128
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K, K,
            C_s0, C_s1, C_s2,
            A_s0, A_s1, A_s2,
            W_s0, W_s1, W_s2,
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into separate streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtype if needed
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
