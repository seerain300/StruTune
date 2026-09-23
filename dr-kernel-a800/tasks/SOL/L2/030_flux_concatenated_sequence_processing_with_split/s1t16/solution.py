import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seqs_kernel(
    encoder_ptr,  # *f32, shape [B, T, K]
    hidden_ptr,   # *f32, shape [B, P, K]
    out_ptr,      # *f32, shape [B, T+P, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, cdiv(T+P, BLOCK_L), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    l_block = tl.program_id(1)
    k_block = tl.program_id(2)

    # Compute indices
    l = l_block * BLOCK_L + tl.arange(0, BLOCK_L)             # [BLOCK_L]
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)             # [BLOCK_K]

    total = T + P
    mask_l = l < total
    mask_k = k < K
    mask = mask_l[:, None] & mask_k[None, :]                  # [BLOCK_L, BLOCK_K]

    # Decide source: if l < T -> encoder[b, l, :], else hidden[b, l-T, :]
    is_encoder = l < T                                       # [BLOCK_L]

    # Build pointers
    # For encoder: ptr = encoder_ptr + b*stride_e_b + l*stride_e_t + k*stride_e_k
    # For hidden: ptr = hidden_ptr + b*stride_h_b + (l - T)*stride_h_p + k*stride_h_k
    # We assume contiguous layout [B, T, K] and [B, P, K] for inputs; out is [B, T+P, K], contiguous.
    # Load from source
    # When is_encoder is True, load from encoder; else load from hidden
    # We'll do masked loads for both possibilities, but only one will be true per element.

    # Compute base offsets
    # For encoder: offset_e = b*stride_e_b + l*stride_e_t + k*stride_e_k
    # For hidden: offset_h = b*stride_h_b + (l - T)*stride_h_p + k*stride_h_k
    # Note: we can't branch on Triton boolean tensors, so we load both with mask applied.

    # We need strides. Inputs are expected contiguous. For [B, T, K], strides are (T*K, K, 1).
    # For [B, P, K], strides are (P*K, K, 1).
    # For out [B, T+P, K], strides are ((T+P)*K, K, 1).

    # Use factored ptr arithmetic:
    # enc_ptr: b*T*K + l*K + k
    # h_ptr: b*P*K + (l - T)*K + k
    # out_ptr: b*(T+P)*K + l*K + k

    # We compute per element address with broadcasting:
    # Create 2D grids with broadcasting
    # But Triton allows elementwise arithmetic:
    # Build 2D offsets for encoder and hidden separately with masks.

    # Since we cannot branch on vector masks, we will compute both pointers and use masks to load from correct source.
    # Create 2D grid for l and k
    l2d = l[:, None]  # [BLOCK_L, 1]
    k2d = k[None, :]  # [1, BLOCK_K]

    # Encoder contributions
    # Pointer for encoder: base = b*T*K + l2d*K + k2d
    # Note: K is scalar, broadcasting happens.
    enc_ptr = encoder_ptr + b * T * K + l2d * K + k2d
    # Hidden contributions (only for l >= T)
    # Pointer for hidden: base = b*P*K + (l2d - T)*K + k2d
    # When l < T, l2d - T < 0, masked loads will ignore
    h_ptr = hidden_ptr + b * P * K + (l2d - T) * K + k2d

    # Load with masks:
    # If is_encoder True: load from enc_ptr, else from h_ptr
    # Triton doesn't support conditional tensor loads here, so we do masked loads per source.
    # We can compute a combined mask for each source: mask & is_encoder, and mask & ~is_encoder
    # Then combine via tl.where? Triton needs two loads. Perform masked loads into a temp tensor.

    # For masked loads, we need a pointer tensor and a boolean mask. Triton supports tl.load(ptr, mask=..., other=...).
    # Let's do two loads: one for encoder, one for hidden; then select via tl.where. But Triton doesn't support per-element tl.where on pointers.
    # Instead, we perform two tl.load calls and then select based on is_encoder. Triton supports elementwise selection in tl.store, but here we select before storing into out.

    # We'll build a 2D output pointer for out: out_ptr + b*(T+P)*K + l2d*K + k2d
    out_base = out_ptr + b * (T + P) * K + l2d * K + k2d

    # Prepare two load pointers with per-element masks:
    # Load from encoder where is_encoder is True
    enc_mask = mask & (is_encoder[:, None])
    # Load from hidden where is_encoder is False
    h_mask = mask & (~is_encoder[:, None])

    # Now perform loads with mask applied; use other=0.0 for masked elements
    # Triton tl.load expects a pointer tensor and mask tensor.
    # But we need the mask to be 2D. is_encoder is 1D; broadcasting is supported in Triton masks.
    enc_vals = tl.load(enc_ptr, mask=enc_mask, other=0.0)
    h_vals = tl.load(h_ptr, mask=h_mask, other=0.0)

    # Select: where is_encoder True -> enc_vals else h_vals
    # Triton supports tl.where on tensors.
    out_vals = tl.where(is_encoder[:, None], enc_vals, h_vals)

    # Store to output with mask
    tl.store(out_base, out_vals, mask=mask)


@triton.jit
def _matmul_kernel(
    A_ptr,  # *f32, shape [M, K], where M = B*(T+P)
    Wt_ptr, # *f32, shape [K, K] (process_weight.T)
    C_ptr,  # *f32, shape [M, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,  # for shape info only; M = B*(T+P)
    M_total: tl.constexpr,  # M = B*(T+P)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 3D grid: (B, cdiv(M_total, BLOCK_M), cdiv(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = m < M_total
    mask_n = n < K
    mask = mask_m[:, None] & mask_n[None, :]

    # Map m to (b, l) where l in [0, T+P)
    total = T + P
    l = m % total  # [BLOCK_M]
    bm = b
    # Now compute A indices and C indices: A row index is m, C row index is m as well.

    # K loop
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k < K
        # Load A[m, k] -> A is [M_total, K], flattened by b in host; here we assume A_ptr is contiguous over M_total*K.
        # For A row m, base = m*K + k
        A_row_ptr = A_ptr + m[:, None] * K + k[None, :]
        A_vals = tl.load(A_row_ptr, mask=mask[:, None] & mask_k[None, :], other=0.0)

        # Load Wt[k, n] -> W_t is [K, K]
        Wt_ptr = Wt_ptr + k[:, None] * K + n[None, :]
        Wt_vals = tl.load(Wt_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(A_vals, Wt_vals)

    # Store C[m, n] -> C is [M_total, K]
    C_row_ptr = C_ptr + m[:, None] * K + n[None, :]
    tl.store(C_row_ptr, acc, mask=mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr,  # *f32, shape [B, T+P, K]
    Out_ptr,  # *f32, shape [B, T, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, cdiv(T, BLOCK_T), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    t_block = tl.program_id(1)
    k_block = tl.program_id(2)

    t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    mask_t = t < T
    mask_k = k < K
    mask = mask_t[:, None] & mask_k[None, :]

    # C base: [B, T+P, K] -> row index m = b*(T+P) + t, col k
    total = T + P
    m = b * total + t
    C_row_ptr = C_ptr + m[:, None] * K + k[None, :]

    # Out is [B, T, K] -> same row index m and col k
    Out_row_ptr = Out_ptr + m[:, None] * K + k[None, :]

    vals = tl.load(C_row_ptr, mask=mask, other=0.0)
    tl.store(Out_row_ptr, vals, mask=mask)


@triton.jit
def _split_hidden_kernel(
    C_ptr,  # *f32, shape [B, T+P, K]
    Out_ptr,  # *f32, shape [B, P, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, cdiv(P, BLOCK_P), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    p_block = tl.program_id(1)
    k_block = tl.program_id(2)

    p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)  # [BLOCK_P]
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    mask_p = p < P
    mask_k = k < K
    mask = mask_p[:, None] & mask_k[None, :]

    total = T + P
    m = b * total + (p + T)  # hidden starts at index T in concatenated
    C_row_ptr = C_ptr + m[:, None] * K + k[None, :]

    Out_row_ptr = Out_ptr + (b * P + p)[:, None] * K + k[None, :]

    vals = tl.load(C_row_ptr, mask=mask, other=0.0)
    tl.store(Out_row_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim in Triton.
        - Perform GEMM in Triton: concatenated @ process_weight.T
        - Split results back into encoder and hidden streams in Triton.
        Returns: (processed_encoder, processed_hidden)
        Shapes:
          - hidden_states: [B, P, K]
          - encoder_hidden_states: [B, T, K]
          - process_weight: [K, K]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device."
        B, P, K = hidden_states.shape
        assert hidden_states.shape[2] == K
        T, _, _ = encoder_hidden_states.shape
        assert encoder_hidden_states.shape[2] == K
        assert process_weight.shape[0] == K and process_weight.shape[1] == K

        # Ensure float32 for Triton kernels
        # Cast if needed
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        # 1) Triton concatenate into Acat [B, T+P, K]
        total = T + P
        Acat = torch.empty((B, total, K), device=hidden_states.device, dtype=torch.float32)

        # Launch concatenation kernel
        # Choose blocks
        BLOCK_L = 64
        BLOCK_K = 128
        grid_concat = (B, triton.cdiv(total, BLOCK_L), triton.cdiv(K, BLOCK_K))
        _concatenate_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Triton GEMM: C_flat = Acat @ process_weight.T, Acat is [M, K], W_t is [K, K]
        M_total = B * total
        C_flat = torch.empty((M_total, K), device=hidden_states.device, dtype=torch.float32)
        Wt = process_weight  # [K, K], already float32

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_matmul = (B, triton.cdiv(M_total, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_matmul](
            Acat.reshape(M_total, K), Wt, C_flat,
            B, T, P, K, M_total,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Triton split C_flat back into [B, T, K] and [B, P, K]
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

        BLOCK_T = 64
        BLOCK_K_split = 128
        grid_e = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(K, BLOCK_K_split))
        _split_encoder_kernel[grid_e](
            C_flat, processed_encoder,
            B, T, P, K,
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        BLOCK_P = 64
        grid_h = (B, triton.cdiv(P, BLOCK_P), triton.cdiv(K, BLOCK_K_split))
        _split_hidden_kernel[grid_h](
            C_flat, processed_hidden,
            B, T, P, K,
            BLOCK_P=BLOCK_P, BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
