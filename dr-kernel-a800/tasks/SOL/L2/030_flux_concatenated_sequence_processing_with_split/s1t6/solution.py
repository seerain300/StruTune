import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_kernel(
    encoder_ptr, hidden_ptr, A_ptr,
    B, T, P, K,
    eb, et, ek,  # strides for encoder
    hb, hp, hk,  # strides for hidden
    Ab, Al, Ak,  # strides for Acat
    BLOCK_T: tl.constexpr, BLOCK_P: tl.constexpr
):
    # One program per (batch b, tile of sequence l)
    b = tl.program_id(0)
    tile = tl.program_id(1)
    l_offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    p_offsets = tile * BLOCK_P + tl.arange(0, BLOCK_P)

    # masks
    mask_l = l_offsets < T
    mask_p = p_offsets < P

    # Base pointers for batch
    encoder_row_ptr = encoder_ptr + b * eb
    hidden_row_ptr = hidden_ptr + b * hb
    A_batch_ptr = A_ptr + b * Ab

    # For each l, write from encoder if l < T, else from hidden with offset l - T
    # We iterate l in chunks of BLOCK_T and p in chunks of BLOCK_P using nested loops to cover both regions
    # This is done via a single program over a 2D tile. We use broadcasting to handle both l and p in one vector.
    # To keep it simple and correct for arbitrary shapes, we run two vectorized loops:
    # 1) l in [0, T) if any, 2) p in [0, P) if any, otherwise we mask out.
    # Triton doesn't support 'for' over runtime ranges directly; we instead compute per vector of offsets and mask.
    # However, Triton supports scalar loop control via ranges if we define vector offsets. The safe approach is:
    # Use two separate tiled writes: one for encoder part (l < T), one for hidden part (l >= T).
    # Since Triton restricts loops, we implement two separate kernel instances or handle via separate programs.
    # Here, we design the kernel to handle both by iterating over l_offsets and p_offsets with masks.
    # If l >= T, we map l -> l - T and read from hidden, otherwise read from encoder.

    # We implement the writes in two parts: l < T, and l >= T (i.e., l = p_offsets + T).
    # Because Triton doesn't allow 'if' on vectors, we use masks to guard loads/stores.

    # Write encoder part: l in [0, T)
    l_idx = l_offsets
    mask_encoder = (l_idx < T)
    # For each l, k in [0, K)
    k = 0
    while k < K:
        enc_vals = tl.load(encoder_row_ptr + l_idx * et + k * ek, mask=mask_encoder, other=0.0)
        A_ptrs = A_batch_ptr + (l_idx + T) * Al + k * Ak  # because when we write, index is l + T
        tl.store(A_ptrs, enc_vals, mask=mask_encoder)
        k += 1

    # Write hidden part: l in [T, T+P) which maps to source l' = l - T, i.e., l = p_offsets + T
    p_idx = p_offsets
    mask_hidden = (p_idx + T < T + P) & (p_idx < P)
    # For each l = p_idx + T, k in [0, K)
    k = 0
    while k < K:
        src_l = p_idx + T
        hid_vals = tl.load(hidden_row_ptr + src_l * hp + k * hk, mask=mask_hidden, other=0.0)
        A_ptrs = A_batch_ptr + src_l * Al + k * Ak
        tl.store(A_ptrs, hid_vals, mask=mask_hidden)
        k += 1


@triton.jit
def _matmul_kernel_3d(
    A_ptr, W_ptr, C_ptr,
    B, M, N, K,
    Ab, Am, Ak,  # A strides: batch, row, col
    Wk, Wk2,      # W strides (W is [K, K]): row=k, col=k
    Cb, Cm, Ck,   # C strides: batch, row, col
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 3D grid: (b, m_tiles, n_tiles)
    b = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)

    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # A_tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + b * Ab + m_offsets[:, None] * Am + k_offsets[None, :] * Ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # W_tile: [BLOCK_K, BLOCK_N], W is [K, K]
        W_ptrs = W_ptr + k_offsets[:, None] * Wk + n_offsets[None, :] * Wk2
        W_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        W_tile = tl.load(W_ptrs, mask=W_mask, other=0.0)

        acc += tl.dot(A_tile, W_tile)

    # Store results to C: C[b, m, n] = acc[m,n]
    C_ptrs = C_ptr + b * Cb + m_offsets[:, None] * Cm + n_offsets[None, :] * Ck
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, K,
    Cb, Cl, Ck,
    out_b, out_k, out_n,
    BLOCK_B: tl.constexpr, BLOCK_T: tl.constexpr
):
    b = tl.program_id(0)
    t_tile = tl.program_id(1)
    t_offsets = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    # Accumulate across batch dimension (B) if needed. Here we just copy for given b.
    # We implement per-batch copy; if b varies, we can loop, but simpler to launch grid per batch.
    # Load C[b, t, k], store to out[b, t, k]
    # We need k loop; to keep simple, assume K small enough to vectorize, else use nested loops.
    # Implement k in vectorized form: k = 0..K-1, using while; but Triton prefers vectorized loads.
    # We'll iterate k in a while loop for simplicity.

    # We will assume K is small and vectorizable. For generality, we implement nested loops.
    # This kernel is per-batch; no need to iterate over B. Launch grid as (B, cdiv(T, BLOCK_T)).
    # For simplicity, handle only b==program_id(0) case. But Triton allows only scalar program_id(0).
    # So we keep it as per-batch program: we ignore t_offsets and just copy slices.

    # We will load a [BLOCK_T, 1] slice and store. But Triton expects 2D tiles; better to loop k.

    # To avoid complexity, we implement per-batch copy with 1D t_offsets and loop k.
    # This ensures correctness for arbitrary K.
    # Note: Triton supports loops. We will write a simple per-batch copy using tl.load/tl.store with masks.

    # Not ideal but correct: Implement per-batch copy via nested loops. This is not vectorized across t but simple.

    # We can instead write a kernel that copies per-batch, per-t using masks. Simpler approach: we launch grid (B, T).
    # However, Triton requires constexpr tiling. So we use (B, ceil_div(T, BLOCK_T)) and loop inside.

    # Instead of nested loops, use a single store for each t: Triton allows loops. We'll loop over k.

    # We will implement the copy using k vectorized where possible. Let's vectorize over k up to 64 and loop otherwise.
    # Better: implement a 1D per-batch copy kernel using while loop over k. But Triton expects some vectorization.

    # Since this is a small part, we keep it simple and correct: implement per-batch split with masked loads/stores.
    # The caller will launch grid=(B, ceil_div(T, BLOCK_T)) and each program handles a tile of T for a given batch.

    # We'll implement the per-batch copy: for each t in tile, copy C[b, t, :] -> out[b, t, :].
    # This is acceptable for correctness first.

    # We need to loop over K to load/store; Triton allows while loops.
    t = 0
    while t < T:
        # For each k in K
        k = 0
        while k < K:
            C_vals = tl.load(C_ptr + b * Cb + t * Cl + k * Ck)
            tl.store(out_ptr + b * out_b + t * out_k + k * out_n, C_vals)
            k += 1
        t += 1


@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    Cb, Cl, Ck,
    out_b, out_p, out_k,
    BLOCK_B: tl.constexpr, BLOCK_P: tl.constexpr
):
    # Similar to split encoder, per-batch copy for hidden part starting at t=P
    b = tl.program_id(0)
    p_tile = tl.program_id(1)
    p_offsets = p_tile * BLOCK_P + tl.arange(0, BLOCK_P)

    # For each p in tile, copy C[b, t+P, k] -> out[b, p, k]
    # Implement nested loops over K (K is runtime, Triton supports while loops)
    p = 0
    while p < P:
        k = 0
        while k < K:
            C_vals = tl.load(C_ptr + b * Cb + (p + T) * Cl + k * Ck)
            tl.store(out_ptr + b * out_b + p * out_p + k * out_k, C_vals)
            k += 1
        p += 1


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          - Concatenate (encoder, image) via Triton kernel
          - Linear projection via Triton matmul kernel
          - Split into encoder and image outputs via Triton kernels
        Returns:
          processed_encoder: [B, T, K]
          processed_hidden: [B, P, K]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be CUDA for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        device = hidden_states.device

        # 1) Concatenate in Triton: Acat [B, T+P, K]
        Acat = torch.empty((B, T + P, K), device=device, dtype=torch.float32)

        # Ensure inputs are contiguous for simple stride usage
        encoder_hidden = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()

        # Launch concatenation kernel with 3D grid: (B, ceil_div(T, BLOCK_T), ceil_div(P, BLOCK_P))
        # Reasonable blocks for these dims
        BLOCK_T = 128
        BLOCK_P = 128
        grid_concat = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(P, BLOCK_P))
        _concatenate_kernel[grid_concat](
            encoder_hidden, hidden, Acat,
            B, T, P, K,
            encoder_hidden.stride(0), encoder_hidden.stride(1), encoder_hidden.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_P=BLOCK_P,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: Acat @ process_weight.T, process_weight is [K, K]
        # Flatten Acat to [M, K], W_t = process_weight.T (contiguous [K, K])
        M = B * (T + P)
        A = Acat.reshape(M, K).contiguous()
        W_t = process_weight.transpose(0, 1).contiguous()  # [K, K]

        # Output C_flat [M, K]
        C_flat = torch.empty((M, K), device=device, dtype=torch.float32)

        # Launch Triton matmul with proper 3D grid: (B, ceil_div(M, BLOCK_M), ceil_div(K, BLOCK_N))
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_matmul = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel_3d[grid_matmul](
            A, W_t, C_flat,
            B, M, K, K,
            A.stride(0), A.stride(0), A.stride(1),  # A strides: row stride (unused), col stride
            0, 1,                                    # W strides: row=k, col=k
            C_flat.stride(0), C_flat.stride(0), C_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape to [B, T+P, K]
        C = C_flat.view(B, T + P, K)

        # 3) Split into encoder and hidden parts via Triton
        processed_encoder = torch.empty((B, T, K), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=torch.float32)

        # Launch split kernels. Grid is (B, ceil_div(T, BLOCK_T)) for encoder, and (B, ceil_div(P, BLOCK_P)) for hidden.
        # Use modest blocks
        BLOCK_T_split = 128
        BLOCK_P_split = 128

        grid_e = (B, triton.cdiv(T, BLOCK_T_split))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2), processed_encoder.stride(1),
            BLOCK_B=1, BLOCK_T=BLOCK_T_split,  # per-batch program; B is grid dim 0
            num_warps=4, num_stages=2
        )

        grid_h = (B, triton.cdiv(P, BLOCK_P_split))
        _split_hidden_kernel[grid_h](
            C, processed_hidden,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_B=1, BLOCK_P=BLOCK_P_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
