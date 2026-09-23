import torch
import triton
import triton.language as tl

# Triton kernel: concatenate encoder_hidden_states [B, T, K] and hidden_states [B, P, K] into Acat [B, T+P, K]
@triton.jit
def _concatenation_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, P, K,
    encoder_stride_b, encoder_stride_t, encoder_stride_k,
    hidden_stride_b, hidden_stride_p, hidden_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)  # l in [0, T+P)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Decide source: if l < T, read from encoder; else read from hidden at l-T
    is_encoder = l < T

    # Compute base pointers
    enc_base = encoder_ptr + b * encoder_stride_b
    dec_base = hidden_ptr + b * hidden_stride_b  # for decoder (hidden), we'll read at l-T
    out_base = out_ptr + b * out_stride_b + l * out_stride_l

    # Pointers for load
    # If is_encoder: load enc_base + l * encoder_stride_t
    # else: load dec_base + (l - T) * hidden_stride_p
    # Use masks on k_offsets
    if is_encoder:
        ptrs = enc_base + l * encoder_stride_t + k_offsets * encoder_stride_k
    else:
        l_prime = l - T
        ptrs = dec_base + l_prime * hidden_stride_p + k_offsets * hidden_stride_k

    vals = tl.load(ptrs, mask=mask_k, other=0.0)
    tl.store(out_base + k_offsets * out_stride_k, vals, mask=mask_k)


# Triton kernel: split C [B, T+P, K] into processed_encoder [B, T, K] and processed_hidden [B, P, K]
@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, K,
    C_stride_b, C_stride_t, C_stride_k,
    out_stride_b, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    C_base = C_ptr + b * C_stride_b + t * C_stride_t
    out_base = out_ptr + b * out_stride_b

    vals = tl.load(C_base + k_offsets * C_stride_k, mask=mask_k, other=0.0)
    tl.store(out_base + k_offsets * out_stride_k, vals, mask=mask_k)


@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, P, K,
    C_stride_b, C_stride_p, C_stride_k,
    out_stride_b, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    p = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # We need to read from C at position t + p, i.e., row index T + p
    # But since C is [B, T+P, K], we can directly index p in C.
    C_base = C_ptr + b * C_stride_b + p * C_stride_p
    out_base = out_ptr + b * out_stride_b

    vals = tl.load(C_base + k_offsets * C_stride_k, mask=mask_k, other=0.0)
    tl.store(out_base + k_offsets * out_stride_k, vals, mask=mask_k)


# Triton kernel: perform GEMM A [M, K] @ W [K, K] -> C [M, K], where A is concatenated [B*(T+P), K], W is process_weight.T
# We use a 3D grid: (B, cdiv(M, BLOCK_M), cdiv(K, BLOCK_N)), but since A is flat, we pass M and N and let grid be (1, tiles_m, tiles_n).
# Alternatively, we can let B be 1 and use M,B,N as 1D/2D by flattening. Here we pass B as 1 and treat A as [M, K].
@triton.jit
def _gemm_matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    M, N, K,  # M = B*(T+P), N = K, K = hidden dim
    A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,  # W is [K, N], we pass strides accordingly
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # We expect grid as (1, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    # Triton does not support dynamic grid dimension for B, so we pass B=1 and rely on host to compute M,N,K.
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_vals = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W tile: W[k, n]
        W_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        W_vals = tl.load(W_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(A_vals, W_vals)

    # Store results
    C_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        All computation happens in Triton kernels:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        - Compute GEMM: concatenated @ process_weight.T
        - Split the result back into encoder and hidden outputs.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [B, dim, K]"
        assert process_weight.shape[1] == hidden_states.shape[2], "process_weight's second dim must equal hidden_states' last dim"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure inputs are contiguous and dtype float32 for Triton
        encoder_hidden = encoder_hidden_states.contiguous().to(torch.float32)
        hidden = hidden_states.contiguous().to(torch.float32)
        process_weight_t = process_weight.transpose(0, 1).contiguous().to(torch.float32)  # shape [K, K]

        # 1) Concatenate encoder_hidden and hidden into Acat [B, T+P, K] using Triton kernel
        total_L = T + P
        Acat = torch.empty((B, total_L, K), device=device, dtype=torch.float32)

        BLOCK_K = 64
        grid_concat = (B, total_L, triton.cdiv(K, BLOCK_K))
        _concatenation_kernel[grid_concat](
            encoder_hidden, hidden, Acat,
            B, T, P, K,
            encoder_hidden.stride(0), encoder_hidden.stride(1), encoder_hidden.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_K=BLOCK_K, num_warps=4, num_stages=2
        )

        # 2) Compute GEMM: Acat [B*(T+P), K] @ process_weight_t [K, K] -> C_flat [B*(T+P), K]
        M = B * total_L
        N = K  # output dimension equals hidden dim
        C_flat = torch.empty((M, N), device=device, dtype=torch.float32)

        # Flatten Acat to [M, K]
        A_flat = Acat.reshape(M, N).contiguous()

        # Launch GEMM kernel with a 3D grid over m_tiles and n_tiles. We set B=1 conceptually by flattening.
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_gemm = (1, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _gemm_matmul_kernel[grid_gemm](
            A_flat, process_weight_t, C_flat,
            M, N, K,
            A_flat.stride(0), A_flat.stride(1),
            process_weight_t.stride(0), process_weight_t.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape back to [B, T+P, K]
        C = C_flat.reshape(B, total_L, K)

        # 3) Split into encoder and hidden outputs using Triton kernels
        processed_encoder = torch.empty((B, T, K), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=torch.float32)

        BLOCK_K_split = 128
        grid_encoder = (B, T, triton.cdiv(K, BLOCK_K_split))
        _split_encoder_kernel[grid_encoder](
            C, processed_encoder,
            B, T, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K_split, num_warps=4, num_stages=2
        )

        grid_hidden = (B, P, triton.cdiv(K, BLOCK_K_split))
        _split_hidden_kernel[grid_hidden](
            C, processed_hidden,
            B, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K_split, num_warps=4, num_stages=2
        )

        # Return results (keep float32 as original code uses default float32)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
