import torch
import triton
import triton.language as tl

# Triton kernel: concatenate encoder_hidden and hidden along sequence dim into Acat [B, T+P, K]
# Inputs:
#   encoder: [B, T, K], row-major
#   hidden: [B, P, K]
#   Acat: [B, T+P, K] (output)
@triton.jit
def _concatenate_kernel(encoder_ptr, hidden_ptr, Acat_ptr,
                         B, T, P, K,
                         encoder_stride_b, encoder_stride_t, encoder_stride_k,
                         hidden_stride_b, hidden_stride_p, hidden_stride_k,
                         Acat_stride_b, Acat_stride_s, Acat_stride_k,
                         BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)  # s in [0, T+P)
    k_offsets = tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    if s < T:
        enc_row_ptr = encoder_ptr + b * encoder_stride_b + s * encoder_stride_t + k_offsets * encoder_stride_k
        vals = tl.load(enc_row_ptr, mask=mask_k, other=0.0)
        out_ptr = Acat_ptr + b * Acat_stride_b + s * Acat_stride_s + k_offsets * Acat_stride_k
        tl.store(out_ptr, vals, mask=mask_k)
    else:
        hid_row_ptr = hidden_ptr + b * hidden_stride_b + (s - T) * hidden_stride_p + k_offsets * hidden_stride_k
        vals = tl.load(hid_row_ptr, mask=mask_k, other=0.0)
        out_ptr = Acat_ptr + b * Acat_stride_b + s * Acat_stride_s + k_offsets * Acat_stride_k
        tl.store(out_ptr, vals, mask=mask_k)


# Triton block matmul kernel: computes C_tile = A_tile @ W_tile
# Inputs:
#   A: [M, K], M = B * (T+P), row-major
#   W: [K, K]
#   C: [M, K] (output)
# Grid: (B, tiles_M, tiles_N)
@triton.jit
def _matmul_block_kernel(A_ptr, W_ptr, C_ptr,
                         B, M, N, K,
                         A_stride_m, A_stride_k,
                         W_stride_k_row, W_stride_k_col,
                         C_stride_m, C_stride_n,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in M
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in N (here N=K)

    # Accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W tile as [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * W_stride_k_row + n_offsets[None, :] * W_stride_k_col
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a_vals, w_vals)

    # Store C tile
    c_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Triton kernel: copy C[b, s, :] -> processed_encoder[b, s, :] for s in [0, T)
@triton.jit
def _split_encoder_kernel(C_ptr, processed_ptr,
                          B, T, K,
                          C_stride_b, C_stride_s, C_stride_k,
                          processed_stride_b, processed_stride_s, processed_stride_k,
                          BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    k_offsets = tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K
    src_ptr = C_ptr + b * C_stride_b + s * C_stride_s + k_offsets * C_stride_k
    dst_ptr = processed_ptr + b * processed_stride_b + s * processed_stride_s + k_offsets * processed_stride_k
    vals = tl.load(src_ptr, mask=mask_k, other=0.0)
    tl.store(dst_ptr, vals, mask=mask_k)


# Triton kernel: copy C[b, s+T, :] -> processed_hidden[b, s, :] for s in [0, P)
@triton.jit
def _split_hidden_kernel(C_ptr, processed_ptr,
                         B, T, P, K,
                         C_stride_b, C_stride_s, C_stride_k,
                         processed_stride_b, processed_stride_p, processed_stride_k,
                         BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)  # s in [0, P)
    k_offsets = tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K
    src_ptr = C_ptr + b * C_stride_b + (s + T) * C_stride_s + k_offsets * C_stride_k
    dst_ptr = processed_ptr + b * processed_stride_b + s * processed_stride_p + k_offsets * processed_stride_k
    vals = tl.load(src_ptr, mask=mask_k, other=0.0)
    tl.store(dst_ptr, vals, mask=mask_k)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [B, P, K]
        # encoder_hidden_states: [B, T, K]
        # process_weight: [K, K]
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA tensors"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == K, "Shape mismatch"
        assert process_weight.shape[0] == K and process_weight.shape[1] == K, "process_weight must be [K, K]"

        # 1) Concatenate along sequence dimension using Triton
        S = T + P
        Acat = torch.empty((B, S, K), device=hidden_states.device, dtype=torch.float32)

        # Choose BLOCK_K for K-vectorization (e.g., 64 or 128). We can set to 64 for robustness.
        BLOCK_K = 64

        # Launch concatenate kernel: grid = (B, S)
        grid_concat = (B, S)
        _concatenate_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: Acat [B, S, K] @ process_weight.T [K, K] -> C [B, S, K]
        # We implement GEMM in Triton using a block matmul kernel.
        M = B * S
        # A is Acat flattened as [M, K] logically in kernel by using strides
        # W is process_weight.T [K, K]
        W_t = process_weight  # we pass [K, K] as is; we don't transpose since weight is [K, K]
        # Output C_flat [M, K] -> we'll reshape to [B, S, K] after kernel
        C_flat = torch.empty((M, K), device=hidden_states.device, dtype=torch.float32)

        # Choose tile sizes. Reasonable defaults:
        BLOCK_M = 64   # rows in output (M dimension)
        BLOCK_N = 64   # cols in output (N = K)
        BLOCK_K = 64   # reduction chunk

        # Grid over (B, tiles_M, tiles_N): tiles_M = cdiv(S, BLOCK_M), tiles_N = cdiv(K, BLOCK_N)
        tiles_M = triton.cdiv(S, BLOCK_M)
        tiles_N = triton.cdiv(K, BLOCK_N)
        grid_gemm = (B, tiles_M, tiles_N)

        _matmul_block_kernel[grid_gemm](
            Acat, W_t, C_flat,
            B, M, K, K,               # M = B*S, N=K, K=K
            Acat.stride(0), Acat.stride(2),   # A strides: m_stride=Acat.stride(0)=K, k_stride=Acat.stride(2)=1 for contiguous
            W_t.stride(0), W_t.stride(1),     # W strides: row_stride=W_t.stride(0)=K, col_stride=W_t.stride(1)=1
            C_flat.stride(0), C_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape C_flat to [B, S, K]
        C = C_flat.view(B, S, K)

        # 3) Split into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

        grid_split_e = (B, T)
        _split_encoder_kernel[grid_split_e](
            C, processed_encoder,
            B, T, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        grid_split_h = (B, P)
        _split_hidden_kernel[grid_split_h](
            C, processed_hidden,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
