import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    enc_ptr, hid_ptr, out_ptr,
    B, T, P, K,
    enc_stride_b, enc_stride_t, enc_stride_k,
    hid_stride_b, hid_stride_p, hid_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, L) where L = T + P
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    L = T + P

    # K tile
    k_offsets = tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source: if l < T, take encoder; else take hidden at index l - T
    use_encoder = pid_l < T

    # Output pointer for this (b, l)
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_l * out_stride_l + k_offsets * out_stride_k

    if use_encoder:
        src_ptrs = enc_ptr + pid_b * enc_stride_b + pid_l * enc_stride_t + k_offsets * enc_stride_k
    else:
        src_l = pid_l - T
        src_ptrs = hid_ptr + pid_b * hid_stride_b + src_l * hid_stride_p + k_offsets * hid_stride_k

    vals = tl.load(src_ptrs, mask=mask_k, other=0.0)
    tl.store(out_ptrs, vals, mask=mask_k)


@triton.jit
def _gemm_bmn_kernel(
    A_ptr, W_ptr, C_ptr,
    B, T, P, K, N,  # N == K here
    A_stride_b, A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, tiles over M=B*(T+P), tiles over N=K)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    M_total = B * (T + P)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    mask_m = m_offsets < M_total
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Map m to (b, l): m = b*(T+P) + l
        b_idx = m_offsets // (T + P)
        l_idx = m_offsets - b_idx * (T + P)  # l in [0, T+P)

        # A[b, l, k]
        A_ptrs = A_ptr + b_idx[:, None] * A_stride_b + l_idx[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = mask_m[:, None] & mask_k[None, :]
        A_vals = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # W[k, n]
        W_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        W_mask = mask_k[:, None] & mask_n[None, :]
        W_vals = tl.load(W_ptrs, mask=W_mask, other=0.0)

        acc += tl.dot(A_vals, W_vals)

    # Store results to C at positions (b, m, n)
    C_ptrs = C_ptr + pid_b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, K,
    C_stride_b, C_stride_l, C_stride_k,
    out_stride_b, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, T)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    k_offsets = tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Copy C[b, l, :] -> out[b, l, :]
    C_ptrs = C_ptr + pid_b * C_stride_b + pid_l * C_stride_l + k_offsets * C_stride_k
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_l * out_stride_k + k_offsets * out_stride_k
    vals = tl.load(C_ptrs, mask=mask_k, other=0.0)
    tl.store(out_ptrs, vals, mask=mask_k)


@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    C_stride_b, C_stride_l, C_stride_k,
    out_stride_b, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, P)
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    k_offsets = tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Copy C[b, T + p, :] -> out[b, p, :]
    l_src = T + pid_p
    C_ptrs = C_ptr + pid_b * C_stride_b + l_src * C_stride_l + k_offsets * C_stride_k
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_p * out_stride_k + k_offsets * out_stride_k
    vals = tl.load(C_ptrs, mask=mask_k, other=0.0)
    tl.store(out_ptrs, vals, mask=mask_k)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate along sequence dimension via Triton.
        - Linear projection via Triton GEMM.
        - Split outputs via Triton.
        """
        # Ensure CUDA and dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == K, "Encoder hidden dim must match image hidden dim."
        assert process_weight.shape[0] == K and process_weight.shape[1] == K, "process_weight must be [K, K]."

        # Allocate concatenated output [B, L, K], L = T + P
        L = T + P
        Acat = torch.empty((B, L, K), device=hidden_states.device, dtype=torch.float32)

        # Launch concatenation kernel: grid = (B, L)
        BLOCK_K = 128  # tile size for K
        grid_concat = (B, L)
        _concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Prepare A as [M, K] where M = B * L, W as [K, K]
        M_total = B * L
        # View Acat as 2D: A[i, k] = Acat[b, l, k]
        A_2d = Acat.view(M_total, K).contiguous()
        W_t = process_weight.t().contiguous()  # [K, K]

        # Allocate C as [M, K] then reshape to [B, L, K]
        C_full = torch.empty((M_total, K), device=hidden_states.device, dtype=torch.float32)

        # Launch GEMM kernel: grid = (B, tiles over M, tiles over N=K)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(M_total, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _gemm_bmn_kernel[grid_gemm](
            A_2d, W_t, C_full,
            B, T, P, K, K,  # N == K
            A_2d.stride(0), A_2d.stride(1),
            W_t.stride(0), W_t.stride(1),
            C_full.stride(0), C_full.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Reshape C_full to [B, L, K]
        C_reshaped = C_full.view(B, L, K)

        # Allocate outputs and split via Triton
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

        # Launch split kernels: grid = (B, T) and (B, P)
        BLOCK_K_split = 128
        grid_encoder = (B, T)
        _split_encoder_kernel[grid_encoder](
            C_reshaped, processed_encoder,
            B, T, K,
            C_reshaped.stride(0), C_reshaped.stride(1), C_reshaped.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2,
        )

        grid_hidden = (B, P)
        _split_hidden_kernel[grid_hidden](
            C_reshaped, processed_hidden,
            B, T, P, K,
            C_reshaped.stride(0), C_reshaped.stride(1), C_reshaped.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
