import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, P, K,
    encoder_stride_b, encoder_stride_t, encoder_stride_k,
    hidden_stride_b, hidden_stride_p, hidden_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles over L=T+P, tiles over K)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_k = tl.program_id(2)

    L = T + P
    L_offsets = pid_l * BLOCK_T + tl.arange(0, BLOCK_T)       # [BLOCK_T]
    K_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)       # [BLOCK_K]
    mask_l = L_offsets < L
    mask_k = K_offsets < K

    # For each l in [0, L), determine if it comes from encoder or hidden
    use_enc = L_offsets < T  # [BLOCK_T]: boolean per l

    # Compute pointers for output
    out_ptrs = out_ptr + pid_b * out_stride_b + L_offsets[:, None] * out_stride_l + K_offsets[None, :] * out_stride_k

    # Compute pointers for enc/hid loads with masks
    # Enc source addresses
    enc_addrs = encoder_ptr + pid_b * encoder_stride_b + L_offsets[:, None] * encoder_stride_t + K_offsets[None, :] * encoder_stride_k
    mask_enc = mask_l[:, None] & use_enc[:, None] & mask_k[None, :]
    vals_enc = tl.load(enc_addrs, mask=mask_enc, other=0.0)

    # Hidden source addresses: l_hid = l_offsets - T
    l_hid = L_offsets[:, None] - T
    hid_addrs = hidden_ptr + pid_b * hidden_stride_b + l_hid * hidden_stride_p + K_offsets[None, :] * hidden_stride_k
    mask_hid = mask_l[:, None] & (~use_enc)[:, None] & mask_k[None, :]
    vals_hid = tl.load(hid_addrs, mask=mask_hid, other=0.0)

    # Select per element: if use_enc -> vals_enc else vals_hid
    out_vals = tl.where(use_enc[:, None], vals_enc, vals_hid)

    # Store to output with combined mask over l and k
    tl.store(out_ptrs, out_vals, mask=(mask_l[:, None] & mask_k[None, :]))


@triton.jit
def _gemm_bmn_kernel(
    A_ptr, W_ptr, C_ptr,
    B, T, P, K, N,  # N == K here
    A_stride_b, A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid over (B, tiles over M=B*(T+P), tiles over N=K)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    M_total = B * (T + P)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = m_offsets < M_total
    mask_n = n_offsets < N

    # Accumulator for C_tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load A_tile: A[m, k]
        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_mask = mask_m[:, None] & mask_k[None, :]
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W_tile: W[k, n]
        w_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        w_mask = mask_k[:, None] & mask_n[None, :]
        W_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        acc += tl.dot(A_tile, W_tile)

    # Store result
    C_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=store_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    C_stride_b, C_stride_m, C_stride_k,
    out_stride_b, out_stride_t, out_stride_k,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles over T, tiles over K)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)       # [BLOCK_T]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)       # [BLOCK_K]
    mask_t = t_offsets < T
    mask_k = k_offsets < K

    # Copy C[b, t, :] -> out[b, t, :]
    C_ptrs = C_ptr + pid_b * C_stride_b + t_offsets[:, None] * C_stride_m + k_offsets[None, :] * C_stride_k
    out_ptrs = out_ptr + pid_b * out_stride_b + t_offsets[:, None] * out_stride_t + k_offsets[None, :] * out_stride_k

    store_mask = mask_t[:, None] & mask_k[None, :]
    vals = tl.load(C_ptrs, mask=store_mask, other=0.0)
    tl.store(out_ptrs, vals, mask=store_mask)


@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    C_stride_b, C_stride_m, C_stride_k,
    out_stride_b, out_stride_p, out_stride_k,
    BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles over P, tiles over K)
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_k = tl.program_id(2)

    p_offsets = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)       # [BLOCK_P]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)       # [BLOCK_K]
    mask_p = p_offsets < P
    mask_k = k_offsets < K

    # Copy C[b, T + p, :] -> out[b, p, :]
    C_ptrs = C_ptr + pid_b * C_stride_b + (T + p_offsets)[:, None] * C_stride_m + k_offsets[None, :] * C_stride_k
    out_ptrs = out_ptr + pid_b * out_stride_b + p_offsets[:, None] * out_stride_p + k_offsets[None, :] * out_stride_k

    store_mask = mask_p[:, None] & mask_k[None, :]
    vals = tl.load(C_ptrs, mask=store_mask, other=0.0)
    tl.store(out_ptrs, vals, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim in Triton.
        - Perform GEMM with process_weight.T in Triton.
        - Split outputs in Triton.
        Returns (processed_encoder, processed_hidden).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be CUDA for Triton kernels."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        N = K  # process_weight.T has shape [K, K]

        # Ensure contiguity
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = hidden_states.contiguous()
        process_weight_t = process_weight.t().contiguous()  # shape [K, K]

        # 1) Triton concatenation: Acat [B, T+P, K]
        L = T + P
        Acat = torch.empty((B, L, K), device=hidden_states.device, dtype=torch.float32)

        # Launch concatenation kernel
        BLOCK_T = 128
        BLOCK_K = 128
        grid_concat = (B, triton.cdiv(L, BLOCK_T), triton.cdiv(K, BLOCK_K))
        _concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Triton GEMM: Acat [B*(T+P), K] @ W_t [K, K] -> C_flat [B*(T+P), K]
        M_total = B * L
        C_flat = torch.empty((M_total, K), device=hidden_states.device, dtype=torch.float32)

        # Launch GEMM kernel with 3D grid over (B, tiles over M, tiles over N)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(M_total, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _gemm_bmn_kernel[grid_gemm](
            Acat, process_weight_t, C_flat,
            B, T, P, K, N,
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            process_weight_t.stride(0), process_weight_t.stride(1),
            C_flat.stride(0), C_flat.stride(1), C_flat.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape C_flat to [B, L, K] and split in Triton
        C = C_flat.view(B, L, K)

        # Triton split: processed_encoder [B, T, K]
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        BLOCK_T_split = 128
        BLOCK_K_split = 128
        grid_e = (B, triton.cdiv(T, BLOCK_T_split), triton.cdiv(K, BLOCK_K_split))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_T=BLOCK_T_split, BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2,
        )

        # Triton split: processed_hidden [B, P, K]
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)
        BLOCK_P_split = 128
        grid_h = (B, triton.cdiv(P, BLOCK_P_split), triton.cdiv(K, BLOCK_K_split))
        _split_hidden_kernel[grid_h](
            C, processed_hidden,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_P=BLOCK_P_split, BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
