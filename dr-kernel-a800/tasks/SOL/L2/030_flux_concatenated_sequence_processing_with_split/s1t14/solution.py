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
    # Grid: (B, T+P)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    # Compute sequence length
    total_L = T + P

    # Determine whether this l comes from encoder or hidden
    use_enc = pid_l < T

    # K loop over tiles
    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        if use_enc:
            # Address for encoder[b, pid_l, k_offsets]
            enc_addrs = enc_ptr + pid_b * enc_stride_b + pid_l * enc_stride_t + k_offsets * enc_stride_k
            out_addrs = out_ptr + pid_b * out_stride_b + pid_l * out_stride_l + k_offsets * out_stride_k
        else:
            # Address for hidden[b, pid_l - T, k_offsets]
            p_idx = pid_l - T
            hid_addrs = hid_ptr + pid_b * hid_stride_b + p_idx * hid_stride_p + k_offsets * hid_stride_k
            out_addrs = out_ptr + pid_b * out_stride_b + pid_l * out_stride_l + k_offsets * out_stride_k

        vals = tl.load(hid_addrs if use_enc else enc_addrs, mask=mask_k, other=0.0)
        tl.store(out_addrs, vals, mask=mask_k)
        k_start += BLOCK_K


@triton.jit
def _gemm_bmn_kernel(
    A_ptr, W_ptr, C_ptr,
    B, T, P, K,  # M_total = B * (T + P), N = K
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

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = m_offsets < M_total
    mask_n = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Compute A_tile [BLOCK_M, BLOCK_K]: A[m, k]
        A_addrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = mask_m[:, None] & mask_k[None, :]
        A_tile = tl.load(A_addrs, mask=A_mask, other=0.0)

        # Compute W_tile [BLOCK_K, BLOCK_N]: W[k, n]
        W_addrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        W_mask = mask_k[:, None] & mask_n[None, :]
        W_tile = tl.load(W_addrs, mask=W_mask, other=0.0)

        # acc += A_tile @ W_tile
        acc += tl.dot(A_tile, W_tile)

        k_start += BLOCK_K

    # Store C_tile
    C_addrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_addrs, acc, mask=C_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, K,
    C_stride_b, C_stride_m, C_stride_k,
    out_stride_b, out_stride_t, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, T, tiles over K)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        C_addrs = C_ptr + pid_b * C_stride_b + pid_t * C_stride_m + k_offsets * C_stride_k
        out_addrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + k_offsets * out_stride_k

        vals = tl.load(C_addrs, mask=mask_k, other=0.0)
        tl.store(out_addrs, vals, mask=mask_k)

        k_start += BLOCK_K


@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, P, K,
    C_stride_b, C_stride_m, C_stride_k,
    out_stride_b, out_stride_p, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, P, tiles over K)
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Source starts at m = T in C
        C_addrs = C_ptr + pid_b * C_stride_b + (pid_p + 0) * C_stride_m + k_offsets * C_stride_k
        out_addrs = out_ptr + pid_b * out_stride_b + pid_p * out_stride_p + k_offsets * out_stride_k

        vals = tl.load(C_addrs, mask=mask_k, other=0.0)
        tl.store(out_addrs, vals, mask=mask_k)

        k_start += BLOCK_K


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we only launch Triton kernels

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, P, K]
        encoder_hidden_states: [B, T, K]
        process_weight: [K, K]
        Returns: (processed_encoder: [B, T, K], processed_hidden: [B, P, K])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [B, *, K]"
        assert process_weight.dim() == 2, "process_weight must be 2D [K, K]"
        B, P, K = hidden_states.shape
        B2, T, K2 = encoder_hidden_states.shape
        assert B == B2 and K == K2, "Batch and feature dimensions must match between inputs"
        K_w, K_w2 = process_weight.shape
        assert K_w == K and K_w2 == K, "process_weight must be [K, K]"

        device = hidden_states.device
        dtype = torch.float32

        # Ensure contiguity and dtype
        encoder_hidden = encoder_hidden_states.contiguous().to(dtype)
        hidden = hidden_states.contiguous().to(dtype)
        weight_T = process_weight.contiguous().to(dtype)  # [K, K]

        # 1) Concatenate along sequence using Triton
        total_L = T + P
        Acat = torch.empty((B, total_L, K), device=device, dtype=dtype)

        # Launch concat kernel: grid over (B, total_L), loop over K in tiles
        BLOCK_K = 128
        grid_concat = (B, total_L)
        _concat_seq_kernel[grid_concat](
            encoder_hidden, hidden, Acat,
            B, T, P, K,
            encoder_hidden.stride(0), encoder_hidden.stride(1), encoder_hidden.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: Acat @ weight_T, Acat is [B, total_L, K], weight_T is [K, K]
        # We'll treat Acat as [M=B*total_L, K], weight_T as [K, K], output C as [M, K]
        M_total = B * total_L
        C_flat = torch.empty((M_total, K), device=device, dtype=dtype)

        # Prepare A view: [M, K] by flattening batch and sequence
        A_flat = Acat.reshape(M_total, K)

        # 3D grid over (B, tiles of M, tiles of N)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K_red = 64
        grid_gemm = (B, triton.cdiv(M_total, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _gemm_bmn_kernel[grid_gemm](
            A_flat, weight_T, C_flat,
            B, T, P, K,  # M_total is implied
            A_flat.stride(0), A_flat.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K_red,
            num_warps=4, num_stages=3
        )

        # 3) Reshape C_flat back to [B, total_L, K]
        C = C_flat.reshape(B, total_L, K)

        # 4) Split into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, T, K), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=dtype)

        BLOCK_K_split = 128
        grid_e = (B, T)
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        grid_h = (B, P)
        _split_hidden_kernel[grid_h](
            C, processed_hidden,
            B, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
