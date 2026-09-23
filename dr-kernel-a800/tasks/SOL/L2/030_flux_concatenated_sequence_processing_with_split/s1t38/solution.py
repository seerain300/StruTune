import torch
import triton
import triton.language as tl


@triton.jit
def _cat_kernel(
    ehs_ptr, hs_ptr, out_ptr,
    B, T, P, K,
    BLOCK_K: tl.constexpr,
):
    # grid: (B, L, tiles_K)
    l = tl.program_id(1)
    tiles_k = tl.cdiv(K, BLOCK_K)
    k_idx = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_idx < K
    b = tl.program_id(0)

    # pointer to out[b, l, :]
    out_off = ((b * (T + P)) + l) * K + k_idx
    # decide source
    is_encoder = l < T
    if is_encoder:
        src_off = b * T * K + l * K + k_idx
    else:
        src_off = b * P * K + (l - T) * K + k_idx
    # masked load
    val = tl.load(ehs_ptr + src_off, mask=mask_k, other=0.0)
    # masked store
    tl.store(out_ptr + out_off, val, mask=mask_k)


@triton.jit
def _matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # grid: (B, tiles_M, tiles_N)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    # pointer matrices for A and C
    # A[b, m, k] shape is [M, K], contiguous in k
    # We treat A as a flattened [M, K] with row stride M and col stride 1 (contiguous K)
    # C is [M, N] similarly
    # But since b contributes only to row base offset, we compute:
    # A row base = m_offsets * K
    a_row_bases = m_offsets * K
    c_row_bases = m_offsets * N

    # initialize C tile
    C_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # A block: [BLOCK_M, BLOCK_K]
        A_block = tl.load(
            A_ptr + a_row_bases[:, None] + k_offsets[None, :],
            mask=(m_offsets[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        # W block: [BLOCK_K, BLOCK_N]
        W_block = tl.load(
            W_ptr + k_offsets[:, None] * N + n_offsets[None, :],
            mask=k_mask[:, None] & (n_offsets[None, :] < N),
            other=0.0,
        )
        C_tile += tl.dot(A_block, W_block)

    # store C tile
    tl.store(
        C_ptr + c_row_bases[:, None] + n_offsets[None, :],
        C_tile,
        mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N),
    )


@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, N, K,
    BLOCK_K: tl.constexpr,
):
    # grid: (B, T, tiles_K)
    t = tl.program_id(1)
    tiles_k = tl.cdiv(K, BLOCK_K)
    k_idx = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_idx < K
    b = tl.program_id(0)

    # C[b, t, k]
    C_off = b * N * K + t * K + k_idx
    val = tl.load(C_ptr + C_off, mask=mask_k, other=0.0)
    # out is [B, T, K]
    out_off = b * T * K + t * K + k_idx
    tl.store(out_ptr + out_off, val, mask=mask_k)


@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    BLOCK_K: tl.constexpr,
):
    # grid: (B, P, tiles_K)
    p = tl.program_id(1)
    tiles_k = tl.cdiv(K, BLOCK_K)
    k_idx = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_idx < K
    b = tl.program_id(0)

    # C[b, T + p, k]
    L = T + P
    C_off = b * L * K + (T + p) * K + k_idx
    val = tl.load(C_ptr + C_off, mask=mask_k, other=0.0)
    # out is [B, P, K]
    out_off = b * P * K + p * K + k_idx
    tl.store(out_ptr + out_off, val, mask=mask_k)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of:
        concatenated = cat([encoder_hidden_states, hidden_states], dim=1)
        processed = concatenated @ process_weight.T
        processed_encoder = processed[:, :text_seq_len, :]
        processed_hidden = processed[:, text_seq_len:, :]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton kernels"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L = T + P

        # 1) Concatenate streams along sequence dim using Triton
        ehs = encoder_hidden_states
        hs = hidden_states
        Acat = torch.empty((B, L, K), device=ehs.device, dtype=torch.float32)

        BLOCK_K_cat = 128
        grid_cat = (B, L, triton.cdiv(K, BLOCK_K_cat))
        _cat_kernel[grid_cat](
            ehs, hs, Acat,
            B, T, P, K,
            BLOCK_K=BLOCK_K_cat,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: Acat [B*L, K] @ process_weight.T [K, K] -> C [B*L, K]
        M = B * L
        Wt = process_weight.t()  # [K, K]
        # Ensure contiguous for pointer math
        Acat = Acat.contiguous()
        Wt = Wt.contiguous()
        C_flat = torch.empty((M, K), device=ehs.device, dtype=torch.float32)

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_Kg = 64
        grid_mm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_mm](
            Acat, Wt, C_flat,
            M, K, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_Kg,
            num_warps=4, num_stages=2
        )

        # Reshape to [B, L, K]
        C = C_flat.view(B, L, K)

        # 3) Split back using Triton
        processed_encoder = torch.empty((B, T, K), device=ehs.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=ehs.device, dtype=torch.float32)

        BLOCK_K_split = 128
        grid_e = (B, T, triton.cdiv(K, BLOCK_K_split))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, K, K,
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        grid_h = (B, P, triton.cdiv(K, BLOCK_K_split))
        _split_hidden_kernel[grid_h](
            C, processed_hidden,
            B, T, P, K,
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
