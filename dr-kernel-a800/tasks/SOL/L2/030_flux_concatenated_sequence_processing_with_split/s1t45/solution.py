import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_kernel(
    encoder_ptr,        # *f32, [B, T, K]
    hidden_ptr,         # *f32, [B, P, K]
    out_ptr,            # *f32, [B, T+P, K]
    B: tl.int32, T: tl.int32, P: tl.int32, K: tl.int32,
    encoder_stride_b: tl.int32, encoder_stride_t: tl.int32, encoder_stride_k: tl.int32,
    hidden_stride_b: tl.int32, hidden_stride_p: tl.int32, hidden_stride_k: tl.int32,
    out_stride_b: tl.int32, out_stride_l: tl.int32, out_stride_k: tl.int32,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    k_block = tl.program_id(2)

    # guard for l
    if l >= (T + P):
        return

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # compute pointers
    out_ptrs = out_ptr + b * out_stride_b + l * out_stride_l + k_offsets * out_stride_k

    if l < T:
        enc_ptrs = encoder_ptr + b * encoder_stride_b + l * encoder_stride_t + k_offsets * encoder_stride_k
        vals = tl.load(enc_ptrs, mask=k_mask, other=0.0)
        tl.store(out_ptrs, vals, mask=k_mask)
    else:
        h_idx = l - T
        hid_ptrs = hidden_ptr + b * hidden_stride_b + h_idx * hidden_stride_p + k_offsets * hidden_stride_k
        vals = tl.load(hid_ptrs, mask=k_mask, other=0.0)
        tl.store(out_ptrs, vals, mask=k_mask)


@triton.jit
def _gemm_matmul_kernel(
    A_ptr,  # *f32, [M, K]
    W_ptr,  # *f32, [K, K]
    C_ptr,  # *f32, [M, K]
    M: tl.int32, N: tl.int32, Kdim: tl.int32,
    A_stride_m: tl.int32, A_stride_k: tl.int32,
    W_stride_k: tl.int32, W_stride_n: tl.int32,
    C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # grid = (B, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K in chunks
    for k0 in range(0, Kdim, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < Kdim

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # W tile: [BLOCK_K, BLOCK_N] from W[k, n]
        w_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        w_mask = k_mask[:, None] & n_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # accumulate
        acc += tl.dot(a, w)

    # write C
    c_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr,               # *f32, [B, T+P, K]
    processed_ptr,       # *f32, [B, T, K]
    B: tl.int32, T: tl.int32, P: tl.int32, K: tl.int32,
    C_stride_b: tl.int32, C_stride_l: tl.int32, C_stride_k: tl.int32,
    processed_stride_b: tl.int32, processed_stride_t: tl.int32, processed_stride_k: tl.int32,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    k_block = tl.program_id(2)
    if b >= B or t >= T:
        return

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    C_ptrs = C_ptr + b * C_stride_b + t * C_stride_l + k_offsets * C_stride_k
    processed_ptrs = processed_ptr + b * processed_stride_b + t * processed_stride_t + k_offsets * processed_stride_k

    vals = tl.load(C_ptrs, mask=k_mask, other=0.0)
    tl.store(processed_ptrs, vals, mask=k_mask)


@triton.jit
def _split_hidden_kernel(
    C_ptr,               # *f32, [B, T+P, K]
    processed_ptr,       # *f32, [B, P, K]
    B: tl.int32, T: tl.int32, P: tl.int32, K: tl.int32,
    C_stride_b: tl.int32, C_stride_l: tl.int32, C_stride_k: tl.int32,
    processed_stride_b: tl.int32, processed_stride_p: tl.int32, processed_stride_k: tl.int32,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    p = tl.program_id(1)
    k_block = tl.program_id(2)
    if b >= B or p >= P:
        return

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # start from index T
    C_ptrs = C_ptr + b * C_stride_b + (T + p) * C_stride_l + k_offsets * C_stride_k
    processed_ptrs = processed_ptr + b * processed_stride_b + p * processed_stride_p + k_offsets * processed_stride_k

    vals = tl.load(C_ptrs, mask=k_mask, other=0.0)
    tl.store(processed_ptrs, vals, mask=k_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # hidden_states: [B, P, K], encoder_hidden_states: [B, T, K], process_weight: [K, K]
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3 and process_weight.ndim == 2
        B, P, K = hidden_states.shape
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == K
        T = encoder_hidden_states.shape[1]
        assert process_weight.shape[0] == K and process_weight.shape[1] == K

        # Ensure float32 and contiguous for Triton
        dtype = torch.float32
        encoder = encoder_hidden_states.contiguous().to(dtype)
        hidden = hidden_states.contiguous().to(dtype)
        W = process_weight.contiguous().to(dtype)  # [K, K], no bias

        # Triton concatenate into Acat: [B, T+P, K]
        total_L = T + P
        Acat = torch.empty((B, total_L, K), device=hidden.device, dtype=dtype)

        BLOCK_K = 128  # tile along K; works for typical K
        grid_concat = (B, total_L, triton.cdiv(K, BLOCK_K))
        _concatenate_kernel[grid_concat](
            encoder, hidden, Acat,
            B, T, P, K,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_K=BLOCK_K, num_warps=4, num_stages=2
        )

        # GEMM: Acat [B*(T+P), K] @ W [K, K] -> C_flat [B*(T+P), K]
        M = B * total_L
        N = K  # output dimension equals K
        C_flat = torch.empty((M, N), device=hidden.device, dtype=dtype)

        # A is Acat flattened to [M, K] by using strides; but we can pass as is:
        # We need pointer to A as 1D. Create a view for A as [M, K]:
        # To pass [M, K], we can create a contiguous row-major view:
        A_2d = Acat.reshape(M, N).contiguous()  # [M, K] contiguous

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32  # reduction tile

        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _gemm_matmul_kernel[grid_gemm](
            A_2d, W, C_flat,
            M, N, K,
            A_2d.stride(0), A_2d.stride(1),
            W.stride(0), W.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape C_flat to [B, T+P, K]
        C = C_flat.view(B, total_L, K)

        # Triton split into processed_encoder [B, T, K] and processed_hidden [B, P, K]
        processed_encoder = torch.empty((B, T, K), device=hidden.device, dtype=dtype)
        BLOCK_Ks = 128
        grid_e = (B, T, triton.cdiv(K, BLOCK_Ks))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_Ks, num_warps=4, num_stages=2
        )

        processed_hidden = torch.empty((B, P, K), device=hidden.device, dtype=dtype)
        grid_i = (B, P, triton.cdiv(K, BLOCK_Ks))
        _split_hidden_kernel[grid_i](
            C, processed_hidden,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_Ks, num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
