import torch
import triton
import triton.language as tl


@triton.jit
def _concat_kernel(
    enc_ptr,  # *fp32, [B, T, K]
    hid_ptr,  # *fp32, [B, P, K]
    out_ptr,  # *fp32, [B, L, K], L = T + P
    B, T, P, K,
    BLOCK_L: tl.constexpr,  # tile size along sequence (L)
    BLOCK_K: tl.constexpr,  # tile size along feature (K)
):
    # 3D grid: (batch, tiles over L, tiles over K)
    b = tl.program_id(0)
    l_tile = tl.program_id(1)
    k_tile = tl.program_id(2)

    L = T + P

    # Offsets along L and K for this program
    l_offsets = l_tile * BLOCK_L + tl.arange(0, BLOCK_L)  # [BLOCK_L]
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    # Valid mask for l,k
    valid_l = l_offsets < L
    valid_k = k_offsets < K

    # Build 2D mask [BLOCK_L, BLOCK_K]
    mask = valid_l[:, None] & valid_k[None, :]

    # For each l in this tile: source tensor and index
    # We'll vectorize over l and K tile, using broadcasting.
    # Compute source indices for enc and hid:
    # enc: b, l, k -> index = b*(T*K) + l*K + k
    # hid: b, l - T, k -> index = b*(P*K) + (l - T)*K + k
    # Note: for l < T, l - T is valid; for l >= T, l - T may be negative and masked.

    # Prepare offsets for enc and hid for broadcasting over l tile
    # Enc offsets
    enc_offsets = b * (T * K) + (l_offsets)[:, None] * K + k_offsets[None, :]  # shape [BLOCK_L, BLOCK_K]
    # Hidden offsets
    hid_offsets = b * (P * K) + (l_offsets - T)[:, None] * K + k_offsets[None, :]  # shape [BLOCK_L, BLOCK_K]

    # Validity mask for source tensors
    # enc is valid for all l < L, but we restrict to l < T region via is_from_enc
    is_from_enc = l_offsets < T
    # We cannot use masked load with per-element condition directly; we will select with tl.where after loading.
    # To avoid double loads, we build masks and select:
    enc_mask = mask & is_from_enc[:, None]
    hid_mask = mask & (~is_from_enc)[:, None]

    # Load values
    enc_vals = tl.load(enc_ptr + enc_offsets, mask=enc_mask, other=0.0)
    hid_vals = tl.load(hid_ptr + hid_offsets, mask=hid_mask, other=0.0)

    # Select source
    vals = tl.where(is_from_enc[:, None], enc_vals, hid_vals)

    # Compute destination offsets for out_ptr
    out_offsets = b * (L * K) + l_offsets[:, None] * K + k_offsets[None, :]
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@triton.jit
def _matmul_gemm_kernel(
    A_ptr,   # *fp32, [M, K], M = B * (T+P)
    W_ptr,   # *fp32, [K, K] (process_weight.T)
    C_ptr,   # *fp32, [M, K] (flat output)
    B, M, K,  # dimensions (batch, rows in A/C, feature dim)
    BLOCK_M: tl.constexpr,  # tile size along rows
    BLOCK_N: tl.constexpr,  # tile size along cols
    BLOCK_K: tl.constexpr,  # reduction tile
):
    # 3D grid: (batch, tiles over M, tiles over N)
    b = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)

    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    valid_m = m_offsets < M
    valid_n = n_offsets < K
    mask_out = valid_m[:, None] & valid_n[None, :]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        valid_k = k_offsets < K

        # Load A tile: A is [M, K], flatten over batch. For each b, rows m in [b*(T+P), ...]
        # We can index A_ptr with m_offsets and k_offsets:
        a_ptrs = A_ptr + m_offsets[:, None] * K + k_offsets[None, :]
        a_mask = valid_m[:, None] & valid_k[None, :]
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile: W is [K, K]
        w_ptrs = W_ptr + k_offsets[:, None] * K + n_offsets[None, :]
        w_mask = valid_k[:, None] & valid_n[None, :]
        W_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(A_tile, W_tile)  # [BLOCK_M, BLOCK_N]

    # Store result into C flat
    c_ptrs = C_ptr + m_offsets[:, None] * K + n_offsets[None, :]
    tl.store(c_ptrs, acc, mask=mask_out)


@triton.jit
def _copy_slice_kernel(
    C_ptr,        # *fp32, [B, T+P, K]
    out_ptr,      # *fp32, [B, S, K], S is T or P
    B, L, S, K,   # L=T+P, S=T or P
    BLOCK_S: tl.constexpr,  # tile over S (row count in slice)
    BLOCK_K: tl.constexpr,  # tile over K
):
    # 3D grid: (batch, tiles over S, tiles over K)
    b = tl.program_id(0)
    s_tile = tl.program_id(1)
    k_tile = tl.program_id(2)

    s_offsets = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    valid_s = s_offsets < S
    valid_k = k_offsets < K
    mask = valid_s[:, None] & valid_k[None, :]

    # Source indices in C3d: b, s, k
    src_offsets = b * (L * K) + s_offsets[:, None] * K + k_offsets[None, :]
    # Destination indices in out: b, s, k
    dst_offsets = b * (S * K) + s_offsets[:, None] * K + k_offsets[None, :]

    vals = tl.load(C_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # hidden_states: [B, P, K], encoder_hidden_states: [B, T, K], process_weight: [K, K]
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # Ensure inputs are on CUDA and dtype float32
        device = hidden_states.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"

        # 1) Concatenate in Triton: [B, T+P, K]
        L = T + P
        Acat = torch.empty((B, L, K), device=device, dtype=torch.float32)

        BLOCK_L = 64
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(K, BLOCK_K))
        _concat_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: Acat [B*L, K] @ W [K, K] -> C_flat [B*L, K]
        M = B * L
        # Flatten Acat to [M, K]
        A_flat = Acat.reshape(M, K).contiguous()
        # W is process_weight.T with shape [K, K]
        W = process_weight.t().contiguous()

        C_flat = torch.empty((M, K), device=device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_gemm_kernel[grid_gemm](
            A_flat, W, C_flat,
            B, M, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Reshape C_flat to [B, L, K]
        C3d = C_flat.reshape(B, L, K)

        # 4) Split in Triton
        processed_encoder = torch.empty((B, T, K), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=torch.float32)

        BLOCK_S = 64
        BLOCK_K = 64

        grid_enc = (B, triton.cdiv(T, BLOCK_S), triton.cdiv(K, BLOCK_K))
        _copy_slice_kernel[grid_enc](
            C3d, processed_encoder,
            B, L, T, K,
            BLOCK_S=BLOCK_S, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        grid_hid = (B, triton.cdiv(P, BLOCK_S), triton.cdiv(K, BLOCK_K))
        _copy_slice_kernel[grid_hid](
            C3d, processed_hidden,
            B, L, P, K,
            BLOCK_S=BLOCK_S, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


# Optional: if you want to keep the original Model and run ModelNew for testing, you can use:
# def run(hidden_states, encoder_hidden_states, process_weight):
#     # ModelNew expects same args
#     model = ModelNew().cuda()
#     with torch.no_grad():
#         e, h = model(hidden_states.cuda(), encoder_hidden_states.cuda(), process_weight.cuda())
#     return e, h


def run(*args):
    return ModelNew()(*args)
