import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,   # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,    # *hidden_states [B, L_img, D]
    dst_ptr,   # *output concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_S: tl.constexpr,  # tile along sequence (S) dimension
    BLOCK_D: tl.constexpr,  # tile along hidden (D) dimension
):
    # 2D grid over (S-tiles, B)
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)

    S_total = L_txt + L_img

    # Offsets for sequence and hidden dims
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offsets = tl.arange(0, BLOCK_D)

    # Masks to avoid OOB
    mask = (s_offsets < S_total) & (d_offsets < D)

    # Base pointers for the current batch
    ehs_base = ehs_ptr + pid_b * ehs_stride_b
    hs_base = hs_ptr + pid_b * hs_stride_b
    dst_base = dst_ptr + pid_b * dst_stride_b

    # Masks for the two parts
    mask_txt = (s_offsets < L_txt) & (d_offsets < D)
    s_img = s_offsets - L_txt
    mask_img = (s_img >= 0) & (s_img < L_img) & (d_offsets < D)

    # Pointers for loads/stores
    ehs_ptrs = ehs_base + s_offsets[:, None] * ehs_stride_s + d_offsets[None, :] * ehs_stride_d
    hs_ptrs = hs_base + s_img[:, None] * hs_stride_s + d_offsets[None, :] * hs_stride_d
    dst_ptrs = dst_base + s_offsets[:, None] * dst_stride_s + d_offsets[None, :] * dst_stride_d

    # Load and store the encoder part
    ehs_vals = tl.load(ehs_ptrs, mask=mask_txt[:, None], other=0.0)
    tl.store(dst_ptrs, ehs_vals, mask=mask_txt[:, None])

    # Load and store the hidden part
    hs_vals = tl.load(hs_ptrs, mask=mask_img[:, None], other=0.0)
    tl.store(dst_ptrs, hs_vals, mask=mask_img[:, None])

    # The dst tile now contains concatenated rows: first L_txt from ehs, next L_img from hs.


@triton.jit
def split_seqs_kernel(
    src_ptr,      # *processed [B, L_txt + L_img, D]
    out_ptr,      # *output for first L_txt [B, L_txt, D]
    out_ptr2,     # *output for remaining [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    src_stride_b: tl.int32, src_stride_s: tl.int32, src_stride_d: tl.int32,
    out_stride_b: tl.int32, out_stride_s: tl.int32, out_stride_d: tl.int32,
    out2_stride_b: tl.int32, out2_stride_s: tl.int32, out2_stride_d: tl.int32,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 2D grid over (S-tiles, B)
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)

    # For processed_encoder: s in [0, L_txt)
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offsets = tl.arange(0, BLOCK_D)
    mask1 = (s_offsets < L_txt) & (d_offsets < D)

    src_base = src_ptr + pid_b * src_stride_b
    out_base = out_ptr + pid_b * out_stride_b

    src_ptrs = src_base + s_offsets[:, None] * src_stride_s + d_offsets[None, :] * src_stride_d
    out_ptrs = out_base + s_offsets[:, None] * out_stride_s + d_offsets[None, :] * out_stride_d

    vals1 = tl.load(src_ptrs, mask=mask1, other=0.0)
    tl.store(out_ptrs, vals1, mask=mask1)

    # For processed_hidden: s in [L_txt, L_txt + L_img)
    s_img = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask2 = (s_img < L_img) & (d_offsets < D)

    src_base2 = src_ptr + pid_b * src_stride_b
    out2_base = out_ptr2 + pid_b * out2_stride_b

    src_ptrs2 = src_base2 + (s_img[:, None] + L_txt) * src_stride_s + d_offsets[None, :] * src_stride_d
    out2_ptrs = out2_base + s_img[:, None] * out2_stride_s + d_offsets[None, :] * out2_stride_d

    vals2 = tl.load(src_ptrs2, mask=mask2, other=0.0)
    tl.store(out2_ptrs, vals2, mask=mask2)


@triton.jit
def bmm_kernel(
    A_ptr, Wt_ptr, C_ptr,
    B: tl.int32,
    M: tl.int32,  # L_txt + L_img
    N: tl.int32,  # D
    K: tl.int32,  # D
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    Wt_stride_k: tl.int32, Wt_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid over (B, tiles along M, tiles along N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k0 in range(0, K, BLOCK_K):
        # Compute current k range
        k_curr = k0 + k_offsets

        # Pointers for A[b, m, k] and Wt[k, n]
        A_ptrs = A_ptr + pid_b * A_stride_b + m_offsets[:, None] * A_stride_m + k_curr[None, :] * A_stride_k
        Wt_ptrs = Wt_ptr + k_curr[:, None] * Wt_stride_k + n_offsets[None, :] * Wt_stride_n

        # Masks for loads
        mask_k = k_curr < K
        A_mask = mask_m[:, None] & mask_k[None, :]
        Wt_mask = mask_k[:, None] & mask_n[None, :]

        # Load tiles
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)
        Wt_tile = tl.load(Wt_ptrs, mask=Wt_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(A_tile.to(tl.float32), Wt_tile.to(tl.float32))

    # Store result
    C_ptrs = C_ptr + pid_b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, L_img, D]
        encoder_hidden_states: [B, L_txt, D]
        process_weight: [D, D]
        Returns:
          processed_encoder: [B, L_txt, D]
          processed_hidden: [B, L_img, D]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton"
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D, "hidden_dim mismatch"

        # 1) Concatenate along sequence dimension using Triton
        S_total = L_txt + L_img
        concatenated = torch.empty((B, S_total, D), device=hidden_states.device, dtype=hidden_states.dtype)

        # Strides
        ehs_stride_b, ehs_stride_s, ehs_stride_d = encoder_hidden_states.stride()
        hs_stride_b, hs_stride_s, hs_stride_d = hidden_states.stride()
        dst_stride_b, dst_stride_s, dst_stride_d = concatenated.stride()

        # Tile sizes for concat
        BLOCK_S = 64
        BLOCK_D = 64
        grid_concat = (triton.cdiv(S_total, BLOCK_S), B)

        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, L_txt, L_img, D,
            ehs_stride_b, ehs_stride_s, ehs_stride_d,
            hs_stride_b, hs_stride_s, hs_stride_d,
            dst_stride_b, dst_stride_s, dst_stride_d,
            BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # 2) Apply linear projection using Triton GEMM: C = A @ Wt
        # A: [B, M, K] = concatenated, M = L_txt + L_img, K = D
        # Wt: [K, N] = process_weight.T, N = D
        # Output C: [B, M, N] = [B, L_txt+L_img, D]
        A = concatenated
        Wt = process_weight.transpose(0, 1).contiguous()  # [D, D]

        # Prepare output C in float32 for accumulation
        C = torch.empty((B, S_total, D), device=hidden_states.device, dtype=torch.float32)

        A_stride_b, A_stride_m, A_stride_k = A.stride()
        Wt_stride_k, Wt_stride_n = Wt.stride()  # Wt is [K=hidden_dim, N=hidden_dim]
        C_stride_b, C_stride_m, C_stride_n = C.stride()

        # Tile sizes for matmul
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_matmul = (triton.cdiv(S_total, BLOCK_M), triton.cdiv(D, BLOCK_N), B)

        bmm_kernel[grid_matmul](
            A, Wt, C,
            B, S_total, D, D,
            A_stride_b, A_stride_m, A_stride_k,
            Wt_stride_k, Wt_stride_n,
            C_stride_b, C_stride_m, C_stride_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split using Triton
        processed_encoder = torch.empty((B, L_txt, D), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=hidden_states.device, dtype=hidden_states.dtype)

        # Cast C back to the original dtype (original tensors are float32, so this is a no-op; keep dtype consistent)
        C_cast = C  # we keep float32; original example uses float32

        # Strides for processed (C_cast) and outputs
        src_stride_b, src_stride_s, src_stride_d = C_cast.stride()
        out_stride_b, out_stride_s, out_stride_d = processed_encoder.stride()
        out2_stride_b, out2_stride_s, out2_stride_d = processed_hidden.stride()

        grid_split = (triton.cdiv(L_txt, 64), B)  # tile along sequence

        split_seqs_kernel[grid_split](
            C_cast, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            src_stride_b, src_stride_s, src_stride_d,
            out_stride_b, out_stride_s, out_stride_d,
            out2_stride_b, out2_stride_s, out2_stride_d,
            BLOCK_S=64, BLOCK_D=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
