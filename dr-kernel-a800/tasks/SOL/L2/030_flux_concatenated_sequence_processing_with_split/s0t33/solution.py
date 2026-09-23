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
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    # program ids
    b_block = tl.program_id(0)  # tile along B
    s_block = tl.program_id(1)  # tile along sequence (S = L_txt + L_img)
    d_block = tl.program_id(2)  # tile along feature (D)

    # compute indices within tiles
    b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)  # BLOCK_B is implicit here; we use one b per program
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)

    # mask: valid batch indices only (BLOCK_B=1 so b[0] is safe; mask_b unnecessary)
    # Note: We'll implement a 3D grid over (B, tiles_S, tiles_D), so each program handles one b.
    # For simplicity and correctness, launch grid as (B, ceil(L/M), ceil(D/N)). We set BLOCK_B=1.
    # Therefore, b is scalar and we do not need masks for b.

    # dst base pointer for each b
    dst_base = dst_ptr + b * dst_stride_b  # b is scalar (size 1) as grid dim 0 spans B

    # For each s in the tile, determine source pointer (ehs or hs)
    # s < L_txt => from ehs, else from hs (offset = s - L_txt)
    for i in range(BLOCK_S):
        s = s_offsets[i]
        valid_s = s < (L_txt + L_img)
        # For dst: s within range
        dst_ptr_s = dst_base + s * dst_stride_s

        # Decide source tensor
        # If s < L_txt, use ehs; else use hs
        use_ehs = s < L_txt

        if use_ehs:
            ehs_ptr_s = ehs_ptr + b * ehs_stride_b + s * ehs_stride_s
            ehs_stride_d = ehs_stride_d
            # load a vector across D tile
            d_vec = d_offsets
            mask_d = d_vec < D
            vals = tl.load(ehs_ptr_s + d_vec * ehs_stride_d, mask=mask_d, other=0.0)
            tl.store(dst_ptr_s + d_vec * dst_stride_d, vals, mask=mask_d & valid_s)
        else:
            hs_s = s - L_txt
            hs_ptr_s = hs_ptr + b * hs_stride_b + hs_s * hs_stride_s
            d_vec = d_offsets
            mask_d = d_vec < D
            vals = tl.load(hs_ptr_s + d_vec * hs_stride_d, mask=mask_d, other=0.0)
            tl.store(dst_ptr_s + d_vec * dst_stride_d, vals, mask=mask_d & valid_s)


@triton.jit
def gemm_bmn_kernel(
    A_ptr,      # *concatenated [B, M, K]
    Wt_ptr,     # *process_weight.T [K, N]
    C_ptr,      # *output [B, M, N]
    B: tl.int32,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    Wt_stride_k: tl.int32, Wt_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 3D grid over (tiles along B, tiles along M, tiles along N)
    b_block = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)          # [BLOCK_B]
    m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)          # [BLOCK_M]
    n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)          # [BLOCK_N]

    # Accumulator in float32
    acc = tl.zeros((BLOCK_B, BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)                     # [BLOCK_K]

        # Compute pointers for A[b, m, k] and Wt[k, n]
        # A: [B, M, K]
        A_ptrs = A_ptr + b[:, None, None] * A_stride_b + m[None, :, None] * A_stride_m + k[None, None, :] * A_stride_k  # shape [BLOCK_B, BLOCK_M, BLOCK_K]
        A_mask = (b[:, None, None] < B) & (m[None, :, None] < M) & (k[None, None, :] < K)

        # Wt: [K, N]
        Wt_ptrs = Wt_ptr + k[:, None] * Wt_stride_k + n[None, :] * Wt_stride_n  # shape [BLOCK_K, BLOCK_N]
        Wt_mask = (k[:, None] < K) & (n[None, :] < N)

        # Load tiles
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)   # [BLOCK_B, BLOCK_M, BLOCK_K]
        Wt_tile = tl.load(Wt_ptrs, mask=Wt_mask, other=0.0) # [BLOCK_K, BLOCK_N]

        # Accumulate: acc[b, m, n] += sum_k A_tile[b, m, k] * Wt_tile[k, n]
        # Broadcast multiply and reduce over k
        # A_tile: [B, M, K], Wt_tile: [K, N]
        # We need to align k dimension: A_tile[:, :, k] * Wt_tile[k, :]
        # Use einsum-like trick: expand dims and multiply, then sum over last dim
        # acc += sum over K of A_tile[:, :, k] * Wt_tile[k, :]
        for kk in range(BLOCK_K):
            a_vec = A_tile[:, :, kk]  # [B, M]
            w_vec = Wt_tile[kk, :]    # [N]
            acc += a_vec[:, :, None] * w_vec[None, None, :]

    # Store results
    C_ptrs = C_ptr + b[:, None, None] * C_stride_b + m[None, :, None] * C_stride_m + n[None, None, :] * C_stride_n
    C_mask = (b[:, None, None] < B) & (m[None, :, None] < M) & (n[None, None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def split_seqs_kernel(
    src_ptr,    # *processed [B, M, D], M=L_txt + L_img
    out1_ptr,   # *processed_encoder [B, L_txt, D]
    out2_ptr,   # *processed_hidden [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    src_stride_b: tl.int32, src_stride_s: tl.int32, src_stride_d: tl.int32,
    out1_stride_b: tl.int32, out1_stride_s: tl.int32, out1_stride_d: tl.int32,
    out2_stride_b: tl.int32, out2_stride_s: tl.int32, out2_stride_d: tl.int32,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    b_block = tl.program_id(0)
    s_block = tl.program_id(1)
    d_block = tl.program_id(2)

    b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)

    # src base pointer
    src_base = src_ptr + b * src_stride_b

    for i in range(BLOCK_S):
        s = s_offsets[i]
        valid_s = s < (L_txt + L_img)

        # First split: [:, :L_txt, :]
        if s < L_txt:
            src_ptr_s = src_base + s * src_stride_s
            out1_ptr_s = out1_ptr + b * out1_stride_b + s * out1_stride_s
            d_vec = d_offsets
            mask_d = d_vec < D
            vals = tl.load(src_ptr_s + d_vec * src_stride_d, mask=mask_d, other=0.0)
            tl.store(out1_ptr_s + d_vec * out1_stride_d, vals, mask=mask_d & valid_s)

        # Second split: [:, L_txt:, :]
        hs_s = s - L_txt
        src_ptr_hs = src_base + s * src_stride_s
        out2_ptr_hs = out2_ptr + b * out2_stride_b + hs_s * out2_stride_s
        d_vec = d_offsets
        mask_d = d_vec < D
        vals = tl.load(src_ptr_hs + d_vec * src_stride_d, mask=mask_d, other=0.0)
        tl.store(out2_ptr_hs + d_vec * out2_stride_d, vals, mask=mask_d & valid_s)


# ModelNew: entry point, uses Triton kernels exclusively in forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D, "encoder_hidden_states hidden_dim must match hidden_states"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # 1) Concatenate along sequence dimension using Triton
        concatenated = torch.empty((B, L_txt + L_img, D), dtype=hidden_states.dtype, device=hidden_states.device)

        # Ensure we use strides correctly
        ehs = encoder_hidden_states
        hs = hidden_states

        ehs_stride_b, ehs_stride_s, ehs_stride_d = ehs.stride()
        hs_stride_b, hs_stride_s, hs_stride_d = hs.stride()
        dst_stride_b, dst_stride_s, dst_stride_d = concatenated.stride()

        # Launch concat kernel: grid over (B, tiles along sequence, tiles along D)
        # We can choose BLOCK_S and BLOCK_D; 64 works well for many sizes
        grid_concat = (B, triton.cdiv(L_txt + L_img, 64), triton.cdiv(D, 64))
        concat_seqs_kernel[grid_concat](
            ehs, hs, concatenated,
            B, L_txt, L_img, D,
            int(ehs_stride_b), int(ehs_stride_s), int(ehs_stride_d),
            int(hs_stride_b), int(hs_stride_s), int(hs_stride_d),
            int(dst_stride_b), int(dst_stride_s), int(dst_stride_d),
            BLOCK_S=64, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # 2) Linear projection using Triton GEMM: C = concatenated @ process_weight.T
        # A: [B, M, K], Wt: [K, N], C: [B, M, N]
        A = concatenated
        Wt = process_weight.t()  # [K, N]

        M = L_txt + L_img
        N = D
        K = D

        C = torch.empty((B, M, N), dtype=torch.float32, device=A.device)  # compute in float32 for stability

        A_stride_b, A_stride_m, A_stride_k = A.stride()
        Wt_stride_k, Wt_stride_n = Wt.stride()
        C_stride_b, C_stride_m, C_stride_n = C.stride()

        # Launch GEMM kernel with 3D grid
        # BLOCK sizes chosen to be robust; tune if needed
        BLOCK_B = 1
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid_gemm = (triton.cdiv(B, BLOCK_B), triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        gemm_bmn_kernel[grid_gemm](
            A, Wt, C,
            B, M, N, K,
            int(A_stride_b), int(A_stride_m), int(A_stride_k),
            int(Wt_stride_k), int(Wt_stride_n),
            int(C_stride_b), int(C_stride_m), int(C_stride_n),
            BLOCK_B=BLOCK_B, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split outputs using Triton
        processed_encoder = torch.empty((B, L_txt, D), dtype=torch.float32, device=C.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=torch.float32, device=C.device)

        src_stride_b, src_stride_s, src_stride_d = C.stride()
        out1_stride_b, out1_stride_s, out1_stride_d = processed_encoder.stride()
        out2_stride_b, out2_stride_s, out2_stride_d = processed_hidden.stride()

        grid_split = (B, triton.cdiv(L_txt, 64), triton.cdiv(D, 64))
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            int(src_stride_b), int(src_stride_s), int(src_stride_d),
            int(out1_stride_b), int(out1_stride_s), int(out1_stride_d),
            int(out2_stride_b), int(out2_stride_s), int(out2_stride_d),
            BLOCK_S=64, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # Cast back to original dtype if needed
        if processed_encoder.dtype != hidden_states.dtype:
            processed_encoder = processed_encoder.to(hidden_states.dtype)
        if processed_hidden.dtype != hidden_states.dtype:
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
