import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,    # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,     # *hidden_states [B, L_img, D]
    dst_ptr,    # *output concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
):
    # grid: (B,)
    b = tl.program_id(axis=0)
    for s in range(L_txt + L_img):
        is_text = s < L_txt
        src = ehs_ptr + b * ehs_stride_b + s * ehs_stride_s if is_text else hs_ptr + b * hs_stride_b + (s - L_txt) * hs_stride_s
        dst_line = dst_ptr + b * dst_stride_b + s * dst_stride_s
        for d in range(D):
            val = tl.load(src + d * (ehs_stride_d if is_text else hs_stride_d))
            tl.store(dst_line + d * dst_stride_d, val)


@triton.jit
def split_seqs_kernel(
    src_ptr,    # *processed [B, M, D]
    out1_ptr,   # *processed_encoder [B, L_txt, D]
    out2_ptr,   # *processed_hidden [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    src_stride_b: tl.int32, src_stride_s: tl.int32, src_stride_d: tl.int32,
    out1_stride_b: tl.int32, out1_stride_s: tl.int32, out1_stride_d: tl.int32,
    out2_stride_b: tl.int32, out2_stride_s: tl.int32, out2_stride_d: tl.int32,
):
    # grid: (B,)
    b = tl.program_id(axis=0)
    # copy first L_txt rows
    for s in range(L_txt):
        src_line = src_ptr + b * src_stride_b + s * src_stride_s
        out1_line = out1_ptr + b * out1_stride_b + s * out1_stride_s
        for d in range(D):
            val = tl.load(src_line + d * src_stride_d)
            tl.store(out1_line + d * out1_stride_d, val)
    # copy next L_img rows
    for s in range(L_img):
        src_line = src_ptr + b * src_stride_b + (s + L_txt) * src_stride_s
        out2_line = out2_ptr + b * out2_stride_b + s * out2_stride_s
        for d in range(D):
            val = tl.load(src_line + d * src_stride_d)
            tl.store(out2_line + d * out2_stride_d, val)


@triton.jit
def matmul_seqs_kernel(
    a_ptr,  # *A [B, M, K]
    w_ptr,  # *W [K, N]
    c_ptr,  # *C [B, M, N]
    B: tl.int32,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    a_stride_b: tl.int32, a_stride_m: tl.int32, a_stride_k: tl.int32,
    w_stride_k: tl.int32, w_stride_n: tl.int32,
    c_stride_b: tl.int32, c_stride_m: tl.int32, c_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid over (tiles along B, M, N)
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    b = pid_b  # one batch element per program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # pointers for output C
    c_ptrs = c_ptr + b * c_stride_b + (offs_m[:, None] * c_stride_m) + (offs_n[None, :] * c_stride_n)
    c_mask = (b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K in blocks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + b * a_stride_b + (offs_m[:, None] * a_stride_m) + (offs_k[None, :] * a_stride_k)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        a_tile = a_tile.to(tl.float32)

        # load W^T tile: [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + (offs_k[:, None] * w_stride_k) + (offs_n[None, :] * w_stride_n)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        w_tile = w_tile.to(tl.float32)

        # accumulate
        acc += tl.dot(a_tile, w_tile)

    # store result
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version that performs:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension (Triton).
        2) Linear projection using Triton matmul (A @ process_weight.T).
        3) Splitting results into encoder and image streams (Triton).
        """
        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]
        M = L_txt + L_img

        # 1) Concatenate sequences using Triton
        dst = torch.empty((B, M, D), dtype=hidden_states.dtype, device=hidden_states.device)

        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        dst = dst.contiguous()

        # Strides
        ehs_stride_b, ehs_stride_s, ehs_stride_d = ehs.stride()
        hs_stride_b, hs_stride_s, hs_stride_d = hs.stride()
        dst_stride_b, dst_stride_s, dst_stride_d = dst.stride()

        grid_concat = (B,)
        concat_seqs_kernel[grid_concat](
            ehs, hs, dst,
            B, L_txt, L_img, D,
            int(ehs_stride_b), int(ehs_stride_s), int(ehs_stride_d),
            int(hs_stride_b), int(hs_stride_s), int(hs_stride_d),
            int(dst_stride_b), int(dst_stride_s), int(dst_stride_d),
            num_warps=2, num_stages=2
        )

        # 2) Linear projection using Triton matmul: processed = dst @ process_weight.T
        # process_weight.T is [K, N] = [D, D]
        Wt = process_weight.t().contiguous()  # [D, D]
        processed = torch.empty((B, M, D), dtype=dst.dtype, device=dst.device)

        a = dst.contiguous()  # [B, M, K]
        w = Wt.contiguous()   # [K, N] where K=D, N=D
        c = processed.contiguous()

        a_stride_b, a_stride_m, a_stride_k = a.stride()
        w_stride_k, w_stride_n = w.stride()
        c_stride_b, c_stride_m, c_stride_n = c.stride()

        # Tile sizes — tune if needed
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid_matmul = (triton.cdiv(B, 1), triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))
        matmul_seqs_kernel[grid_matmul](
            a, w, c,
            B, M, D, D,  # K == D
            int(a_stride_b), int(a_stride_m), int(a_stride_k),
            int(w_stride_k), int(w_stride_n),
            int(c_stride_b), int(c_stride_m), int(c_stride_n),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3
        )

        # 3) Split results using Triton
        processed_encoder = torch.empty((B, L_txt, D), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=processed.dtype, device=processed.device)

        processed = processed.contiguous()
        out1_stride_b, out1_stride_s, out1_stride_d = processed_encoder.stride()
        out2_stride_b, out2_stride_s, out2_stride_d = processed_hidden.stride()

        src_stride_b, src_stride_s, src_stride_d = processed.stride()

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            int(src_stride_b), int(src_stride_s), int(src_stride_d),
            int(out1_stride_b), int(out1_stride_s), int(out1_stride_d),
            int(out2_stride_b), int(out2_stride_s), int(out2_stride_d),
            num_warps=2, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
