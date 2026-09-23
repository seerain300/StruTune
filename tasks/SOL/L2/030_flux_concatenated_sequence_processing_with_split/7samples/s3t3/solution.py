import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,        # *float32, output [N, L_total, K]
    enc_ptr,        # *float32, encoder_hidden_states [N, L_txt, K]
    h_ptr,          # *float32, hidden_states [N, L_img, K]
    N: tl.constexpr,        # batch size
    L_txt: tl.constexpr,    # text sequence length
    L_img: tl.constexpr,    # image sequence length
    K: tl.constexpr,        # hidden dim
    BLOCK_K: tl.constexpr,  # tile size along K
):
    # Grid: (N, L_total, ceil_div(K, BLOCK_K))
    n = tl.program_id(0)  # batch index
    t = tl.program_id(1)  # concatenated sequence index (0..L_txt+L_img-1)
    tile_k = tl.program_id(2)  # tile index along K

    # compute k offsets
    k_offsets = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # decide source tensor based on t
    # If t < L_txt: src = enc_ptr[n, t, :]
    # Else: src = h_ptr[n, t - L_txt, :]
    src_sel = t < L_txt
    src_offset = tl.where(src_sel, enc_ptr + n * (L_txt * K) + t * K, h_ptr + n * (L_img * K) + (t - L_txt) * K)

    # load
    src_vals = tl.load(src_offset + k_offsets, mask=mask_k, other=0.0)

    # out pointer offset for [n, t, :]
    out_offset = n * (L_txt + L_img) * K + t * K
    tl.store(out_ptr + out_offset + k_offsets, src_vals, mask=mask_k)


@triton.jit
def _matmul_batched_rows_kernel(
    C_ptr,          # *float32, output rows [N_rows, K]
    A_rows_ptr,     # *float32, concatenated rows [N_rows, K]
    B_ptr,          # *float32, process_weight.T [K, K]
    N_rows: tl.constexpr,  # total rows = N * (L_txt + L_img)
    K: tl.constexpr,       # hidden dim
    BLOCK_M: tl.constexpr, # number of rows processed per program (we use 1)
    BLOCK_N: tl.constexpr, # output columns tile
    BLOCK_K: tl.constexpr, # reduction tile
):
    # Each program handles a tile of BLOCK_N output columns for one row.
    row_id = tl.program_id(0)  # in [0, N_rows)
    col_block = tl.program_id(1)  # in [0, ceil_div(K, BLOCK_N))
    tile_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = tile_n < K

    # Accumulator for this row tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Reduction loop over K in chunks of BLOCK_K
    for k0 in tl.static_range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[row_id, k_offsets]
        A_row_off = row_id * K
        A_vals = tl.load(A_rows_ptr + A_row_off + k_offsets, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # Load B[k_offsets, tile_n]
        B_offsets = k_offsets[:, None] * K + tile_n[None, :]  # shape [BLOCK_K, BLOCK_N]
        B_vals = tl.load(B_ptr + B_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.sum(A_vals[:, None] * B_vals, axis=0)  # reduce over K-chunk

    # Store results
    C_row_off = row_id * K
    tl.store(C_ptr + C_row_off + tile_n, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run() function.
        - Concatenation is done by a Triton kernel.
        - The linear projection (matmul) is done by a Triton GEMM kernel.
        - Returns (processed_encoder, processed_hidden).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 for consistency"

        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == K and process_weight.shape[0] == K and process_weight.shape[1] == K, "Mismatched hidden_dim/weight shape"

        L_total = L_txt + L_img

        # 1) Concatenate sequences along sequence dimension using Triton
        # out_cat: [N, L_total, K], contiguous
        out_cat = torch.empty((N, L_total, K), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch concatenation kernel: grid = (N, L_total, ceil_div(K, BLOCK_K))
        BLOCK_K = 128  # tile along K
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            out_cat, encoder_hidden_states, hidden_states,
            N, L_txt, L_img, K, BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Prepare B as process_weight.T to match A_rows [N_rows, K] @ B [K, K] = C [N_rows, K]
        B = process_weight.t().contiguous()  # [K, K]

        # 3) Flatten concatenated rows into [N_rows, K]
        A_rows = out_cat.view(-1, K)  # N_rows = N * L_total
        N_rows = A_rows.shape[0]

        # Output buffer for GEMM: [N_rows, K], contiguous
        C_rows = torch.empty((N_rows, K), device=hidden_states.device, dtype=hidden_states.dtype)

        # 4) Launch Triton GEMM kernel
        # Choose tiling parameters. For robustness, use BLOCK_M=1 (one row per program), BLOCK_N=128 or 256, BLOCK_K=64/128.
        BLOCK_M = 1
        BLOCK_N = 128
        BLOCK_K = 128

        grid_gemm = (N_rows, triton.cdiv(K, BLOCK_N))
        _matmul_batched_rows_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K,
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 5) Reshape back to [N, L_total, K] and split into two streams
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
