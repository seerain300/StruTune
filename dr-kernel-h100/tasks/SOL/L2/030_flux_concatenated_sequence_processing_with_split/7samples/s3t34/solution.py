import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,           # *float32, shape [N, L_total, K]
    encoder_ptr,       # *float32, shape [N, L_txt, K]
    hidden_ptr,        # *float32, shape [N, L_img, K]
    N: tl.constexpr,   # batch size
    L_txt: tl.constexpr,  # text sequence length
    L_img: tl.constexpr,  # image sequence length
    K: tl.constexpr,   # hidden dim
    BLOCK_K: tl.constexpr,  # tile along K
):
    # Grid: (N, L_total, tiles along K)
    n = tl.program_id(0)
    t = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source: t < L_txt -> encoder, else -> hidden (offset t - L_txt in hidden)
    # Base pointers for out, encoder, hidden at (n, :, :)
    out_base = out_ptr + n * L_txt * K + t * K
    if t < L_txt:
        src_base = encoder_ptr + n * L_txt * K + t * K
    else:
        src_base = hidden_ptr + n * L_img * K + (t - L_txt) * K

    # Load and store
    src = tl.load(src_base + k_offsets, mask=mask_k, other=0.0)
    tl.store(out_base + k_offsets, src, mask=mask_k)


@triton.jit
def _matmul_row_kernel(
    C_ptr,             # *float32, shape [N_rows, K], output rows
    A_ptr,             # *float32, shape [N_rows, K], input rows (concatenated)
    B_ptr,             # *float32, shape [K, K], process_weight.T
    N_rows: tl.constexpr,  # total rows = N * (L_txt + L_img)
    K: tl.constexpr,        # hidden dim
    BLOCK_K: tl.constexpr,  # reduction tile
):
    # Each program handles one row
    row = tl.program_id(0)
    if row >= N_rows:
        return

    # Accumulator for this row
    acc = tl.zeros((K,), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    # B_ptr is [K, K], A_ptr row is [K], we want sum_j A[row, j] * B[j, :]
    j = 0
    while j < K:
        b_cols = j + tl.arange(0, BLOCK_K)
        a_cols = j + tl.arange(0, BLOCK_K)

        mask_b = b_cols < K
        mask_a = a_cols < K

        # Load A[row, j:j+BLOCK_K] -> vector of length BLOCK_K
        a_vec = tl.load(A_ptr + row * K + a_cols, mask=mask_a, other=0.0)

        # Load B[j:j+BLOCK_K, :] -> [BLOCK_K, K]
        b_mat = tl.load(B_ptr + b_cols[:, None] * K + tl.arange(0, K), mask=mask_b[:, None], other=0.0)

        # Multiply and accumulate: [1, BLOCK_K] * [BLOCK_K, K] -> [1, K], then [K]
        # acc += sum over BLOCK_K of a_vec * b_mat[:, :]
        acc += tl.sum(a_vec[:, None] * b_mat, axis=0)

        j += BLOCK_K

    # Store result
    tl.store(C_ptr + row * K + tl.arange(0, K), acc, mask=tl.arange(0, K) < K)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # Ensure contiguous and float32 for Triton
        device = hidden_states.device
        hidden = hidden_states.contiguous().to(torch.float32)
        encoder = encoder_hidden_states.contiguous().to(torch.float32)
        weight = process_weight.contiguous().to(torch.float32)  # [K, K]

        # 1) Concatenate into [N, L_total, K] using Triton
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        BLOCK_K = 128  # tile along K
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            out, encoder, hidden,
            N, L_txt, L_img, K, BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Flatten to rows and GEMM: out @ weight.T -> [N, L_total, K]
        # A_rows: [N_rows, K], N_rows = N * L_total
        A_rows = out.view(N * L_total, K).contiguous()
        B = weight.t().contiguous()  # [K, K]
        C_rows = torch.empty((N * L_total, K), device=device, dtype=torch.float32)

        # GEMM per row: A_rows @ B
        BLOCK_K_GEMM = 256 if K >= 256 else 128
        grid_gemm = (N * L_total,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N * L_total, K, BLOCK_K_GEMM,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Return in original dtype
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
