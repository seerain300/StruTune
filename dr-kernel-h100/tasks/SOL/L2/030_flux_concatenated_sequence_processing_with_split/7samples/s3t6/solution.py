import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,         # *float32 [N, L_total, K]
    encoder_ptr,     # *float32 [N, L_txt, K]
    hidden_ptr,      # *float32 [N, L_img, K]
    N, L_txt, L_img, K,
    BLOCK_K: tl.constexpr,
):
    # Grid: (N, L_total, tiles_along_K)
    n = tl.program_id(0)
    t = tl.program_id(1)
    tile_k = tl.program_id(2)

    k_offsets = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source: first L_txt rows from encoder, remaining from hidden
    is_encoder = t < L_txt

    if is_encoder:
        src = encoder_ptr + n * (L_txt * K) + t * K + k_offsets
    else:
        src_i = t - L_txt
        src = hidden_ptr + n * (L_img * K) + src_i * K + k_offsets

    dst = out_ptr + n * (L_total * K) + t * K + k_offsets

    # Copy
    tl.store(dst, tl.load(src, mask=mask_k))


@triton.jit
def _matmul_row_kernel(
    C_row_ptr,       # *float32 [N_rows, K]
    A_row_ptr,       # *float32 [N_rows, K]
    B_ptr,           # *float32 [K, K]
    N_rows, K,
    BLOCK_K: tl.constexpr,
):
    # One row per program: pid_row in [0, N_rows)
    pid_row = tl.program_id(0)
    if pid_row >= N_rows:
        return

    # Accumulator in fp32
    acc = tl.zeros((K,), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    k0 = 0
    while k0 < K:
        kk = k0 + tl.arange(0, BLOCK_K)
        mask_k = kk < K

        # Load A_row chunk
        a = tl.load(A_row_ptr + pid_row * K + kk, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Load B chunk (rows = kk, all columns)
        b_rows = tl.load(B_ptr + kk[:, None] * K + tl.arange(0, K)[None, :], mask=mask_k[:, None], other=0.0)  # [BLOCK_K, K]
        # Outer product accumulation: acc += a[j] * b_rows[j, :]
        acc += tl.sum(a[:, None] * b_rows, axis=0)
        k0 += BLOCK_K

    # Store result
    tl.store(C_row_ptr + pid_row * K + tl.arange(0, K), acc, mask=(True))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates [N, L_txt, K] and [N, L_img, K] along sequence dim to [N, L_total, K].
        - Computes processed = concatenated @ process_weight.T using Triton GEMM.
        - Splits back into two streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA device for Triton."
        device = hidden_states.device
        N = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate along sequence dimension using Triton
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)
        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()

        BLOCK_K = 128  # tile along K for copy
        grid = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid](
            out, enc, hid, N, L_txt, L_img, K,
            BLOCK_K=BLOCK_K,
        )

        # 2) GEMM: per-row Triton kernel, accumulate in float32
        N_rows = N * L_total
        A_rows = out.reshape(N_rows, K).contiguous()  # [N_rows, K]
        B = process_weight.t().contiguous()          # [K, K]

        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Choose BLOCK_K for reduction; use 128 for typical K, while loop will handle tails
        BLOCK_K_GEMM = 128
        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B, N_rows, K,
            BLOCK_K=BLOCK_K_GEMM,
        )

        # 3) Reshape back to [N, L_total, K] and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype if needed (process_weight is typically float32 in these tasks)
        if processed_encoder.dtype != encoder_hidden_states.dtype:
            processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        if processed_hidden.dtype != hidden_states.dtype:
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
