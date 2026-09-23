import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,        # *f32, [N, L_total, K]
    encoder_ptr,    # *f32, [N, L_txt, K]
    hidden_ptr,     # *f32, [N, L_img, K]
    N: tl.int32, L_txt: tl.int32, L_img: tl.int32, K: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # Grid: (N, L_total, tiles along K)
    n = tl.program_id(0)
    t = tl.program_id(1)
    k_block = tl.program_id(2)

    # Compute K offsets for this tile
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Select source based on t: if t < L_txt, take encoder; else take hidden
    # Note: t ranges from 0 to L_total-1
    use_encoder = t < L_txt

    # Row indices
    # Hidden row index corresponds to t - L_txt when t >= L_txt
    row_hidden = t - L_txt

    # Compute pointers
    # out[n, t, k_offsets]
    out_row_ptr = out_ptr + n * (L_txt + L_img) * K + t * K + k_offsets
    # encoder[n, t, k_offsets]
    enc_row_ptr = encoder_ptr + n * L_txt * K + t * K + k_offsets
    # hidden[n, row_hidden, k_offsets]
    hid_row_ptr = hidden_ptr + n * L_img * K + row_hidden * K + k_offsets

    # Load with masks
    val = tl.zeros([BLOCK_K], dtype=tl.float32)
    if use_encoder:
        val = tl.load(enc_row_ptr, mask=mask_k, other=0.0)
    else:
        val = tl.load(hid_row_ptr, mask=mask_k, other=0.0)

    # Store to out
    tl.store(out_row_ptr, val, mask=mask_k)


@triton.jit
def _matmul_tiled_kernel(
    C_ptr,          # *f32, [N_rows, K]
    A_ptr,          # *f32, [N_rows, K] (each row is one sequence position)
    B_ptr,          # *f32, [K, K]
    N_rows: tl.int32, K: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program computes a tile of C of shape (BLOCK_M, BLOCK_N)
    pid_m = tl.program_id(0)  # tile index along rows (N_rows)
    pid_n = tl.program_id(1)  # tile index along output columns (K)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)       # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)       # [BLOCK_N]
    k_offsets = tl.arange(0, BLOCK_K)                         # [BLOCK_K]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        k_ids = k + k_offsets  # [BLOCK_K]
        # Load A tile: shape [BLOCK_M, BLOCK_K]
        A_tile_ptrs = A_ptr + m_offsets[:, None] * K + k_ids[None, :]
        A_mask = (m_offsets[:, None] < N_rows) & (k_ids[None, :] < K)
        A_tile = tl.load(A_tile_ptrs, mask=A_mask, other=0.0)

        # Load B tile: shape [BLOCK_K, BLOCK_N]
        B_tile_ptrs = B_ptr + k_ids[:, None] * K + n_offsets[None, :]
        B_mask = (k_ids[:, None] < K) & (n_offsets[None, :] < K)
        B_tile = tl.load(B_tile_ptrs, mask=B_mask, other=0.0)

        # acc += A_tile @ B_tile
        # A_tile: [BLOCK_M, BLOCK_K], B_tile: [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(A_tile, B_tile)

    # Write back results to C
    C_ptrs = C_ptr + m_offsets[:, None] * K + n_offsets[None, :]
    C_mask = (m_offsets[:, None] < N_rows) & (n_offsets[None, :] < K)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Extract shapes
        N = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        K = hidden_states.shape[2]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure contiguous
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight = process_weight.contiguous()

        # 1) Concatenate along sequence dimension using Triton
        L_total = L_txt + L_img
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)  # we'll cast back later

        # Triton grid for concatenation: (N, L_total, tiles along K)
        BLOCK_K_concat = 128
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K_concat))
        _concat_sequences_kernel[grid_concat](
            out, encoder, hidden,
            N, L_txt, L_img, K,
            BLOCK_K=BLOCK_K_concat,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: out @ weight.T using Triton tiled matmul
        # A: [N_rows, K], where N_rows = N * L_total
        A = out.view(N * L_total, K).contiguous()  # [N_rows, K], float32
        B = weight.t().contiguous()               # [K, K], float32
        C_rows = torch.empty((N * L_total, K), device=device, dtype=torch.float32)

        # Tiling parameters for GEMM. These are robust defaults.
        BLOCK_M = 16   # rows per program tile (handled in grid by multiple pid_m)
        BLOCK_N = 128  # output columns per tile
        BLOCK_K = 64   # reduction chunk
        grid_gemm = (triton.cdiv(N * L_total, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_tiled_kernel[grid_gemm](
            C_rows, A, B,
            N * L_total, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split into encoder and image streams
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype
        processed_encoder = processed_encoder.to(dtype)
        processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden