import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(out_ptr, enc_ptr, hid_ptr, N, L_txt, L_img, K, BLOCK_K: tl.constexpr):
    # Grid: (N, L_total, tiles along K)
    n = tl.program_id(0)
    t = tl.program_id(1)
    tile_k = tl.program_id(2)

    # Compute offsets along K
    k_offsets = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source
    # If t < L_txt, read from encoder; else read from hidden at index t - L_txt
    # Note: t is a runtime integer; branch for selection
    is_encoder = t < L_txt
    if is_encoder:
        # Load from encoder[n, t, :]
        enc_row_ptr = enc_ptr + n * (L_txt * K) + t * K + k_offsets
        vals = tl.load(enc_row_ptr, mask=mask_k, other=0.0)
        # Store to out[n, t, :]
        out_row_ptr = out_ptr + n * (L_txt + L_img) * K + t * K + k_offsets
        tl.store(out_row_ptr, vals, mask=mask_k)
    else:
        # Load from hidden[n, t - L_txt, :]
        src_t = t - L_txt
        hid_row_ptr = hid_ptr + n * (L_img * K) + src_t * K + k_offsets
        vals = tl.load(hid_row_ptr, mask=mask_k, other=0.0)
        # Store to out[n, t, :]
        out_row_ptr = out_ptr + n * (L_txt + L_img) * K + t * K + k_offsets
        tl.store(out_row_ptr, vals, mask=mask_k)


@triton.jit
def _matmul_tiled_kernel(C_ptr, A_ptr, B_ptr, N_rows, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Each program instance computes one BLOCK_N-sized chunk of columns for a BLOCK_M-sized chunk of rows.
    # Here we set BLOCK_M = 1 so each instance handles one row vector.
    row_block_id = tl.program_id(0)
    col_block_id = tl.program_id(1)

    # Compute row indices this program handles
    rows = row_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < N_rows  # for BLOCK_M > 1; we keep BLOCK_M=1 to simplify

    # Compute column indices (output columns) this program handles
    cols = col_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for kk in range(0, K, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A rows x reduction chunk: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + rows[:, None] * K + k_offsets[None, :]
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load B reduction chunk x cols: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * K + cols[None, :]
        b = tl.load(b_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)

    # Store results into C
    c_ptrs = C_ptr + rows[:, None] * K + cols[None, :]
    store_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension
        2) Apply linear projection via GEMM
        3) Split back into separate encoder and image streams
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2
        N = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == N and encoder_hidden_states.shape[2] == K
        assert process_weight.shape[0] == K and process_weight.shape[1] == K

        device = hidden_states.device
        dtype = hidden_states.dtype

        L_txt = encoder_hidden_states.shape[1]
        L_total = L_txt + L_img

        # 1) Concatenate sequences along sequence dimension using Triton
        # Allocate output concatenated tensor
        concatenated = torch.empty((N, L_total, K), device=device, dtype=dtype)

        # Choose BLOCK_K for K-tiles in concatenation (e.g., 128)
        BLOCK_K = 128
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            concatenated, encoder_hidden_states, hidden_states,
            N, L_txt, L_img, K,
            BLOCK_K=BLOCK_K,
        )

        # 2) GEMM: A_rows = concatenated.view(N_rows, K), B = process_weight.T (K, K)
        # Flatten rows and make contiguous
        A_rows = concatenated.reshape(N * L_total, K).contiguous()
        B = process_weight.t().contiguous()  # [K, K]

        # Output buffer for rows (fp32 accumulation)
        C_rows = torch.empty((N * L_total, K), device=device, dtype=torch.float32)

        # Tiling parameters: per-row GEMM (BLOCK_M=1) with column tiles
        # Choose BLOCK_N and BLOCK_K based on K
        if K >= 256:
            BLOCK_N = 128
            BLOCK_K = 128
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_N = 64
            BLOCK_K = 64
            num_warps = 4
            num_stages = 2

        grid_gemm = (triton.cdiv(N * L_total, 1), triton.cdiv(K, BLOCK_N))
        _matmul_tiled_kernel[grid_gemm](
            C_rows, A_rows, B,
            N * L_total, K,
            BLOCK_M=1, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )

        # Cast back to original dtype if needed
        if C_rows.dtype != dtype:
            C_rows = C_rows.to(dtype)

        # 3) Reshape back to [N, L_total, K] and split into two streams
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden