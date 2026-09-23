import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,           # *const T, shape [N, L_total, K]
    encoder_ptr,       # *const T, shape [N, L_txt, K]
    hidden_ptr,        # *const T, shape [N, L_img, K]
    N, L_txt, L_img, K,
    # strides
    out_stride_n, out_stride_t, out_stride_k,
    enc_stride_n, enc_stride_t, enc_stride_k,
    hid_stride_n, hid_stride_t, hid_stride_k,
):
    # grid: (N, L_total, tiles_along_K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_ktile = tl.program_id(2)

    # K tile
    K_block = 128  # fixed block size for simplicity; mask will handle tails
    k_start = pid_ktile * K_block
    k_offsets = k_start + tl.arange(0, K_block)
    mask_k = k_offsets < K

    # Determine source based on pid_t
    # pid_t in [0, L_txt): encoder, else hidden at (t - L_txt)
    if pid_t < L_txt:
        src_ptr = encoder_ptr + pid_n * enc_stride_n + pid_t * enc_stride_t
    else:
        src_ptr = hidden_ptr + pid_n * hid_stride_n + (pid_t - L_txt) * hid_stride_t

    # out pointer for this (n, t) row, starting at k_start
    out_row_ptr = out_ptr + pid_n * out_stride_n + pid_t * out_stride_t

    # Load values for this K tile and store to out
    vals = tl.load(src_ptr + k_offsets * enc_stride_k, mask=mask_k, other=0.0)
    tl.store(out_row_ptr + k_offsets * out_stride_k, vals, mask=mask_k)


@triton.jit
def _matmul_row_kernel(
    C_ptr,             # *float32, shape [N_rows, K] (output rows)
    A_ptr,             # *float32, shape [N_rows, K] (concatenated rows)
    B_ptr,             # *float32, shape [K, K] (process_weight.T)
    N_rows, K,
    STRIDE_C_ROW, STRIDE_C_COL,
    STRIDE_A_ROW, STRIDE_A_COL,
    STRIDE_B_ROW, STRIDE_B_COL,
    BLOCK_K: tl.constexpr,
):
    # One program per row
    pid = tl.program_id(0)
    # Accumulator for this row
    acc = tl.zeros((K,), dtype=tl.float32)

    # Reduction over K in chunks
    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A_row chunk (vector of length BLOCK_K)
        a_row_chunk = tl.load(A_ptr + pid * STRIDE_A_ROW + k_offsets * STRIDE_A_COL, mask=mask_k, other=0.0)

        # Load B chunk (BLOCK_K x BLOCK_K)
        b_chunk = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
        for kk in range(BLOCK_K):
            k2 = k0 + kk
            mask_kk = k2 < K
            b_col = tl.load(B_ptr + k_offsets * STRIDE_B_ROW + k2 * STRIDE_B_COL, mask=mask_k & mask_kk, other=0.0)
            b_chunk[kk, :] = b_col

        # Accumulate dot: sum_j A[j] * B[j, k]
        # Since a_row_chunk is 1xBLOCK_K and b_chunk is BLOCK_K x BLOCK_K,
        # we need to multiply corresponding elements across kk dimension.
        # Implement per output element k in tile:
        for kk in range(BLOCK_K):
            k2 = k0 + kk
            if k2 < K:
                b_col = b_chunk[kk, :]  # [BLOCK_K]
                # Multiply scalar a_j with b_col and accumulate
                # Note: a_row_chunk is a vector; for each kk, b_col is vector.
                # We need scalar a_j times vector b_col for each j in BLOCK_K.
                # Better approach: compute partial dot by iterating j:
                # Initialize partial as zeros
                partial = tl.zeros((), dtype=tl.float32)
                # Loop over j in BLOCK_K (constexpr), masked by kk position
                # Here a_j = a_row_chunk[kk], but a_row_chunk is vector. Instead, we recompute a_j from A_ptr.
                # To keep things simple and correct, load a_j per kk from A_ptr:
                a_j = tl.load(A_ptr + pid * STRIDE_A_ROW + k2 * STRIDE_A_COL, mask=(k2 < K), other=0.0)
                b_col = tl.load(B_ptr + k_offsets * STRIDE_B_ROW + k2 * STRIDE_B_COL, mask=mask_k, other=0.0)
                partial += a_j * b_col
                acc += partial
        k0 += BLOCK_K

    # Store the row result
    tl.store(C_ptr + pid * STRIDE_C_ROW + tl.arange(0, K) * STRIDE_C_COL, acc, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [N, L_img, K]
        encoder_hidden_states: [N, L_txt, K]
        process_weight: [K, K] (no bias)
        Returns: (processed_encoder: [N, L_txt, K], processed_hidden: [N, L_img, K])
        """
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        device = hidden_states.device

        # Ensure contiguous for simpler stride handling
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight = process_weight.contiguous()

        # 1) Concatenate in Triton: [N, L_total, K]
        L_total = L_txt + L_img
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)  # we'll compute in fp32

        # Strides
        out_stride_n, out_stride_t, out_stride_k = out.stride()
        enc_stride_n, enc_stride_t, enc_stride_k = encoder.stride()
        hid_stride_n, hid_stride_t, hid_stride_k = hidden.stride()

        # Grid: (N, L_total, tiles along K)
        # Choose tiles along K
        K_block = 128 if K >= 128 else 64
        grid = (N, L_total, triton.cdiv(K, K_block))
        _concat_sequences_kernel[grid](
            out, encoder, hidden,
            N, L_txt, L_img, K,
            out_stride_n, out_stride_t, out_stride_k,
            enc_stride_n, enc_stride_t, enc_stride_k,
            hid_stride_n, hid_stride_t, hid_stride_k,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM in Triton: C_rows = out @ weight.T, shape [N * L_total, K]
        A_rows = out.view(N * L_total, K).contiguous()  # [N_rows, K], fp32
        B = weight.t().contiguous()  # [K, K], fp32

        C_rows = torch.empty((N * L_total, K), device=device, dtype=torch.float32)

        # Launch one program per row; reduction in BLOCK_K chunks
        N_rows = N * L_total
        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K,
            C_rows.stride(0), C_rows.stride(1),
            A_rows.stride(0), A_rows.stride(1),
            B.stride(0), B.stride(1),
            BLOCK_K=128 if K >= 128 else 64,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split
        processed = C_rows.view(N, L_total, K)

        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype (match PyTorch behavior)
        orig_dtype = hidden_states.dtype
        if processed_encoder.dtype != orig_dtype:
            processed_encoder = processed_encoder.to(orig_dtype)
        if processed_hidden.dtype != orig_dtype:
            processed_hidden = processed_hidden.to(orig_dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
