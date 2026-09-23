import torch
import triton
import triton.language as tl

# --------------------------
# Triton kernels
# --------------------------
@triton.jit
def _concat_sequences_kernel(out_ptr, enc_ptr, hid_ptr, N, L_txt, L_img, K,
                             STRIDE_OUT_N, STRIDE_OUT_T, STRIDE_OUT_K,
                             STRIDE_ENC_N, STRIDE_ENC_T, STRIDE_ENC_K,
                             STRIDE_HID_N, STRIDE_HID_T, STRIDE_HID_K):
    # Grid: (N, L_total, tiles along K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute K tile
    BLOCK_K = 128  # tile along the last dimension
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source tensor based on t
    # total sequence length
    L_total = L_txt + L_img
    # out index for t
    # For pid_t < L_txt: use encoder; else use hidden at (pid_t - L_txt)
    use_encoder = pid_t < L_txt

    # Build pointers
    # out[n, t, k] -> out_ptr + n*STRIDE_OUT_N + t*STRIDE_OUT_T + k*STRIDE_OUT_K
    out_row_ptrs = out_ptr + pid_n * STRIDE_OUT_N + pid_t * STRIDE_OUT_T + k_offsets * STRIDE_OUT_K

    if use_encoder:
        src_ptrs = enc_ptr + pid_n * STRIDE_ENC_N + pid_t * STRIDE_ENC_T + k_offsets * STRIDE_ENC_K
    else:
        src_ptrs = hid_ptr + pid_n * STRIDE_HID_N + (pid_t - L_txt) * STRIDE_HID_T + k_offsets * STRIDE_HID_K

    # Load and store with mask
    vals = tl.load(src_ptrs, mask=mask_k, other=0.0)
    tl.store(out_row_ptrs, vals, mask=mask_k)


@triton.jit
def _matmul_batched_rows_kernel(C, A, B, N_rows, K,
                                STRIDE_C_ROW, STRIDE_C_COL,
                                STRIDE_A_ROW, STRIDE_A_COL,
                                STRIDE_B_ROW, STRIDE_B_COL,
                                BLOCK_M: tl.constexpr,  # we'll use 1 here
                                BLOCK_N: tl.constexpr,  # e.g., 128
                                BLOCK_K: tl.constexpr):  # e.g., 128
    # Grid: (rows_tiles, cols_tiles)
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_start = pid_row * BLOCK_M
    col_start = pid_col * BLOCK_N

    # Only one row per program (BLOCK_M=1), masks handle rows >= N_rows
    row_idx = row_start  # single row; mask_row will guard
    mask_row = row_idx < N_rows

    # Output row pointers
    c_row_ptrs = C + row_idx * STRIDE_C_ROW + col_start + tl.arange(0, BLOCK_N) * STRIDE_C_COL
    mask_out = mask_row & (col_start + tl.arange(0, BLOCK_N) < K)

    # Accumulator for this output row segment
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A_row segment: A[row_idx, k_offsets]
        a_ptrs = A + row_idx * STRIDE_A_ROW + k_offsets * STRIDE_A_COL
        a_vals = tl.load(a_ptrs, mask=mask_k & mask_row, other=0.0)  # [BLOCK_K]

        # Load B block: B[k_offsets, col_start:col_start+BLOCK_N]
        b_ptrs = B + k_offsets[:, None] * STRIDE_B_ROW + (col_start + tl.arange(0, BLOCK_N)) * STRIDE_B_COL
        b_mask = mask_k[:, None] & (mask_out[None, :])
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += sum_k a_vals[k] * b_vals[k, :]
        # Implement outer product accumulation
        acc += tl.sum(b_vals * a_vals[:, None], axis=0)

    # Store results
    tl.store(c_row_ptrs, acc, mask=mask_out)


# --------------------------
# ModelNew: Triton-optimized forward
# --------------------------
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim.
        - Apply linear projection via Triton GEMM (A_cat @ process_weight.T).
        - Split back into two streams.
        """
        # Ensure tensors are on CUDA and contiguous
        device = hidden_states.device
        assert device.type == 'cuda', "ModelNew.forward requires CUDA tensors."

        # Shapes
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate in Triton: out [N, L_total, K]
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)
        out = out.contiguous()
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()

        # Strides
        STRIDE_OUT_N, STRIDE_OUT_T, STRIDE_OUT_K = out.stride()
        STRIDE_ENC_N, STRIDE_ENC_T, STRIDE_ENC_K = enc.stride()
        STRIDE_HID_N, STRIDE_HID_T, STRIDE_HID_K = hid.stride()

        # Grid for concatenation: (N, L_total, tiles along K)
        BLOCK_K_concat = 128
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K_concat))
        _concat_sequences_kernel[grid_concat](
            out, enc, hid, N, L_txt, L_img, K,
            STRIDE_OUT_N, STRIDE_OUT_T, STRIDE_OUT_K,
            STRIDE_ENC_N, STRIDE_ENC_T, STRIDE_ENC_K,
            STRIDE_HID_N, STRIDE_HID_T, STRIDE_HID_K,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM in Triton: C_rows = out @ process_weight.T
        # A_rows: flatten out to [N_rows, K], with N_rows = N * L_total
        N_rows = N * L_total
        A_rows = out.reshape(N_rows, K).contiguous()  # [N_rows, K]
        # B: process_weight.T -> [K, K] contiguous
        B = process_weight.t().contiguous()  # [K, K]

        # Allocate C_rows as float32 for stable accumulation
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Tiled GEMM grid
        BLOCK_N = 128  # output columns per program
        BLOCK_K = 128  # reduction chunk
        grid_gemm = (triton.cdiv(N_rows, 1), triton.cdiv(K, BLOCK_N))  # rows tiles = 1, cols tiles = K/128
        _matmul_batched_rows_kernel[grid_gemm](
            C_rows, A_rows, B, N_rows, K,
            C_rows.stride(0), C_rows.stride(1),
            A_rows.stride(0), A_rows.stride(1),
            B.stride(0), B.stride(1),
            BLOCK_M=1, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split into encoder and hidden parts
        processed = C_rows.view(N, L_total, K)

        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast outputs back to original dtype (match PyTorch behavior)
        orig_dtype = hidden_states.dtype
        if processed_encoder.dtype != orig_dtype:
            processed_encoder = processed_encoder.to(orig_dtype)
        if processed_hidden.dtype != orig_dtype:
            processed_hidden = processed_hidden.to(orig_dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
