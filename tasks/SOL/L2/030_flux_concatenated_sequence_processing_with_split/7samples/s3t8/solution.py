import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,   # *float, shape [N, L_total, K]
    enc_ptr,   # *float, shape [N, L_txt, K]
    hid_ptr,   # *float, shape [N, L_img, K]
    N, L_txt, L_img, K,
    BLOCK_K: tl.constexpr,
):
    # Grid: (N, L_total, tiles_along_K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute K offsets for this tile
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source tensor based on t
    # t index within the concatenated sequence
    # If pid_t < L_txt -> from encoder, else from hidden (offset by L_txt)
    t = pid_t

    # Base pointers for out, enc, hid
    out_base = out_ptr + pid_n * (L_total * K) + t * K
    enc_base = enc_ptr + pid_n * (L_txt * K) + t * K
    hid_base = hid_ptr + pid_n * (L_img * K) + (t - L_txt) * K

    # Select source: if t < L_txt, use enc_base else use hid_base
    # Triton requires scalar control; use simple branch
    use_enc = t < L_txt
    src_ptr = enc_base if use_enc else hid_base

    # Load and store
    vals = tl.load(src_ptr + k_offsets, mask=mask_k, other=0.0)
    tl.store(out_base + k_offsets, vals, mask=mask_k)


@triton.jit
def _matmul_row_kernel(
    C_ptr,      # *float, shape [N_rows, K]
    A_ptr,      # *float, shape [N_rows, K]
    B_ptr,      # *float, shape [K, K]
    N_rows, K,
    BLOCK_K: tl.constexpr,
):
    # One program per row
    row_id = tl.program_id(0)
    # Output vector for this row
    # C_ptr[row_id, :] is computed as sum over k of A[row_id, k] * B[k, :]
    # We accumulate in float32
    acc = tl.zeros((K,), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[row_id, k_offsets]
        a = tl.load(A_ptr + row_id * K + k_offsets, mask=mask_k, other=0.0).to(tl.float32)
        # Load B[k_offsets, :]
        b = tl.load(B_ptr + k_offsets[:, None] * K + tl.arange(0, K)[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)

        # Accumulate: for each kk in BLOCK_K, add a[kk] * b[kk, :]
        for kk in range(BLOCK_K):
            k_idx = k_start + kk
            if k_idx < K:
                # b[kk, :] is vector of length K (loads above only masked for columns, we need the kk row)
                # Correct b load should be B[k_idx, :] which is contiguous over columns
                # Here we reconstruct that row: b[:, kk] would be better, but since we loaded full block, take slice
                # Simpler: since B is [K, K], B[k_idx, :] = load with k_offsets == k_idx across columns
                # However, we previously loaded a 2D block; to get B[k_idx, :], we can load B[k_idx, col] vector
                # but we already loaded b as [BLOCK_K, K]. To get B[k_idx, :], we must have loaded B[k_idx, :] directly.
                # Fix: load B[k_idx, :] as a vector for this kk
                b_vec = tl.load(B_ptr + k_idx * K + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0).to(tl.float32)
                acc += a[kk] * b_vec
        k_start += BLOCK_K

    # Store acc to C[row_id, :]
    tl.store(C_ptr + row_id * K + tl.arange(0, K), acc, mask=tl.arange(0, K) < K)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,   # [N, L_img, K]
        encoder_hidden_states: torch.Tensor,  # [N, L_txt, K]
        process_weight: torch.Tensor,         # [K, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version that:
          - Concatenates along sequence dimension in Triton
          - Performs GEMM using Triton kernel (for compliance), and returns results
        """
        # Ensure CUDA and dtype
        device = hidden_states.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors"
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate in Triton
        out_cat = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # We need to pass dtypes to Triton kernel. The original run uses float32.
        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()

        BLOCK_K = 128
        grid = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid](
            out_cat, enc, hid, N, L_txt, L_img, K,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM in Triton (for compliance; even if PyTorch matmul is faster, we keep Triton usage)
        N_rows = N * L_total
        A_rows = out_cat.reshape(N_rows, K).contiguous()  # [N_rows, K]
        B = process_weight.t().contiguous()               # [K, K]
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Launch Triton GEMM kernel: one program per row, reduce over K in chunks
        BLOCK_K_GEMM = 128
        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B, N_rows, K,
            BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back and split
        processed = C_rows.view(N, L_total, K)

        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Match original dtypes
        if processed_encoder.dtype != encoder_hidden_states.dtype:
            processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        if processed_hidden.dtype != hidden_states.dtype:
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
