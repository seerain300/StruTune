import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_row_kernel(
    out_ptr,      # *fp32, [N_rows, K]
    a_ptr,        # *fp32, [N_rows, K] (row-major)
    b_ptr,        # *fp32, [K, K] (row-major, process_weight.T)
    N_rows: tl.int32,
    K: tl.int32,
    BLOCK_K: tl.int32,
):
    # Each program handles one output row
    row_id = tl.program_id(0)
    # Base offsets
    out_row_base = row_id * K
    a_row_base = row_id * K

    # Accumulator for this row
    acc = tl.zeros((K,), dtype=tl.float32)

    # Reduction over K in chunks
    k0 = 0
    while k0 < K:
        # Load B[k0:k0+BLOCK_K, :] -> shape [BLOCK_K, K]
        b_offsets = k0 + tl.arange(0, BLOCK_K)[:, None] * K + tl.arange(0, K)[None, :]
        # Mask for B: rows must be < K, cols valid as we iterate only up to K
        b_mask = (k0 + tl.arange(0, BLOCK_K))[:, None] < K
        b_tile = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)  # [BLOCK_K, K]

        # Load A[row_id, k0:k0+BLOCK_K] -> shape [BLOCK_K]
        a_offsets = a_row_base + (k0 + tl.arange(0, BLOCK_K))
        a_mask = (k0 + tl.arange(0, BLOCK_K)) < K
        a_vec = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)  # [BLOCK_K]

        # Compute partial dot: sum_{i in BLOCK_K} a_vec[i] * b_tile[i, :]
        # Broadcast a_vec over columns
        acc += tl.sum(a_vec[:, None] * b_tile, axis=0)

        k0 += BLOCK_K

    # Store the accumulated row
    out_offsets = out_row_base + tl.arange(0, K)
    tl.store(out_ptr + out_offsets, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          1) Concatenate sequences (PyTorch)
          2) Apply linear projection using Triton GEMM (per-row kernel)
          3) Split outputs
        """
        # Shapes
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate along sequence dimension using PyTorch (simple and reliable)
        concatenated = torch.cat(
            [encoder_hidden_states, hidden_states], dim=1
        )  # [N, L_total, K]
        device = concatenated.device
        dtype = concatenated.dtype

        # 2) Prepare inputs for Triton GEMM: A_rows [N_rows, K], B [K, K]
        N_rows = N * L_total
        # Flatten rows for GEMM
        A_rows = concatenated.reshape(N_rows, K).contiguous().to(torch.float32)  # [N_rows, K]
        B = process_weight.t().contiguous().to(torch.float32)  # [K, K]

        # Allocate output rows buffer (fp32 accumulation)
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Choose reduction chunk size; simple heuristic
        if K >= 1024:
            BLOCK_K = 256
            num_warps = 4
        elif K >= 256:
            BLOCK_K = 128
            num_warps = 4
        else:
            BLOCK_K = 64
            num_warps = 2

        # Launch Triton kernel: one program per row
        grid = (N_rows,)
        _matmul_row_kernel[grid](
            C_rows, A_rows, B,
            N_rows, K, BLOCK_K,
            num_warps=num_warps,
            num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split into encoder and image streams
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Return in original dtype (reference doesn't use bias and keeps dtype)
        if processed.dtype != hidden_states.dtype:
            processed_encoder = processed_encoder.to(hidden_states.dtype)
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden