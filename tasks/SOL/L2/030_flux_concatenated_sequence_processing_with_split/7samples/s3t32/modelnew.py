import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,              # *f32 [N, L_total, K]
    encoder_ptr,          # *f32 [N, L_txt, K]
    hidden_ptr,           # *f32 [N, L_img, K]
    N: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    K: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # Grid: (N, L_total, tiles along K)
    n = tl.program_id(0)
    t = tl.program_id(1)
    tile_k = tl.program_id(2)

    # Compute offsets along K
    k_offsets = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Determine source tensor based on position t
    # t in [0, L_total), where first L_txt rows come from encoder, rest from hidden.
    is_encoder = t < L_txt
    # Base offsets for out, encoder, hidden
    out_row_offset = n * L_txt * K + t * K
    out_col_offset = out_row_offset + k_offsets

    if is_encoder:
        src_row_offset = n * L_txt * K + t * K
        src_col_offset = src_row_offset + k_offsets
        # Load from encoder
        x = tl.load(encoder_ptr + src_col_offset, mask=k_mask, other=0.0)
        # Store to out
        tl.store(out_ptr + out_col_offset, x, mask=k_mask)
    else:
        src_row_offset = n * L_img * K + (t - L_txt) * K
        src_col_offset = src_row_offset + k_offsets
        x = tl.load(hidden_ptr + src_col_offset, mask=k_mask, other=0.0)
        tl.store(out_ptr + out_col_offset, x, mask=k_mask)


@triton.jit
def _matmul_row_kernel(
    C_ptr,   # *f32 [N_rows, K]
    A_ptr,   # *f32 [N_rows, K]
    B_ptr,   # *f32 [K, K]
    N_rows: tl.int32,
    K: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # Each program instance handles one row of A (i.e., one output row in C)
    row_id = tl.program_id(0)
    # Accumulator in fp32
    acc = tl.zeros((K,), dtype=tl.float32)

    # Reduction over K using a while loop
    k_start = 0
    while k_start < K:
        kk = k_start + tl.arange(0, BLOCK_K)
        k_mask = kk < K

        # Load A row segment: A[row_id, kk]
        a = tl.load(A_ptr + row_id * K + kk, mask=k_mask, other=0.0)  # shape [BLOCK_K]
        # Load B segment: B[kk, :]
        b = tl.load(B_ptr + kk * K + tl.arange(0, K), mask=k_mask, other=0.0)  # shape [BLOCK_K, K]
        # Accumulate: acc += a * b (broadcast a over columns)
        acc += tl.sum(a[:, None] * b, axis=0)

        k_start += BLOCK_K

    # Store result
    tl.store(C_ptr + row_id * K + tl.arange(0, K), acc, mask=True)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Applies linear projection (out @ process_weight.T) using a Triton row-wise GEMM kernel.
        - Splits outputs into encoder and hidden streams.

        Inputs:
          hidden_states: [N, L_img, K]
          encoder_hidden_states: [N, L_txt, K]
          process_weight: [K, K]
        Returns:
          (processed_encoder: [N, L_txt, K], processed_hidden: [N, L_img, K])
        """
        # Ensure inputs on same device
        device = hidden_states.device
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # 1) Allocate concatenated tensor [N, L_total, K]
        L_total = L_txt + L_img
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # 2) Launch Triton kernel to fill out with concatenation
        BLOCK_K = 256
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            out, encoder_hidden_states, hidden_states,
            N, L_txt, L_img, K,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # 3) GEMM using Triton: A_rows = out [N_rows, K], B = process_weight.T [K, K]
        N_rows = N * L_total
        A_rows = out.reshape(N_rows, K).contiguous()  # [N_rows, K]
        B = process_weight.t().contiguous()  # [K, K]

        # Output rows buffer for accumulation
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # 4) Launch row-wise GEMM kernel
        BLOCK_K_GEMM = 256 if K >= 256 else 128
        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K,
            BLOCK_K=BLOCK_K_GEMM,
            num_warps=4,
            num_stages=2,
        )

        # 5) Reshape back to [N, L_total, K] and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Return in original dtype if desired (inputs are float32 in typical usage)
        return processed_encoder, processed_hidden