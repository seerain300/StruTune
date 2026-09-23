import torch
import triton
import triton.language as tl


@triton.jit
def triton_gemm_rowwise_tiles_kernel(
    X_ptr,  # pointer to X of shape [B * M, D]
    W_ptr,  # pointer to W of shape [D, D]
    Y_ptr,  # pointer to Y of shape [B * M, D]
    M, D, B,
    sX_row, sX_col,  # strides for X: sX_row = D, sX_col = 1
    sW0, sW1,        # strides for W: sW0 = D, sW1 = 1
    sY_row, sY_col,  # strides for Y: sY_row = D, sY_col = 1
    BLOCK_K: tl.constexpr,
):
    """
    Compute Y[row] = sum_k X[row, k] * W[k, :] for row in [0, M).
    We launch with 2D grid: axis 0 over tiles of M, axis 1 over batch.
    Each program handles one row tile and one batch.
    """
    # pid_m: tile id along M, pid_b: batch id
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Compute row index within the batch for this program
    row = pid_m * 1  # BLOCK_M=1: each program handles one row (simple and robust)
    # For axis=0 tiling, if you want multiple rows per program, set BLOCK_M > 1 and adjust row = pid_m * BLOCK_M + arange(0, BLOCK_M)
    # Here, we ensure grid is (cdiv(M, 1), B) so row is exactly pid_m.
    # Bounds check (defensive)
    if row >= M:
        return

    # Output pointer for this (batch, row)
    out_row_ptr = Y_ptr + pid_b * sY_row + row * sY_col

    # Accumulator for D columns
    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D

        # Load E_vec: X[row, offs_k]
        x_vec_ptr = X_ptr + pid_b * sX_row + row * sX_col + offs_k * sX_col
        E_vec = tl.load(x_vec_ptr, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # Load W_sub: W[offs_k, :] -> shape [BLOCK_K, D]
        w_sub_ptr = W_ptr + offs_k[:, None] * sW0 + tl.arange(0, D)[None, :] * sW1
        W_sub = tl.load(w_sub_ptr, mask=mask_k[:, None], other=0.0)  # shape [BLOCK_K, D]

        # Contribution: sum over k of E_vec[k] * W_sub[k, :]
        # We reduce along the BLOCK_K axis
        # Note: Triton supports tensor multiply; we reduce by summing along axis 0.
        contrib = tl.sum(W_sub * E_vec[:, None], axis=0)  # shape [D]
        acc += contrib

    # Store acc to output row
    out_ptr = out_row_ptr + tl.arange(0, D) * sY_col
    store_mask = tl.arange(0, D) < D  # always true, but keep for safety
    tl.store(out_ptr, acc, mask=store_mask)


@triton.jit
def expand_to_batch_row_kernel(
    src_ptr,      # pointer to per-row data of length D
    dst_ptr,      # pointer to output row slice of shape [D]
    B, M, D,
    s_src,        # stride of src (1 for row vector)
    s_dst_b, s_dst_m, s_dst_d,  # strides of dst: [B, M, D]
    BLOCK_M: tl.constexpr,
):
    """
    Expand a per-row vector into the corresponding slice of an output tensor of shape [B, M, D].
    We launch with grid (M, B). Each program handles one row and one batch.
    """
    pid_m = tl.program_id(0)  # row index in [0, M)
    pid_b = tl.program_id(1)  # batch index in [0, B)

    # Base pointer for this (batch, row) slice
    dst_base = dst_ptr + pid_b * s_dst_b + pid_m * s_dst_m

    # Load src row
    src_row_ptr = src_ptr + pid_m * s_src  # src is a vector; stride s_src typically 1
    src_vec = tl.load(src_row_ptr)  # shape [D]

    # Store into dst row
    dst_out_ptr = dst_base + tl.arange(0, D) * s_dst_d
    tl.store(dst_out_ptr, src_vec)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          - Compute yA = encoder_hidden_states @ process_weight.T per row, expand to [B, T, D]
          - Compute yB = hidden_states @ process_weight.T per row, expand to [B, I, D]
          - Return (yA_expanded, yB_expanded)
        """
        # Shapes
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        device = hidden_states.device

        # Ensure contiguity and dtype
        dtype = torch.float32
        hs = hidden_states.contiguous().to(dtype)
        ehs = encoder_hidden_states.contiguous().to(dtype)
        W = process_weight.contiguous().to(dtype)  # [D, D]

        # Allocate outputs
        output_E = torch.empty((B, T, D), device=device, dtype=dtype)  # per-batch per-T rows
        output_H = torch.empty((B, I, D), device=device, dtype=dtype)  # per-batch per-I rows

        # 1) Compute yA per row: [T, D]
        # Grid: axis 0 over tiles of M=T, axis 1 over batch B
        grid_yA = (triton.cdiv(T, 1), B)  # one row per program; B axis covers batches
        triton_gemm_rowwise_tiles_kernel[grid_yA](
            ehs.view(-1, D),  # X rows are flattened per batch: total M = B*T rows
            W,                 # W of shape [D, D]
            output_E.view(-1, D),  # Y of shape [B*T, D]
            T, D, B,
            D, 1,               # sX_row=D, sX_col=1
            W.stride(0), W.stride(1),
            T * D, 1,           # sY_row = T*D (stride across rows), sY_col = 1
            BLOCK_K=64,
            num_warps=1,
            num_stages=1,
        )

        # 2) Compute yB per row: [I, D]
        # Grid: axis 0 over tiles of M=I, axis 1 over batch B
        grid_yB = (triton.cdiv(I, 1), B)
        triton_gemm_rowwise_tiles_kernel[grid_yB](
            hs.view(-1, D),    # X rows are flattened per batch: total M = B*I rows
            W,                 # W of shape [D, D]
            output_H.view(-1, D),  # Y of shape [B*I, D]
            I, D, B,
            D, 1,               # sX_row=D, sX_col=1
            W.stride(0), W.stride(1),
            I * D, 1,           # sY_row = I*D (stride across rows), sY_col = 1
            BLOCK_K=64,
            num_warps=1,
            num_stages=1,
        )

        return output_E, output_H