import torch
import triton
import triton.language as tl

# Triton kernel: compute per-row dot products for a batch of sequences.
# X_ptr: pointer to [B*ROWS, D] rows laid out contiguously (we pass hs_view and ehs_view as 1D).
# W_ptr: pointer to [D, D]
# Y_ptr: pointer to [B*ROWS, D]
# Arguments:
#   B: number of batches
#   ROWS: number of rows in X (T or I)
#   D: hidden dimension
#   sX: stride of X (in elements), typically 1 for 1D view
#   sW0, sW1: strides of W (row and col), typically (D, 1) for [D, D]
#   sY: stride of Y (in elements), typically 1 for 1D view
#   BLOCK_K: tile size along K (hidden_dim), e.g., 64
@triton.jit
def triton_gemm_rowwise_tiles_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, ROWS, D,
    sX, sW0, sW1, sY,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: axis 0 over tiles along ROWS, axis 1 over batch
    pid_row = tl.program_id(0)  # tile index along rows
    b = tl.program_id(1)        # batch index

    # Compute base row for this program
    row_start = pid_row * BLOCK_K
    rows = row_start + tl.arange(0, BLOCK_K)
    # Total rows in this batch is ROWS; we will loop over tiles
    total_rows = ROWS

    # Accumulator for this tile of rows
    acc = tl.zeros((BLOCK_K, D), dtype=tl.float32)

    # Loop over hidden_dim in tiles of BLOCK_K
    for k in range(0, D, BLOCK_K):
        k_idx = k + tl.arange(0, BLOCK_K)  # current tile in D
        # Masks: valid rows and valid k columns
        mask_rows = rows < total_rows
        mask_k = k_idx < D

        # Load E_vec for the rows in this tile: shape (BLOCK_K,)
        x_ptrs = X_ptr + b * total_rows + rows * sX
        e_vec = tl.load(x_ptrs, mask=mask_rows, other=0.0)  # [BLOCK_K]

        # Load W_sub: a [BLOCK_K, D] submatrix of W for columns k_idx
        w_ptrs = W_ptr + k_idx[:, None] * sW0 + tl.arange(0, D)[None, :] * sW1
        # Broadcast mask_k across rows: shape (BLOCK_K, D)
        w_mask = mask_k[None, :] & (rows[:, None] < total_rows)
        w_sub = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, D]

        # Accumulate: acc += e_vec[:, None] * w_sub
        # e_vec[:, None] broadcasts to [BLOCK_K, 1], then multiply with [BLOCK_K, D]
        acc += e_vec[:, None] * w_sub

    # Store the accumulated result into Y at rows 'rows'
    y_ptrs = Y_ptr + b * total_rows + rows * sY
    # Store only valid rows
    tl.store(y_ptrs, acc, mask=rows < total_rows)

# Triton kernel: expand a per-row vector (Y_ptr) into batched output (Out_ptr) for a given batch b.
# Y_ptr: pointer to per-row results of shape [ROWS, D] (we pass per-stream results for T and I)
# Out_ptr: pointer to output of shape [B, ROWS, D]
# Arguments:
#   Y_ptr: pointer to per-row results
#   Out_ptr: pointer to batched output
#   B, ROWS, D: shapes
#   sY: stride of Y (in elements), typically 1 for 1D view
#   sO_b, sO_r, sO_d: strides of Out (batch, row, dim)
@triton.jit
def expand_to_batch_row_kernel(
    Y_ptr, Out_ptr,
    B, ROWS, D,
    sY, sO_b, sO_r, sO_d,
):
    b = tl.program_id(0)  # batch index
    r = tl.program_id(1)  # row index within ROWS

    # Load the per-row vector from Y: [D]
    y_ptrs = Y_ptr + r * sY
    y_vec = tl.load(y_ptrs)  # [D]

    # Store into Out at [b, r, :]
    out_ptrs = Out_ptr + b * sO_b + r * sO_r + tl.arange(0, D) * sO_d
    tl.store(out_ptrs, y_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states: [B, I, D]
        encoder_hidden_states: [B, T, D]
        process_weight: [D, D]
        Returns: (processed_encoder: [B, T, D], processed_hidden: [B, I, D])
        """
        # Ensure CUDA and float32
        device = hidden_states.device
        assert device.type == 'cuda', "ModelNew requires CUDA tensors"
        dtype = torch.float32
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.to(dtype)
        if encoder_hidden_states.dtype != dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype)
        if process_weight.dtype != dtype:
            process_weight = process_weight.to(dtype)

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, D), "encoder_hidden_states shape must be [B, T, D]"
        assert process_weight.shape == (D, D), "process_weight shape must be [D, D]"

        # We will compute per-row projections using Triton:
        # yA: [T, D] = encoder_hidden_states @ process_weight.T
        # yB: [I, D] = hidden_states @ process_weight.T

        # Create 1D views of per-row inputs to feed into Triton GEMM kernel
        # Note: PyTorch does not provide cat/stack here; we rely on Triton for all compute.
        # Prepare X for encoder_hidden_states: [B*T, D]
        ehs_view = encoder_hidden_states.view(-1, D)  # [B*T, D]
        # Prepare W: [D, D]
        w = process_weight  # keep as [D, D], contiguous
        w = w.contiguous()
        # Allocate yA buffer: [B*T, D]
        yA = torch.empty((B * T, D), device=device, dtype=dtype)

        # Launch Triton GEMM for encoder hidden states
        BLOCK_K = 64  # tile along D
        grid_e = (triton.cdiv(T, BLOCK_K), B)
        triton_gemm_rowwise_tiles_kernel[grid_e](
            ehs_view, w, yA,
            B, T, D,
            1,  # sX
            w.stride(0), w.stride(1),
            1,  # sY
            BLOCK_K=BLOCK_K,
            num_warps=2, num_stages=2,
        )

        # Prepare X for hidden states: [B*I, D]
        hs_view = hidden_states.view(-1, D)  # [B*I, D]
        # Allocate yB buffer: [B*I, D]
        yB = torch.empty((B * I, D), device=device, dtype=dtype)

        # Launch Triton GEMM for image hidden states
        grid_h = (triton.cdiv(I, BLOCK_K), B)
        triton_gemm_rowwise_tiles_kernel[grid_h](
            hs_view, w, yB,
            B, I, D,
            1,  # sX
            w.stride(0), w.stride(1),
            1,  # sY
            BLOCK_K=BLOCK_K,
            num_warps=2, num_stages=2,
        )

        # Now expand per-row results into batched outputs using Triton kernels
        # Output_E: [B, T, D]
        output_E = torch.empty((B, T, D), device=device, dtype=dtype)
        # Launch expand for each batch b
        for b in range(B):
            # 2D grid over rows T and batch b
            grid_b = (T, B)
            expand_to_batch_row_kernel[grid_b](
                yA[b * T:(b + 1) * T], output_E[b],
                B, T, D,
                1,  # sY
                output_E.stride(0), output_E.stride(1), output_E.stride(2),
                num_warps=1, num_stages=1,
            )

        # Output_H: [B, I, D]
        output_H = torch.empty((B, I, D), device=device, dtype=dtype)
        for b in range(B):
            grid_b_h = (I, B)
            expand_to_batch_row_kernel[grid_b_h](
                yB[b * I:(b + 1) * I], output_H[b],
                B, I, D,
                1,  # sY
                output_H.stride(0), output_H.stride(1), output_H.stride(2),
                num_warps=1, num_stages=1,
            )

        return output_E, output_H