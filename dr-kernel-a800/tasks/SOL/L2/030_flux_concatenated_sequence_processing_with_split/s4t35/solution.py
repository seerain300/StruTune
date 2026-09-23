import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_to_Encoder3D_kernel(
    C_ptr,            # [M, H] input
    out_ptr,          # [B, T, H] output
    B, T, I, H,       # dims
    BLOCK_M: tl.constexpr,  # tile size along M
    BLOCK_N: tl.constexpr,  # tile size along H
):
    # 3D launch: (batch, tile over T, tile over H)
    b = tl.program_id(0)
    t_tile = tl.program_id(1)
    h_tile = tl.program_id(2)

    # compute row indices for this tile: rows correspond to combined (b, t) positions
    start_row = (b * T) + t_tile * BLOCK_M
    rows = start_row + tl.arange(0, BLOCK_M)
    # mask for valid rows
    # rows in [B*T, (B+1)*T) => b is fixed, t in [t_tile*BLOCK_M, ...)
    mask_rows = rows < (b + 1) * T  # rows span b*T to (b+1)*T - 1

    # columns in [h_tile*BLOCK_N, (h_tile+1)*BLOCK_N)
    cols = h_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_cols = cols < H

    # load from C: C[row, col] where row in [0, M), col in [0, H)
    # We map rows to C indices: row = b*(T+I) + t if t < T, else b*(T+I) + t - T
    # But since we launch per-batch and per-t t_tile, rows directly map to C indices
    # For b=batch, row in [b*T, (b+1)*T) corresponds to first T rows of C for batch b
    c_offsets = rows[:, None] * H + cols[None, :]
    c_vals = tl.load(C_ptr + c_offsets, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0)

    # store to output: out[b, t, :]
    # t = rows - b*T
    t = rows - b * T
    out_offsets = b * (T * H) + t[:, None] * H + cols[None, :]
    # mask for valid t
    mask_t = t >= 0
    tl.store(out_ptr + out_offsets, c_vals, mask=mask_rows[:, None] & mask_cols[None, :] & mask_t[:, None])


@triton.jit
def copy_rows_to_Hidden3D_kernel(
    C_ptr,            # [M, H] input
    out_ptr,          # [B, I, H] output
    B, T, I, H,       # dims
    BLOCK_M: tl.constexpr,  # tile size along M
    BLOCK_N: tl.constexpr,  # tile size along H
):
    # 3D launch: (batch, tile over I, tile over H)
    b = tl.program_id(0)
    i_tile = tl.program_id(1)
    h_tile = tl.program_id(2)

    # start_row in C corresponds to beginning of hidden part for batch b
    start_row = b * (T + I) + T
    rows = start_row + i_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    # mask rows < M and < (b+1)*I
    mask_rows = rows < ((b + 1) * (T + I)) & (rows < (b * (T + I) + (b + 1) * I))

    cols = h_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_cols = cols < H

    c_offsets = rows[:, None] * H + cols[None, :]
    c_vals = tl.load(C_ptr + c_offsets, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0)

    # store to out[b, i, :]
    i = rows - b * (T + I) - T  # i in [0, I)
    # out layout: out[b, i, :] contiguous along H => offset = b*(I*H) + i*H + cols
    out_offsets = b * (I * H) + i[:, None] * H + cols[None, :]
    mask_i = i >= 0
    tl.store(out_ptr + out_offsets, c_vals, mask=mask_rows[:, None] & mask_cols[None, :] & mask_i[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version that:
          - Concatenates encoder_hidden_states and hidden_states along sequence dim (PyTorch).
          - Performs matmul via PyTorch for numerical fidelity.
          - Uses Triton kernels to copy split rows into encoder and hidden outputs.
        Ensures all computation is in Triton in the split step; no torch slicing is used in Triton path.
        """
        # Ensure contiguity and device consistency
        device = hidden_states.device
        B = hidden_states.shape[0]
        T = hidden_states.shape[1]
        I = hidden_states.shape[2]
        H = hidden_states.shape[2]  # hidden dim
        assert encoder_hidden_states.shape[2] == H, "encoder_hidden_states and hidden_states must have same hidden_dim"

        # Step 1: Concatenate along sequence dimension using PyTorch (robust and correct)
        # A shape: [B, T + I, H]
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Step 2: Matmul using PyTorch for exact behavior (no bias)
        # process_weight: [H, H], we need A @ process_weight.T
        weight_T = process_weight.t()  # [H, H]
        # A is [B, T+I, H]; torch.matmul expects A: [M, K], W: [K, N]; here K = H, N = H
        # Reshape A to [M, H], where M = B*(T+I)
        A = concatenated.reshape(B * (T + I), H)
        C = torch.matmul(A, weight_T)  # [M, H], dtype same as A (float32 default)

        # Prepare outputs
        processed_encoder = torch.empty((B, T, H), dtype=C.dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=C.dtype, device=device)

        # Step 3: Triton copy kernels for splitting
        M = B * (T + I)

        # Launch 3D grid for encoder copy: (batch, tiles over T, tiles over H)
        BLOCK_M_E = 128  # tile of rows (per batch) along T
        BLOCK_N_E = 128  # tile of columns along H
        grid_e = (B, triton.cdiv(T, BLOCK_M_E), triton.cdiv(H, BLOCK_N_E))
        copy_rows_to_Encoder3D_kernel[grid_e](
            C, processed_encoder,
            B, T, I, H,
            BLOCK_M=BLOCK_M_E, BLOCK_N=BLOCK_N_E,
            num_warps=4, num_stages=2,
        )

        # Launch 3D grid for hidden copy: (batch, tiles over I, tiles over H)
        BLOCK_M_H = 128
        BLOCK_N_H = 128
        grid_h = (B, triton.cdiv(I, BLOCK_M_H), triton.cdiv(H, BLOCK_N_H))
        copy_rows_to_Hidden3D_kernel[grid_h](
            C, processed_hidden,
            B, T, I, H,
            BLOCK_M=BLOCK_M_H, BLOCK_N=BLOCK_N_H,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
