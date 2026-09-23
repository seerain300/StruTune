import torch
import triton
import triton.language as tl


@triton.jit
def triton_gemv_rowwise_kernel(
    X_ptr,        # *fp32, pointer to input X [M, D]
    W_ptr,        # *fp32, pointer to weight W [D, D]
    Y_ptr,        # *fp32, pointer to output Y [M, D]
    M: tl.constexpr,      # number of rows in X (sequence length)
    D: tl.constexpr,      # hidden_dim (columns)
    sX_b, sX_m, sX_d,     # strides for X: stride on batch, row, col
    sW0, sW1,             # strides for W: row, col (we use row-major [D, D])
    sY_m, sY_d,           # strides for Y: row, col
    BLOCK_K: tl.constexpr = 64,
):
    # Each program handles one row m of X
    pid_m = tl.program_id(0)
    # Accumulator vector of length D
    acc = tl.zeros([D], dtype=tl.float32)
    # Loop over K in tiles
    for k0 in range(0, D, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < D
        # Load X[m, k] vector (one row)
        # X is [M, D] with strides (sX_b, sX_m, sX_d). We assume M==1 per batch row, but here M is total rows.
        # We load the row m across D columns:
        x_vals = tl.load(
            X_ptr + pid_m * sX_m + k_idx * sX_d,
            mask=mask_k,
            other=0.0,
        )
        # Load corresponding slice of W[:, k] for k in [k0, k0+BLOCK_K)
        w_vals = tl.load(
            W_ptr + k_idx * sW0,  # row index is 0..D-1, col indices are k_idx
            mask=mask_k,
            other=0.0,
        )
        # Accumulate dot product for this tile
        acc += tl.sum(x_vals[:, None] * w_vals[None, :], axis=0)
    # Store result to Y[m, :]
    tl.store(Y_ptr + pid_m * sY_m + tl.arange(0, D) * sY_d, acc, mask=True)


@triton.jit
def triton_copy_expand_to_batch_encoder_kernel(
    Y_ptr,        # *fp32, input Y [T, D]
    Out_ptr,      # *fp32, output [B, T, D]
    T: tl.constexpr,       # text_seq_len
    D: tl.constexpr,       # hidden_dim
    B: tl.constexpr,       # batch size
    sY_t, sY_d,            # strides for Y: row, col
    sO_b, sO_t, sO_d,      # strides for Out: batch, row, col
):
    # For each batch b, copy Y[:T, :] into Out[b, :T, :]
    b = tl.program_id(0)  # grid = (B,)
    for t in range(0, T):
        y_row = tl.load(Y_ptr + t * sY_t + tl.arange(0, D) * sY_d)
        tl.store(Out_ptr + b * sO_b + t * sO_t + tl.arange(0, D) * sO_d, y_row)


@triton.jit
def triton_copy_expand_to_batch_hidden_kernel(
    Y_ptr,        # *fp32, input Y [I, D]
    Out_ptr,      # *fp32, output [B, I, D]
    I: tl.constexpr,       # img_seq_len
    D: tl.constexpr,       # hidden_dim
    B: tl.constexpr,       # batch size
    sY_i, sY_d,            # strides for Y: row, col
    sO_b, sO_i, sO_d,      # strides for Out: batch, row, col
):
    # For each batch b, copy Y[:I, :] into Out[b, :I, :]
    b = tl.program_id(0)  # grid = (B,)
    for i in range(0, I):
        y_row = tl.load(Y_ptr + i * sY_i + tl.arange(0, D) * sY_d)
        tl.store(Out_ptr + b * sO_b + i * sO_i + tl.arange(0, D) * sO_d, y_row)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure contiguity
        hs = hidden_states.contiguous()  # [B, I, D]
        ehs = encoder_hidden_states.contiguous()  # [B, T, D]
        W = process_weight.contiguous()  # [D, D]

        B_hs, I, D = hs.shape
        B_ehs, T, D = ehs.shape
        assert B_hs == B_ehs, "Batch sizes of hidden_states and encoder_hidden_states must match"
        B = B_hs

        # Compute yA = encoder_hidden_states @ W.T -> [B, T, D]
        # Flatten batch into rows for simplicity: treat each batch as independent
        # We will run the GEMV per batch by launching grid=(T,) and looping across D.
        # Prepare output buffer [B, T, D]
        yA = torch.empty((B, T, D), dtype=torch.float32, device=hs.device)
        # Launch GEMV for each batch. Note: Triton does not accept dynamic grid based on B; we loop inside or handle per-batch.
        # To keep it simple, we launch per-batch via Python loop since Triton requires grid as tuple.
        for b in range(B):
            X = ehs[b]  # [T, D]
            Y = yA[b]   # [T, D]
            # Strides
            sX_b, sX_m, sX_d = X.stride()  # X is [T, D]; sX_b is 0 for contiguous
            sW0, sW1 = W.stride()          # W is [D, D]
            sY_m, sY_d = Y.stride()
            # Grid: one program per row (T)
            grid = (T,)
            triton_gemv_rowwise_kernel[grid](
                X, W, Y,
                M=T, D=D,
                sX_b=sX_b, sX_m=sX_m, sX_d=sX_d,
                sW0=sW0, sW1=sW1,
                sY_m=sY_m, sY_d=sY_d,
                BLOCK_K=64,
                num_warps=2, num_stages=2,
            )
        # Now compute yB = hidden_states @ W.T -> [B, I, D]
        yB = torch.empty((B, I, D), dtype=torch.float32, device=hs.device)
        for b in range(B):
            X = hs[b]   # [I, D]
            Y = yB[b]   # [I, D]
            sX_b, sX_m, sX_d = X.stride()
            sW0, sW1 = W.stride()
            sY_m, sY_d = Y.stride()
            grid = (I,)
            triton_gemv_rowwise_kernel[grid](
                X, W, Y,
                M=I, D=D,
                sX_b=sX_b, sX_m=sX_m, sX_d=sX_d,
                sW0=sW0, sW1=sW1,
                sY_m=sY_m, sY_d=sY_d,
                BLOCK_K=64,
                num_warps=2, num_stages=2,
            )
        # Expand to batch using Triton (avoid torch.expand/stack)
        # For encoder: copy yA into output_E [B, T, D]
        output_E = torch.empty((B, T, D), dtype=torch.float32, device=hs.device)
        grid_e = (B,)
        triton_copy_expand_to_batch_encoder_kernel[grid_e](
            yA, output_E,
            T=T, D=D, B=B,
            sY_t=yA.stride(0), sY_d=yA.stride(1),
            sO_b=output_E.stride(0), sO_t=output_E.stride(1), sO_d=output_E.stride(2),
            num_warps=1, num_stages=1,
        )
        # For hidden: copy yB into output_H [B, I, D]
        output_H = torch.empty((B, I, D), dtype=torch.float32, device=hs.device)
        grid_h = (B,)
        triton_copy_expand_to_batch_hidden_kernel[grid_h](
            yB, output_H,
            I=I, D=D, B=B,
            sY_i=yB.stride(0), sY_d=yB.stride(1),
            sO_b=output_H.stride(0), sO_i=output_H.stride(1), sO_d=output_H.stride(2),
            num_warps=1, num_stages=1,
        )
        return output_E, output_H


def run(*args):
    return ModelNew()(*args)
