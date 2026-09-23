import torch
import triton
import triton.language as tl


# Triton kernel: concatenate along sequence axis for a single batch.
# For each batch b, writes rows of either encoder_hidden_states[b] or hidden_states[b]
# into X_cat[b] of shape [(T+I), H].
@triton.jit
def cat_rows_kernel(
    e_ptr, i_ptr, x_ptr,
    T, I,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_xb, stride_xm, stride_xn,
    H: tl.constexpr,
):
    b = tl.program_id(0)  # batch index
    p = tl.program_id(1)  # sequence position
    M = T + I

    # If p < T: take from encoder; else take from hidden states
    if p < T:
        row_idx = p
        e_row_ptr = e_ptr + b * stride_eb + row_idx * stride_et
        # Store entire row into X_cat[b, p, :]
        x_row_ptr = x_ptr + b * stride_xb + p * stride_xm
        # Copy H elements
        for n in range(H):
            val = tl.load(e_row_ptr + n * stride_eh)
            tl.store(x_row_ptr + n * stride_xn, val)
    else:
        row_idx = p - T
        i_row_ptr = i_ptr + b * stride_ib + row_idx * stride_it
        x_row_ptr = x_ptr + b * stride_xb + p * stride_xm
        for n in range(H):
            val = tl.load(i_row_ptr + n * stride_ih)
            tl.store(x_row_ptr + n * stride_xn, val)


# Triton kernel: batched GEMM Y[b] = X_cat[b] @ W where X_cat[b] is [M, H], W is [H, H], Y[b] is [M, H]
@triton.jit
def batched_matmul_kernel(
    x_ptr, w_ptr, y_ptr,
    M, H,
    stride_xb, stride_xm, stride_xn,
    stride_wk, stride_wn,  # W is [H, H] (k,n)
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k in range(0, H, BLOCK_K):
        # Load X tile: [BLOCK_M, BLOCK_N] with masking
        rows = tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        m_mask = rows < M
        n_mask = cols < H

        x_tile_ptr = x_ptr + b * stride_xb + rows[:, None] * stride_xm + (k + cols[None, :]) * stride_xn
        x_tile = tl.load(x_tile_ptr, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N], masking
        w_rows = tl.arange(0, BLOCK_K)
        w_cols = tl.arange(0, BLOCK_N)
        w_mask = (w_rows[:, None] < H) & (w_cols[None, :] < H)

        w_tile_ptr = w_ptr + (k + w_rows)[:, None] * stride_wk + w_cols[None, :] * stride_wn
        w_tile = tl.load(w_tile_ptr, mask=w_mask, other=0.0)

        # Multiply-accumulate
        acc += tl.dot(x_tile.to(tl.float32), w_tile.to(tl.float32))

    # Store result Y[b, :, :]
    y_rows = tl.arange(0, BLOCK_M)
    y_cols = tl.arange(0, BLOCK_N)
    y_mask = y_rows < M & y_cols < H
    y_ptr_tile = y_ptr + b * stride_yb + y_rows[:, None] * stride_ym + y_cols[None, :] * stride_yn
    tl.store(y_ptr_tile, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."

        B = hidden_states.shape[0]
        assert B == encoder_hidden_states.shape[0], "Batch sizes must match."
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert H == encoder_hidden_states.shape[2], "Hidden dimensions must match."
        assert process_weight.shape == (H, H), "process_weight must be [H, H]."

        # Allocate concatenated input per batch
        X_cat = [torch.empty((T + I, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Launch cat kernel: grid = (B, T+I)
        grid_cat = (B, T + I)

        # Extract strides (in elements)
        stride_eb, stride_et, stride_eh = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        stride_ib, stride_it, stride_ih = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        stride_xb, stride_xm, stride_xn = X_cat[0].stride(0), X_cat[0].stride(1), X_cat[0].stride(2)

        # Launch kernel (note: we pass H as a constexpr to simplify indexing)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat[0],
            T, I,
            stride_eb, stride_et, stride_eh,
            stride_ib, stride_it, stride_ih,
            stride_xb, stride_xm, stride_xn,
            H=H,
            num_warps=1, num_stages=1,
        )

        # Compute output per batch
        Y = [torch.empty((T + I, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Batched matmul kernel launch: grid = (B,)
        grid_mm = (B,)
        for b in range(B):
            # Strides
            stride_xb, stride_xm, stride_xn = X_cat[b].stride(0), X_cat[b].stride(1), X_cat[b].stride(2)
            stride_wk, stride_wn = process_weight.stride(0), process_weight.stride(1)
            stride_yb, stride_ym, stride_yn = Y[b].stride(0), Y[b].stride(1), Y[b].stride(2)

            # Choose block sizes; H is typically small (<=128). Use BLOCK_M=128, BLOCK_N=128, BLOCK_K=32.
            batched_matmul_kernel[grid_mm](
                X_cat[b], process_weight, Y[b],
                T + I, H,
                stride_xb, stride_xm, stride_xn,
                stride_wk, stride_wn,
                stride_yb, stride_ym, stride_yn,
                BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                num_warps=4, num_stages=3,
            )

        # Split results into encoder and hidden streams (host slicing only)
        processed_encoder = [Y[b][:T, :] for b in range(B)]
        processed_hidden = [Y[b][T:, :] for b in range(B)]

        # Cast back to original dtype to match original function
        processed_encoder = [pe.to(hidden_states.dtype) for pe in processed_encoder]
        processed_hidden = [ph.to(hidden_states.dtype) for ph in processed_hidden]

        # Return for B=1
        if B == 1:
            return processed_encoder[0], processed_hidden[0]
        else:
            return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
