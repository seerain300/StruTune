import torch
import triton
import triton.language as tl


# Triton kernel: build concatenated input X_cat[b] = cat(encoder_hidden_states[b], hidden_states[b]) along sequence axis.
# Output:
#   X_cat[b] with shape [M, H], where M = T + I
# For each row p in [0, M):
#   if p < T: X_cat[b, p, :] = encoder_hidden_states[b, p, :]
#   else:     X_cat[b, p, :] = hidden_states[b, p - T, :]
@triton.jit
def cat_rows_kernel(
    e_ptr,  # encoder [B, T, H]
    i_ptr,  # hidden [B, I, H]
    out_ptr,  # output [B, M, H]
    B, T, I, H, M,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_ob, stride_om, stride_on,
):
    b = tl.program_id(0)
    p = tl.program_id(1)

    # Compute output offsets
    out_off = b * stride_ob + p * stride_om + tl.arange(0, H) * stride_on

    # Determine source tensor: encoder if p < T else hidden
    is_encoder = p < T
    # Compute source offsets accordingly
    if is_encoder:
        e_off = b * stride_eb + p * stride_et + tl.arange(0, H) * stride_eh
        vals = tl.load(e_ptr + e_off)
        tl.store(out_ptr + out_off, vals)
    else:
        src_row_i = p - T
        i_off = b * stride_ib + src_row_i * stride_it + tl.arange(0, H) * stride_ih
        vals = tl.load(i_ptr + i_off)
        tl.store(out_ptr + out_off, vals)


# Triton kernel: batched matmul for each batch b
# Compute Y[b] = X[b] @ W, where X[b] is [M, H] and W is [H, H]
# Inputs:
#   X_ptr: pointer to X[b] (we will pass X_cat[b] as X_ptr)
#   W_ptr: pointer to process_weight [H, H]
# Output:
#   Y_ptr: pointer to Y[b] [M, H]
@triton.jit
def batched_matmul_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, H,
    stride_Xb, stride_Xm, stride_Xn,
    stride_Wm, stride_Wn,
    stride_Yb, stride_Ym, stride_Yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < H:
        h_off = k + tl.arange(0, BLOCK_K)
        m_off = m_start + tl.arange(0, BLOCK_M)

        m_mask = m_off < M
        h_mask = h_off < H

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + pid_b * stride_Xb + m_off[:, None] * stride_Xm + h_off[None, :] * stride_Xn
        x_mask = m_mask[:, None] & h_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N]
        n_cols = n_start + tl.arange(0, BLOCK_N)
        w_ptrs = W_ptr + h_off[:, None] * stride_Wm + n_cols[None, :] * stride_Wn
        w_mask = h_mask[:, None] & (n_cols < H)[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, w_tile)

        k += BLOCK_K

    # Store accumulator to Y
    y_ptrs = Y_ptr + pid_b * stride_Yb + m_start * stride_Ym + n_start * stride_Yn + tl.arange(0, BLOCK_N) * stride_Yn
    y_mask = (m_start + tl.arange(0, BLOCK_M))[:, None] < M & (n_start + tl.arange(0, BLOCK_N))[None, :] < H
    tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA tensors for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."

        B = hidden_states.shape[0]
        assert B == encoder_hidden_states.shape[0], "Batch sizes must match."
        H = hidden_states.shape[2]
        assert H == encoder_hidden_states.shape[2], "Hidden dimensions must match."
        assert process_weight.shape == (H, H), "process_weight must be [H, H]."

        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        M = T + I

        # Allocate concatenated input per batch
        X_cat = [torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Strides for inputs and outputs
        stride_eb, stride_et, stride_eh = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        stride_ib, stride_it, stride_ih = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        stride_ob, stride_om, stride_on = X_cat[0].stride(0), X_cat[0].stride(1), X_cat[0].stride(2)

        # Launch cat kernel: grid = (B, M), one program per row
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat[0],
            B, T, I, H, M,
            stride_eb, stride_et, stride_eh,
            stride_ib, stride_it, stride_ih,
            stride_ob, stride_om, stride_on,
            num_warps=1, num_stages=1,
        )

        # Batched matmul: Y[b] = X_cat[b] @ process_weight
        X_ptrs = X_cat
        W = process_weight  # [H, H]
        Y = [torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Tiling parameters
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        tiles_m = (M + BLOCK_M - 1) // BLOCK_M
        tiles_n = (H + BLOCK_N - 1) // BLOCK_N

        grid_matmul = (B, tiles_m, tiles_n)
        batched_matmul_kernel[grid_matmul](
            X_ptrs[0], W, Y[0],
            M, H,
            stride_eb, stride_om, stride_on,   # X strides: b, m, n
            stride_eh, stride_eh,              # W strides: m, n
            stride_ob, stride_om, stride_on,   # Y strides: b, m, n
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Slicing to match original return: split the concatenated result back
        processed_encoder = Y[0][:T] if B == 1 else [Y[b][:T] for b in range(B)]
        processed_hidden = Y[0][T:] if B == 1 else [Y[b][T:] for b in range(B)]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
