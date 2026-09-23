import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr,        # *f32, encoder_hidden_states [B, T, H]
    i_ptr,        # *f32, hidden_states [B, I, H]
    out_ptr,      # *f32, output concatenated [B, T+I, H]
    T, I, H,      # ints
    stride_e_b, stride_e_t, stride_e_h,
    stride_i_b, stride_i_i, stride_i_h,
    stride_out_b, stride_out_p, stride_out_h,
):
    # 2D grid: (batch, row index)
    b = tl.program_id(0)
    p = tl.program_id(1)

    # If p < T, take row from e; else take row from i at index p - T
    src_i = p - T
    offs_h = tl.arange(0, H)

    e_row_ptr = e_ptr + b * stride_e_b + p * stride_e_t
    i_row_ptr = i_ptr + b * stride_i_b + src_i * stride_i_i
    out_row_ptr = out_ptr + b * stride_out_b + p * stride_out_p

    if p < T:
        vals = tl.load(e_row_ptr + offs_h * stride_e_h, mask=offs_h < H, other=0.0)
    else:
        vals = tl.load(i_row_ptr + offs_h * stride_i_h, mask=offs_h < H, other=0.0)

    tl.store(out_row_ptr + offs_h * stride_out_h, vals, mask=offs_h < H)


@triton.jit
def batched_matmul_rows_kernel(
    x_ptr,        # *f32, X_cat [B, M, H]
    w_ptr,        # *f32, process_weight [H, H]
    y_ptr,        # *f32, output [B, M, H]
    B, M, H,      # ints
    stride_x_b, stride_x_m, stride_x_h,
    stride_w_h, stride_w_k,  # for [H, H], stride_w_k = stride_w_h
    stride_y_b, stride_y_m, stride_y_h,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: (batch, row index)
    b = tl.program_id(0)
    m = tl.program_id(1)

    # Accumulate in float32
    acc = tl.zeros((H,), dtype=tl.float32)

    # Loop over k dimension in tiles
    for k_start in range(0, H, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # Load x[b, m, k_tile] as vector of length BLOCK_K
        x_row_ptr = x_ptr + b * stride_x_b + m * stride_x_m
        x_vals = tl.load(x_row_ptr + offs_k * stride_x_h, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load w[k_tile, :] as [BLOCK_K, H] by iterating kk and forming columns
        w_mat = tl.zeros((BLOCK_K, H), dtype=tl.float32)
        for kk in range(BLOCK_K):
            col_k = offs_k[kk]
            valid = col_k < H
            # For invalid kk, skip (masked loads handled via valid). Use masked load or assign 0.
            w_col_ptr = w_ptr + col_k * stride_w_h
            w_mat[kk, :] = tl.load(w_col_ptr + tl.arange(0, H) * stride_w_k, mask=valid, other=0.0)

        # Accumulate: acc += sum over k in tile of x_vals[k] * w_mat[k, :]
        for kk in range(BLOCK_K):
            col_k = offs_k[kk]
            valid = col_k < H
            if valid:
                acc += x_vals[kk] * w_mat[kk, :]

    # Store result to y[b, m, :]
    y_row_ptr = y_ptr + b * stride_y_b + m * stride_y_m
    tl.store(y_row_ptr + tl.arange(0, H) * stride_y_h, acc, mask=tl.arange(0, H) < H)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton kernels."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, H), "encoder_hidden_states must be [B, T, H]"
        assert hidden_states.shape == (B, I, H), "hidden_states must be [B, I, H]"
        assert process_weight.shape == (H, H), "process_weight must be [H, H]"

        M = T + I

        # Allocate concatenated tensor [B, T+I, H]
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton kernel to concatenate rows
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat,
            T, I, H,
            *encoder_hidden_states.stride(),   # stride_e_b, stride_e_t, stride_e_h
            *hidden_states.stride(),           # stride_i_b, stride_i_i, stride_i_h
            *X_cat.stride(),                   # stride_out_b, stride_out_p, stride_out_h
            num_warps=1, num_stages=1,
        )

        # Allocate output for processed [B, T+I, H]
        processed = torch.empty((B, M, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton batched matmul rows kernel
        grid_mm = (B, M)
        batched_matmul_rows_kernel[grid_mm](
            X_cat, process_weight, processed,
            B, M, H,
            *X_cat.stride(),             # stride_x_b, stride_x_m, stride_x_h
            *process_weight.stride(),    # stride_w_h, stride_w_k
            *processed.stride(),         # stride_y_b, stride_y_m, stride_y_h
            BLOCK_K=64,
            num_warps=1, num_stages=1,
        )

        # Split streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
