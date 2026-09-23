import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr,        # *ptr to encoder_hidden_states [B, T, H]
    h_ptr,        # *ptr to hidden_states [B, I, H]
    out_ptr,      # *ptr to X_cat [B, M, H], where M = T + I
    B, T, I, H, M,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_out_b, stride_out_m, stride_out_h,
    BLOCK_H: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # row index in [0, M)

    # bounds check
    if pid_b >= B or pid_m >= M:
        return

    # determine source: first T rows from e, remaining from h
    is_e = pid_m < T

    # compute base offsets
    # row offsets
    # For e: row idx = pid_m
    # For h: row idx = pid_m - T
    src_idx = pid_m if is_e else (pid_m - T)

    # strides for e/h
    stride_e = stride_e_b * pid_b + stride_e_t * src_idx
    stride_h = stride_h_b * pid_b + stride_h_i * (pid_m - T)  # pid_m >= T guaranteed here

    # output offsets
    out_offset = stride_out_b * pid_b + stride_out_m * pid_m

    # iterate over hidden dim in tiles
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask = offs_h < H

        # load source row
        val_e = tl.load(e_ptr + stride_e + offs_h * stride_e_h, mask=mask, other=0.0)
        val_h = tl.load(h_ptr + stride_h + offs_h * stride_h_h, mask=mask, other=0.0)
        # select appropriate source
        val = tl.where(is_e, val_e, val_h)

        # store into output
        tl.store(out_ptr + out_offset + offs_h * stride_out_h, val, mask=mask)


@triton.jit
def matmul_rows_kernel(
    x_ptr,        # *ptr to X_cat [B, M, H]
    w_ptr,        # *ptr to process_weight [H, H]
    out_ptr,      # *ptr to Y [B, M, H]
    B, M, H,
    stride_x_b, stride_x_m, stride_x_h,
    stride_w_h, stride_w_h2,  # W is [H, H], second dim is h (columns)
    stride_out_b, stride_out_m, stride_out_h,
    BLOCK_K: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # row index in [0, M)

    if pid_b >= B or pid_m >= M:
        return

    # base offsets
    x_base = stride_x_b * pid_b + stride_x_m * pid_m
    out_base = stride_out_b * pid_b + stride_out_m * pid_m

    # accumulate in float32
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # loop over K (hidden dim) in tiles
    for k_start in range(0, H, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # load x row segment: X[pid_b, pid_m, offs_k]
        x_vals = tl.load(x_ptr + x_base + offs_k * stride_x_h, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]
        # load W segment rows: W[offs_k, :] -> shape [BLOCK_K, H]
        w_ptrs = w_ptr + offs_k[:, None] * stride_w_h + tl.arange(0, H) * stride_w_h2  # broadcast over H
        # We need a mask for the second dimension: for k < H we keep all H, but we must guard offs_k < H
        # Since offs_k < H for valid k_start, we can load with mask on the first dim only
        w_vals = tl.load(w_ptrs, mask=(offs_k[:, None] < H), other=0.0)  # [BLOCK_K, H]
        w_vals = w_vals.to(tl.float32)

        # acc += x_vals[:, None] * w_vals
        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

    # store result (cast to original dtype as needed; here we keep float32 for stability)
    # out_ptr stores float32 in typical torch float32; if inputs were fp16/bf16, we cast accordingly.
    tl.store(out_ptr + out_base + tl.arange(0, H) * stride_out_h, acc, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation of the original run function:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Performs linear projection using a Triton row-wise GEMM per batch.
        - Splits the result back into encoder and hidden streams (host slicing).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Prepare tensors and allocate outputs
        # Ensure contiguous for simple stride arithmetic
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()  # [H, H]

        # Allocate concatenated X_cat [B, M, H]
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch concatenation kernel: grid = (B, M)
        BLOCK_H = 64  # tile over hidden dim
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            e, h, X_cat,
            B, T, I, H, M,
            e.stride(0), e.stride(1), e.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=2, num_stages=2,
        )

        # Allocate output Y [B, M, H]
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)  # accumulate/store as float32

        # Launch row-wise matmul kernel: grid = (B, M)
        grid_mm = (B, M)
        matmul_rows_kernel[grid_mm](
            X_cat, w, Y,
            B, M, H,
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            w.stride(0), w.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_K=64,  # tile over hidden dim
            num_warps=2, num_stages=2,
        )

        # Convert to original dtype for consistency
        Y = Y.to(hidden_states.dtype)

        # Split results: first T rows for encoder, remaining for hidden
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
