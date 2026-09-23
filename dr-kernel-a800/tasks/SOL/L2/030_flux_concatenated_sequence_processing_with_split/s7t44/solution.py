import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr,  # *float32, [B, T, H]
    i_ptr,  # *float32, [B, I, H]
    out_ptr,  # *float32, [B, M, H]
    B: tl.constexpr,  # batch size (for loops; actual used as pointer stride)
    T: tl.constexpr,  # text seq len
    I: tl.constexpr,  # img seq len
    H: tl.constexpr,  # hidden dim
    M: tl.constexpr,  # total seq len = T + I
    BLOCK_M: tl.constexpr,  # tile over rows (unused, but can be used for vectorization)
):
    # Grid: (B, M)
    b = tl.program_id(0)
    p = tl.program_id(1)
    # Bounds check: in case grid > M, but we launch grid=(B, M) exactly.
    # Compute base offsets
    stride_e_b = 1 * H  # hidden dim step between batches for contiguous [B, T, H]
    stride_e_t = H      # step between time steps
    stride_e_h = 1      # contiguous along H

    stride_i_b = 1 * H
    stride_i_i = H
    stride_i_h = 1

    stride_out_b = 1 * H
    stride_out_p = H
    stride_out_h = 1

    # Pointers for source
    e_row_ptr = e_ptr + b * stride_e_b + p * stride_e_t
    i_row_ptr = i_ptr + b * stride_i_b + (p - T) * stride_i_i  # if p < T, p-T < 0 -> masked

    # Mask for row p valid
    valid_p = p < T  # if true, take from e; else take from i
    # We will select source pointer based on valid_p
    src_ptr = tl.where(valid_p, e_row_ptr, i_row_ptr)

    # Destination pointer
    out_row_ptr = out_ptr + b * stride_out_b + p * stride_out_p

    # Copy H elements: contiguous along H
    for h in range(0, H):
        val = tl.load(src_ptr + h * stride_e_h)  # stride_e_h is 1; works for both e/i
        tl.store(out_row_ptr + h * stride_out_h, val)


@triton.jit
def batched_matmul_rows_kernel(
    x_ptr,   # *float32, [B, M, H]
    w_ptr,   # *float32, [H, H]
    y_ptr,   # *float32, [B, M, H]
    B: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, M)
    b = tl.program_id(0)
    p = tl.program_id(1)

    # Accumulator for Y[b, p, :]
    acc = tl.zeros((H,), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k_start in range(0, H, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_range < H

        # Load X[b, p, k_range] (1D vector of size BLOCK_K)
        x_row_ptr = x_ptr + b * (M * H) + p * H  # since x is [B, M, H] contiguous, step M*H between batches, H between rows
        x_vals = tl.load(x_row_ptr + k_range, mask=mask_k, other=0.0)  # shape: [BLOCK_K]

        # Load W[k_range, :] (matrix row segment)
        w_row_ptr = w_ptr + k_range * H  # W is [H, H], contiguous rows
        w_vals = tl.load(w_row_ptr, mask=mask_k, other=0.0)  # shape: [BLOCK_K]

        # Accumulate dot product: acc += sum(x_vals * w_vals) over k
        # x_vals: [BLOCK_K], w_vals: [BLOCK_K]
        acc += tl.sum(x_vals * w_vals, axis=0)

    # Store result to y[b, p, :]
    y_row_ptr = y_ptr + b * (M * H) + p * H
    tl.store(y_row_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of the original run function:
        Concatenates along sequence dimension, applies linear projection, and splits back.
        All computation is done via Triton kernels. No torch operations are used in forward.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Ensure contiguous tensors
        e = encoder_hidden_states.contiguous()
        i = hidden_states.contiguous()
        w = process_weight.contiguous()

        # Allocate concatenated input: [B, M, H], float32 for kernel
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch cat_rows_kernel: grid = (B, M)
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            e, i, X_cat,
            B, T, I, H, M,
            BLOCK_M=1,
            num_warps=1, num_stages=1,
        )

        # Launch batched_matmul_rows_kernel: grid = (B, M)
        grid_mm = (B, M)
        # Choose BLOCK_K based on H; for robustness, use 64 or 128
        BLOCK_K = 64 if H >= 64 else 32
        batched_matmul_rows_kernel[grid_mm](
            X_cat, w, Y,
            B, M, H,
            BLOCK_K=BLOCK_K,
            num_warps=1, num_stages=1,
        )

        # Split results: cast back to original dtype if needed
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        # Match original return types: convert back to original dtypes
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
