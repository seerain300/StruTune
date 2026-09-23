import torch
import triton
import triton.language as tl


@triton.jit
def cat_lane_kernel(
    e_ptr, i_ptr, x_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_xb, stride_xm, stride_xh,
):
    # 2D grid: (batch, col)
    b = tl.program_id(0)
    n = tl.program_id(1)

    M = T + I

    # Loop over rows p = 0..M-1
    for p in range(0, M):
        is_img = p >= T
        src = tl.where(is_img, i_ptr + b * stride_ib, e_ptr + b * stride_eb)
        p_src = tl.where(is_img, p - T, p)
        val = tl.load(src + p_src * tl.where(is_img, stride_it, stride_et) + n * tl.where(is_img, stride_ih, stride_eh))
        tl.store(x_ptr + b * stride_xb + p * stride_xm + n * stride_xh, val)


@triton.jit
def batched_matmul_lane_kernel(
    x_ptr, w_ptr, y_ptr,
    B, T, I, H,  # T and I for consistency; M = T + I is used implicitly
    stride_xb, stride_xm, stride_xh,
    stride_wk, stride_wh,  # w is [H, H]
    stride_yb, stride_ym, stride_yh,
):
    # 2D grid: (batch, col)
    b = tl.program_id(0)
    n = tl.program_id(1)

    # Accumulator for this (batch, col)
    acc = tl.zeros((T + I,), dtype=tl.float32)

    # Loop over rows p = 0..M-1
    for p in range(0, T + I):
        # Load x[b, p, n]
        x_val = tl.load(x_ptr + b * stride_xb + p * stride_xm + n * stride_xh)
        # Accumulate sum over k of x[b, p, k] * w[k, n]
        # We compute scalar product for this row p across all k by looping k
        # Note: H is known at compile time; this loop is valid in Triton.
        for k in range(0, H):
            w_val = tl.load(w_ptr + k * stride_wk + n * stride_wh)
            acc[p] += x_val * w_val

    # Store the accumulator to y[b, :, n]
    for p in range(0, T + I):
        tl.store(y_ptr + b * stride_yb + p * stride_ym + n * stride_yh, acc[p])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Triton-only forward: ensure tensors are CUDA float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All tensors must be on CUDA for Triton kernels."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, \
            "This Triton implementation currently supports float32 tensors."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Allocate concatenated matrix X_cat: [B, M, H]
        X_cat = torch.empty((B, M, H), dtype=torch.float32, device=hidden_states.device)

        # Launch cat_lane_kernel: grid = (B, H)
        grid_cat = (B, H)
        cat_lane_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            num_warps=4, num_stages=2,
        )

        # Allocate output Y: [B, M, H]
        Y = torch.empty((B, M, H), dtype=torch.float32, device=hidden_states.device)

        # Launch batched_matmul_lane_kernel: grid = (B, H)
        grid_mm = (B, H)
        batched_matmul_lane_kernel[grid_mm](
            X_cat, process_weight, Y,
            B, T, I, H,
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            num_warps=4, num_stages=2,
        )

        # Split results: first T rows for encoder, remaining I rows for hidden
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
