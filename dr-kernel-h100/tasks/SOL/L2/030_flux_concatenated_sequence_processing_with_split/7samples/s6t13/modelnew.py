import torch
import triton
import triton.language as tl


@triton.jit
def concat_cat_kernel(
    encoder_ptr,   # [B, T, H], float32, contiguous
    hidden_ptr,    # [B, I, H], float32, contiguous
    out_ptr,       # [B, T+I, H], float32, contiguous
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
):
    # Grid: (B, T+I). Each program writes one element out[n, s, k].
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Determine source: encoder vs hidden
    is_encoder = pid_s < T
    # Compute offsets
    out_offset = pid_n * stride_out_n + pid_s * stride_out_s  # row base

    if is_encoder:
        # Load from encoder: e[n, pid_s, k]
        # k is implicit in this flat kernel, but we can write a loop over H.
        # However, Triton expects loops over tiles; better to restructure grid.
        # Instead, we switch to a 3D grid over (B, T+I, H) to directly compute each element.
        # Define a separate kernel that uses 3D grid for safety and correctness.
        pass
    # Note: The above 'pass' is just a placeholder; we'll use a 3D kernel below.


@triton.jit
def concat_cat_kernel_3d(
    encoder_ptr,   # [B, T, H]
    hidden_ptr,    # [B, I, H]
    out_ptr,       # [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
):
    # Grid: (B, T+I, H). Each program computes one element out[n, s, k].
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_k = tl.program_id(2)  # index along hidden_dim H

    is_encoder = pid_s < T

    out_offset = pid_n * stride_out_n + pid_s * stride_out_s + pid_k * stride_out_h

    if is_encoder:
        # Load encoder[n, s, k]
        val = tl.load(encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s + pid_k * stride_e_h)
    else:
        # s - T selects the hidden sequence position
        s_hidden = pid_s - T
        val = tl.load(hidden_ptr + pid_n * stride_h_n + s_hidden * stride_h_s + pid_k * stride_h_h)

    tl.store(out_ptr + out_offset, val)


@triton.jit
def batched_matmul_row_kernel(
    x_ptr,      # [B, L, H], float32, contiguous, where L = T + I
    w_ptr,      # [H, H], float32, contiguous
    out_ptr,    # [B, L, H], float32, contiguous
    B: tl.int32, L: tl.int32, H: tl.int32,
    stride_x_n: tl.int32, stride_x_s: tl.int32, stride_x_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
):
    # Grid: (B, L). Each program computes out[n, s, :] for one row.
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Output base pointer for this row
    out_row_base = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s

    # Initialize accumulator
    acc = tl.zeros((H,), dtype=tl.float32)

    # Iterate over hidden dimension H: x[n, s, k] * w[k, h]
    # Since L could be different, but here L == H for this operation, we iterate k in [0..H-1].
    for k in range(0, H):
        # Load x[n, s, k]
        x_val = tl.load(x_ptr + pid_n * stride_x_n + pid_s * stride_x_s + k * stride_x_h)
        # Load w[k, :] row
        w_row = tl.load(w_ptr + k * stride_w_h + tl.arange(0, H) * stride_w_k)
        # Dot product of x_val (scalar) with w_row (vector)
        # Broadcast x_val to match vector: x_val * w_row
        acc += x_val * w_row

    # Store the accumulated result back to out[n, s, :]
    tl.store(out_row_base + tl.arange(0, H) * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        Computes:
            concatenated = cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
            processed = concatenated @ process_weight.T                         # [B, T+I, H]
            processed_encoder = processed[:, :T, :]
            processed_hidden   = processed[:, T:, :]
        All core computation is performed by Triton kernels.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, L, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure contiguous tensors
        e = encoder_hidden_states.contiguous()  # [B, T, H]
        h = hidden_states.contiguous()          # [B, I, H]
        w = process_weight.contiguous()         # [H, H]

        # 1) Concatenate along sequence dimension inside Triton: out_cat [B, L, H]
        out_cat = torch.empty((B, L, H), device=device, dtype=dtype)

        stride_e_n, stride_e_s, stride_e_h = e.stride()
        stride_h_n, stride_h_s, stride_h_h = h.stride()
        stride_out_n, stride_out_s, stride_out_h = out_cat.stride()

        # Launch 3D grid for concat: (B, T+I, H)
        grid_cat = (B, L, H)
        concat_cat_kernel_3d[grid_cat](
            e, h, out_cat,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_out_n, stride_out_s, stride_out_h,
            num_warps=4,
        )

        # 2) Matmul per row: out [B, L, H] = out_cat @ w.T
        out = torch.empty((B, L, H), device=device, dtype=dtype)

        stride_x_n, stride_x_s, stride_x_h = out_cat.stride()
        stride_w_h, stride_w_k = w.stride()
        stride_out_n, stride_out_s, stride_out_h = out.stride()

        # Grid: (B, L)
        grid_mat = (B, L)
        batched_matmul_row_kernel[grid_mat](
            out_cat, w, out,
            B, L, H,
            stride_x_n, stride_x_s, stride_x_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            num_warps=4,
        )

        # 3) Split outputs into encoder and hidden streams
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden