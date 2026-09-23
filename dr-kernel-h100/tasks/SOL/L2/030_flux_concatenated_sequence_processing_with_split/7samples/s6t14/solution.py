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
    # Grid: (B, T+I, H). Each program computes one element out[n, s, h].
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    total_seq = T + I

    # Determine source tensor: if s < T -> encoder, else hidden
    is_encoder = pid_s < T

    # Compute pointers for the source row and store into out
    if is_encoder:
        src_ptr = encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s
        val = tl.load(src_ptr + pid_h * stride_e_h)
    else:
        src_ptr = hidden_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s
        val = tl.load(src_ptr + pid_h * stride_h_h)

    # Store to concatenated output
    out_ptr_el = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s + pid_h * stride_out_h
    tl.store(out_ptr_el, val)


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
    # Grid: (B, L). Each program computes one output row (n, s).
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Prepare output vector pointer
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s

    # Accumulator for output row (float32)
    acc = tl.zeros((H,), dtype=tl.float32)

    # Load input row from x[n, s, :]
    x_row_ptr = x_ptr + pid_n * stride_x_n + pid_s * stride_x_s
    # Iterate over H (input features) and accumulate dot with weight
    for h in range(0, H):
        x_val = tl.load(x_row_ptr + h * stride_x_h)  # input at feature h
        # weight[h, :] dot product: sum_k x_val * w[h, k]
        # Simple loop over k; H is typically moderate (e.g., 128-512)
        for k in range(0, H):
            w_val = tl.load(w_ptr + h * stride_w_h + k * stride_w_k)
            acc[h] += x_val * w_val

    # Store the output row
    for h in range(0, H):
        tl.store(out_row_ptr + h * stride_out_h, acc[h])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure contiguity and dtype
        device = hidden_states.device
        # We will run Triton kernels expecting float32; if not, cast safely
        dtype = hidden_states.dtype
        if dtype not in (torch.float32,):
            # For safety, cast to float32 for kernel; output will be float32
            hidden_states = hidden_states.float()
            encoder_hidden_states = encoder_hidden_states.float()
            process_weight = process_weight.float()

        B, T, H = encoder_hidden_states.shape
        I = hidden_states.shape[1]
        total_seq = T + I
        L = total_seq

        # 1) Concatenate along sequence dimension using Triton
        out_cat = torch.empty((B, L, H), device=device, dtype=torch.float32)
        stride_e_n, stride_e_s, stride_e_h = encoder_hidden_states.contiguous().stride()
        stride_h_n, stride_h_s, stride_h_h = hidden_states.contiguous().stride()
        stride_out_n, stride_out_s, stride_out_h = out_cat.stride()

        # Launch 3D grid over (B, L, H)
        grid_cat = (B, L, H)
        concat_cat_kernel[grid_cat](
            encoder_hidden_states.contiguous(), hidden_states.contiguous(), out_cat,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_out_n, stride_out_s, stride_out_h,
            num_warps=2,
        )

        # 2) Matmul: out = out_cat @ process_weight.T using Triton per-row
        weight = process_weight.contiguous()  # [H, H], float32
        out = torch.empty((B, L, H), device=device, dtype=torch.float32)

        stride_x_n, stride_x_s, stride_x_h = out_cat.stride()
        stride_w_h, stride_w_k = weight.stride()
        stride_out_n, stride_out_s, stride_out_h = out.stride()

        grid_mat = (B, L)
        batched_matmul_row_kernel[grid_mat](
            out_cat, weight, out,
            B, L, H,
            stride_x_n, stride_x_s, stride_x_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            num_warps=4,
        )

        # 3) Split into encoder and hidden streams
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
