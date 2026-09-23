import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    e_ptr,            # *f32, [B, Stext, H]
    h_ptr,            # *f32, [B, Simg, H]
    out_ptr,          # *f32, [B, T, H]
    B,                # int
    Stext,            # int
    Simg,             # int
    H,                # int
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_t, stride_h_h,
    stride_out_b, stride_out_t, stride_out_h,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    if pid_b >= B or pid_t >= (Stext + Simg):
        return

    # Choose source based on sequence position
    is_encoder = pid_t < Stext
    base_src = e_ptr + pid_b * stride_e_b + pid_t * stride_e_t if is_encoder \
               else h_ptr + pid_b * stride_h_b + (pid_t - Stext) * stride_h_t

    # Write entire hidden dimension vector
    h_vec = tl.arange(0, H)
    vals = tl.load(base_src + h_vec * stride_e_h, mask=h_vec < H, other=0.0)
    tl.store(out_ptr + pid_b * stride_out_b + pid_t * stride_out_t + h_vec * stride_out_h, vals)


@triton.jit
def apply_weight_kernel(
    a_ptr,            # *f32, [B, T, H] (concatenated)
    wT_ptr,           # *f32, [H, H] (transpose of process_weight)
    out_ptr,          # *f32, [B, T, H]
    B, T, H,
    stride_a_b, stride_a_t, stride_a_h,
    stride_w_b, stride_w_t, stride_w_h,
    stride_out_b, stride_out_t, stride_out_h,
):
    # One program per (b, t, h)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    if pid_b >= B or pid_t >= T or pid_h >= H:
        return

    # Compute out[b, t, h] = sum_k a[b, t, k] * wT[k, h]
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, H):
        a_val = tl.load(a_ptr + pid_b * stride_a_b + pid_t * stride_a_t + k * stride_a_h)
        w_val = tl.load(wT_ptr + k * stride_w_b + pid_h * stride_w_h)
        acc += a_val * w_val

    tl.store(out_ptr + pid_b * stride_out_b + pid_t * stride_out_t + pid_h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension (Triton).
        - Applies linear projection via elementwise accumulation with process_weight.T (Triton).
        - Splits results back into encoder and image parts (view only).
        """
        # Ensure CUDA and dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA device."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        # Make inputs contiguous
        e = encoder_hidden_states.contiguous()   # [B, Stext, H]
        h = hidden_states.contiguous()           # [B, Simg, H]
        W = process_weight.contiguous()          # [H, H]
        B, Stext, H = e.shape
        B_h, Simg, H2 = h.shape
        assert B == B_h and H == H2, "Batch and hidden_dim must match."
        T = Stext + Simg

        # Allocate concatenated tensor
        out_cat = torch.empty((B, T, H), device=e.device, dtype=torch.float32)

        # Launch concat kernel: grid (B, T)
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            e, h, out_cat,
            B, Stext, Simg, H,
            e.stride(0), e.stride(1), e.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=1,
            num_stages=1,
        )

        # Prepare W_T = process_weight.T
        W_T = W.t().contiguous()  # [H, H]

        # Allocate output
        out = torch.empty((B, T, H), device=e.device, dtype=torch.float32)

        # Launch apply-weight kernel: grid (B, T, H)
        grid_apply = (B, T, H)
        apply_weight_kernel[grid_apply](
            out_cat, W_T, out,
            B, T, H,
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            W_T.stride(0), W_T.stride(1), W_T.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=1,
            num_stages=1,
        )

        # Split along sequence dimension (views)
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
