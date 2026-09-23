import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,            # *const float32: [B, T, H]
    i_ptr,            # *const float32: [B, I, H]
    out_ptr,          # *float32: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2, # strides for e_ptr
    i_s0, i_s1, i_s2, # strides for i_ptr
    o_s0, o_s1, o_s2, # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)  # batch
    pid_l = tl.program_id(1)  # tiles over sequence (T+I)
    pid_h = tl.program_id(2)  # tiles over hidden dim (H)

    # Index vectors
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)     # sequence positions in [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)     # hidden dim indices in [0, H)

    # Masks
    mask_l = l < (T + I)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Base pointers for the batch
    base_e = e_ptr + pid_b * e_s0
    base_i = i_ptr + pid_b * i_s0
    base_o = out_ptr + pid_b * o_s0

    # Compute source addresses
    # If l < T -> from e_ptr[b, l, h]; else -> from i_ptr[b, l - T, h]
    l_is_encoder = l < T
    src_e_ptr = base_e + l[:, None] * e_s1 + h[None, :] * e_s2
    src_i_ptr = base_i + (l - T)[:, None] * i_s1 + h[None, :] * i_s2

    # Select source with masks
    src_ptr = tl.where(l_is_encoder[:, None], src_e_ptr, src_i_ptr)

    # Load and store
    val = tl.load(src_ptr, mask=mask, other=0.0)
    dst_ptr = base_o + l[:, None] * o_s1 + h[None, :] * o_s2
    tl.store(dst_ptr, val, mask=mask)


@triton.jit
def copy_rows_kernel(
    in_ptr, out_ptr,                  # *const float32, *float32
    Bsz: tl.int32, num_rows: tl.int32, Hsz: tl.int32,
    in_s0, in_s1, in_s2,             # strides for input
    out_s0, out_s1, out_s2,          # strides for output
    ROW_START: tl.int32,             # starting row to copy
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_l = tl.program_id(1)  # tiles over rows
    pid_h = tl.program_id(2)  # tiles over hidden dim

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # row indices in [ROW_START, ROW_START+num_rows)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden dim indices

    mask_l = l < ROW_START + num_rows
    mask_h = h < Hsz
    mask = mask_l[:, None] & mask_h[None, :]

    # Base pointers
    base_in = in_ptr + pid_b * in_s0
    base_out = out_ptr + pid_b * out_s0

    src_ptr = base_in + (l - ROW_START)[:, None] * in_s1 + h[None, :] * in_s2
    dst_ptr = base_out + l[:, None] * out_s1 + h[None, :] * out_s2

    val = tl.load(src_ptr, mask=mask, other=0.0)
    tl.store(dst_ptr, val, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure tensors are on CUDA and float32
        if not hidden_states.is_cuda or not encoder_hidden_states.is_cuda or not process_weight.is_cuda:
            raise RuntimeError("All inputs must be CUDA tensors.")
        if hidden_states.dtype != torch.float32 or encoder_hidden_states.dtype != torch.float32 or process_weight.dtype != torch.float32:
            raise RuntimeError("All inputs must be float32 tensors.")

        Bsz, T, H = encoder_hidden_states.shape
        I = hidden_states.shape[1]

        # Step 1: Concatenate along sequence dimension using Triton
        out = torch.empty((Bsz, T + I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch concat kernel
        BLOCK_l = 128
        BLOCK_h = 128
        grid_concat = (Bsz, triton.cdiv(T + I, BLOCK_l), triton.cdiv(H, BLOCK_h))
        concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out,
            Bsz, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # Step 2: Linear projection (use PyTorch for numerical robustness)
        # concatenated: [B, T+I, H], weight: [H, H] -> output: [B, T+I, H]
        processed = torch.matmul(out, process_weight.t())

        # Step 3: Split into encoder and image streams using Triton
        processed_encoder = torch.empty((Bsz, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((Bsz, I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        BLOCK_l_copy = 128
        BLOCK_h_copy = 128

        # Copy first T rows
        grid_copy_enc = (Bsz, triton.cdiv(T, BLOCK_l_copy), triton.cdiv(H, BLOCK_h_copy))
        copy_rows_kernel[grid_copy_enc](
            processed, processed_encoder,
            Bsz, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0,
            BLOCK_l=BLOCK_l_copy, BLOCK_h=BLOCK_h_copy,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows (from T to T+I)
        grid_copy_img = (Bsz, triton.cdiv(I, BLOCK_l_copy), triton.cdiv(H, BLOCK_h_copy))
        copy_rows_kernel[grid_copy_img](
            processed, processed_hidden,
            Bsz, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T,
            BLOCK_l=BLOCK_l_copy, BLOCK_h=BLOCK_h_copy,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
