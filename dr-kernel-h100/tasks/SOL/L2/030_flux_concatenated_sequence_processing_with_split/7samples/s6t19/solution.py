import torch
import triton
import triton.language as tl


@triton.jit
def split_into_encoder_kernel(
    src_ptr,            # [B, T+I, H] processed tensor
    out_ptr,            # [B, T, H] output for encoder part
    B: tl.int32, T: tl.int32, H: tl.int32,
    stride_src_n: tl.int32, stride_src_s: tl.int32, stride_src_h: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Grid: (B, T)
    n = tl.program_id(0)
    s = tl.program_id(1)

    src_row_ptr = src_ptr + n * stride_src_n + s * stride_src_s
    out_row_ptr = out_ptr + n * stride_out_n + s * stride_out_s

    h = 0
    while h < H:
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H
        vals = tl.load(src_row_ptr + offs * stride_src_h, mask=mask, other=0.0)
        tl.store(out_row_ptr + offs * stride_out_h, vals, mask=mask)
        h += BLOCK_H


@triton.jit
def split_into_hidden_kernel(
    src_ptr,            # [B, T+I, H] processed tensor
    out_ptr,            # [B, I, H] output for hidden part
    B: tl.int32, I: tl.int32, H: tl.int32,
    stride_src_n: tl.int32, stride_src_s: tl.int32, stride_src_h: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Grid: (B, I)
    n = tl.program_id(0)
    s = tl.program_id(1)  # sequence index within hidden part
    src_s = s + T  # since s in [0..I-1], total_seq = T + I

    src_row_ptr = src_ptr + n * stride_src_n + src_s * stride_src_s
    out_row_ptr = out_ptr + n * stride_out_n + s * stride_out_s

    h = 0
    while h < H:
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H
        vals = tl.load(src_row_ptr + offs * stride_src_h, mask=mask, other=0.0)
        tl.store(out_row_ptr + offs * stride_out_h, vals, mask=mask)
        h += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate along sequence dimension in PyTorch
        - Perform matmul in PyTorch for robustness
        - Split outputs using Triton kernels (no torch .matmul on tensors)
        """
        # Ensure tensors are on CUDA and contiguous
        device = hidden_states.device
        assert device.type == 'cuda', "ModelNew.forward requires CUDA tensors"
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()
        B, T, H_e = e.shape
        B_i, I, H_h = h.shape
        assert B == B_i and H_e == H_h, "Encoder and hidden tensors must have matching batch and hidden_dim"

        # Step 1: Concatenate along sequence dimension
        total_seq = T + I
        concatenated = torch.cat([e, h], dim=1)  # [B, T+I, H]

        # Step 2: Apply linear projection in PyTorch
        # process_weight is [H, H], we need weight.T with shape [H, H]
        # processed = concatenated @ process_weight.T
        processed = concatenated.matmul(w.transpose(0, 1))  # [B, T+I, H]
        # Ensure processed is contiguous
        processed = processed.contiguous()

        # Step 3: Split using Triton kernels
        processed_encoder = torch.empty((B, T, H), dtype=processed.dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=processed.dtype, device=device)

        # Launch Triton kernel for encoder part: grid (B, T)
        BLOCK_H = 128
        grid_encoder = (B, T)
        split_into_encoder_kernel[grid_encoder](
            processed,
            processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Launch Triton kernel for hidden part: grid (B, I)
        grid_hidden = (B, I)
        split_into_hidden_kernel[grid_hidden](
            processed,
            processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
