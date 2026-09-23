import torch
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_kernel(
    A_ptr,          # *f32, pointer to A [B, T, H], contiguous
    B_ptr,          # *f32, pointer to B [H, H], contiguous
    Out_ptr,        # *f32, pointer to Out [B, T, H], contiguous
    B_size, T, H,   # int32 sizes (B_size = batch)
    stride_a_b, stride_a_t, stride_a_h,   # int32 strides for A
    stride_b_k, stride_b_h,               # int32 strides for B (k=input hidden, h=output hidden)
    stride_out_b, stride_out_t, stride_out_h,  # int32 strides for Out
    BLOCK_T: tl.constexpr,  # tile along T dimension
    BLOCK_H: tl.constexpr,  # tile along H (output hidden) dimension
    BLOCK_K: tl.constexpr,  # tile along K (input hidden) reduction dimension
):
    # program ids: batch, tile along T, tile along H
    b = tl.program_id(0)
    t_block = tl.program_id(1)
    h_block = tl.program_id(2)

    # Compute tile offsets
    t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    # mask for out-of-bounds
    t_mask = t_offsets < T
    h_mask = h_offsets < H

    # Accumulator
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

    # Loop over K (input hidden dimension) in chunks
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load A[b, t, k_offsets] as a [BLOCK_T, BLOCK_K] tile
        a_ptrs = (
            A_ptr
            + b * stride_a_b
            + t_offsets[:, None] * stride_a_t
            + k_offsets[None, :] * stride_a_h
        )
        a_mask = t_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B[k_offsets, h_offsets] as [BLOCK_K, BLOCK_H]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_b_k + h_offsets[None, :] * stride_b_h
        b_mask = k_mask[:, None] & h_mask[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store the result into Out[b, t, h]
    out_ptrs = Out_ptr + b * stride_out_b + t_offsets[:, None] * stride_out_t + h_offsets[None, :] * stride_out_h
    out_mask = t_mask[:, None] & h_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension
        - Applies linear projection using Triton GEMM: concatenated @ process_weight.T
        - Splits back into processed_encoder and processed_hidden
        """
        # Shapes and basic checks
        batch = encoder_hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        Bsize = hidden_states.shape[0]
        assert Bsize == batch, "batch sizes must match"
        assert hidden_states.shape[2] == H, "hidden_dim must match between inputs"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Concatenate along sequence dimension (PyTorch data movement)
        T = Stext + hidden_states.shape[1]
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        concatenated = concatenated.contiguous()  # ensure contiguous for Triton

        # Ensure process_weight is [H, H] and contiguous, and cast to dtype if needed
        process_weight_T = process_weight.transpose(0, 1).contiguous()
        if process_weight_T.dtype != dtype:
            process_weight_T = process_weight_T.to(dtype)

        # Allocate output [B, T, H]
        out = torch.empty((batch, T, H), device=device, dtype=dtype)

        # Strides for contiguous layout
        # A (concatenated): [B, T, H]
        stride_a_b = T * H
        stride_a_t = H
        stride_a_h = 1

        # B (process_weight_T): [H, H]
        stride_b_k = process_weight_T.stride(0)  # along input hidden (k)
        stride_b_h = process_weight_T.stride(1)  # along output hidden (h)

        # Out: [B, T, H]
        stride_out_b = T * H
        stride_out_t = H
        stride_out_h = 1

        # Launch grid: (B, ceil_div(T, BLOCK_T), ceil_div(H, BLOCK_H))
        BLOCK_T = 64
        BLOCK_H = 64
        BLOCK_K = 32

        grid = (batch, triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))

        _batched_matmul_kernel[grid](
            concatenated,          # A_ptr
            process_weight_T,      # B_ptr
            out,                   # Out_ptr
            batch, T, H,
            stride_a_b, stride_a_t, stride_a_h,
            stride_b_k, stride_b_h,
            stride_out_b, stride_out_t, stride_out_h,
            BLOCK_T=BLOCK_T,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            num_warps=4,   # default; tune if needed
            num_stages=2,  # pipeline stages
        )

        # Split back along the concatenated sequence dimension
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
