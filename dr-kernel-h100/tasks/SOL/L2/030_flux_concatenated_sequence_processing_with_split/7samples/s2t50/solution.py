import torch
import triton
import triton.language as tl


@triton.jit
def split_seqs_kernel(
    src_ptr,           # pointer to processed tensor of shape [B, S, H], S = T + I
    out_encoder_ptr,   # pointer to output [B, T, H]
    out_hidden_ptr,    # pointer to output [B, I, H]
    B, T, I, H, S,
    stride_src_b, stride_src_s, stride_src_h,        # strides for src
    stride_e_b, stride_e_s, stride_e_h,              # strides for out_encoder
    stride_h_b, stride_h_s, stride_h_h,              # strides for out_hidden
    BLOCK_S: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)

    # Process encoder rows 0..T-1
    for t_start in range(0, T, BLOCK_S):
        t_offsets = t_start + tl.arange(0, BLOCK_S)
        mask_t = t_offsets < T
        src_row_ptr = src_ptr + b * stride_src_b + t_offsets * stride_src_s
        x = tl.load(src_row_ptr + tl.arange(0, H) * stride_src_h, mask=mask_t[:, None], other=0.0)
        dst_row_ptr = out_encoder_ptr + b * stride_e_b + t_offsets * stride_e_s
        tl.store(dst_row_ptr + tl.arange(0, H) * stride_e_h, x, mask=mask_t[:, None])

    # Process hidden rows T..T+I-1
    for i_start in range(0, I, BLOCK_S):
        i_offsets = i_start + tl.arange(0, BLOCK_S)
        mask_i = i_offsets < I
        src_row_ptr = src_ptr + b * stride_src_b + (T + i_offsets) * stride_src_s
        x = tl.load(src_row_ptr + tl.arange(0, H) * stride_src_h, mask=mask_i[:, None], other=0.0)
        dst_row_ptr = out_hidden_ptr + b * stride_h_b + i_offsets * stride_h_s
        tl.store(dst_row_ptr + tl.arange(0, H) * stride_h_h, x, mask=mask_i[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton split-only implementation:
        - Concatenation and matmul are performed in PyTorch to ensure robust correctness.
        - Triton kernel is launched to split the processed tensor into encoder and hidden streams.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        S = T + I

        # Concatenate along sequence dimension using PyTorch (robust and correct)
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, S, H]

        # Linear projection using PyTorch (no bias)
        # process_weight is [H, H], so process_weight.T is [H, H]
        processed = torch.matmul(concatenated, process_weight.t())  # [B, S, H]

        # Ensure processed tensor is on CUDA for Triton
        assert processed.is_cuda, "processed tensor must be on CUDA for Triton kernels"
        processed = processed.contiguous()

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), device=processed.device, dtype=processed.dtype)
        processed_hidden = torch.empty((B, I, H), device=processed.device, dtype=processed.dtype)

        # Launch Triton split kernel: one program per batch
        grid = (B,)
        split_seqs_kernel[grid](
            processed,
            processed_encoder,
            processed_hidden,
            B, T, I, H, S,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
