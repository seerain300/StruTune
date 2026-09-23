import torch
import triton
import triton.language as tl


@triton.jit
def batched_gemm_outer_kernel(
    input_ptr,         # *f32, in [B, T, H_in] where T = Stext + Simg
    weightT_ptr,       # *f32, process_weight.T [H_in, H_out] (H_in = H_out = hidden_dim)
    output_ptr,        # *f32, out [B, T, H_out]
    B, T, H_in, H_out,
    stride_in_b, stride_in_t, stride_in_h,
    stride_w_k, stride_w_n,
    stride_out_b, stride_out_t, stride_out_h,
    BLOCK_K: tl.constexpr,  # tile along input H_in
    BLOCK_N: tl.constexpr,  # tile along output H_out
):
    # One program per (b, t). It computes a block of outputs along the hidden dim.
    b = tl.program_id(0)
    t = tl.program_id(1)

    if (b >= B) or (t >= T):
        return

    # Initialize accumulator for this (b, t) over a block of outputs
    n_start = tl.program_id(2) * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H_out
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over input dimension in chunks
    k0 = 0
    while k0 < H_in:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input chunk: input[b, t, k_offsets] -> [BLOCK_K]
        in_ptrs = input_ptr + b * stride_in_b + t * stride_in_t + k_offsets * stride_in_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight block: weightT[k_offsets, n_offsets] -> [BLOCK_K, BLOCK_N]
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        w_block = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += in_chunk[:, None] * w_block
        acc += tl.sum(in_chunk[:, None] * w_block, axis=0)

        k0 += BLOCK_K

    # Store the accumulated outputs for this block
    out_ptrs = output_ptr + b * stride_out_b + t * stride_out_t + n_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, Simg, H]
        encoder_hidden_states: torch.Tensor,  # [B, Stext, H]
        process_weight: torch.Tensor,         # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure device and dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and \
            process_weight.dtype == torch.float32, "Use float32 tensors."

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        H = hidden_states.shape[2]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg

        # 1) Concatenate along sequence dimension on host (metadata op)
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T, H]

        # 2) Triton GEMV: out[b, t, :] = concatenated[b, t, :] @ process_weight.T
        H_in = concatenated.shape[2]  # H
        H_out = process_weight.shape[1]  # H
        assert H_in == H_out, "process_weight must be square with dim matching hidden size."

        out = torch.empty((B, T, H_out), dtype=torch.float32, device=hidden_states.device)

        # Grid: one program per (b, t); third dim tiles the output hidden dim.
        BLOCK_N = 128  # tile size along output hidden dim
        BLOCK_K = 64   # chunk size along input hidden dim
        grid = (B, T, triton.cdiv(H_out, BLOCK_N))

        batched_gemm_outer_kernel[grid](
            concatenated, process_weight.transpose(0, 1), out,   # process_weight.T is [H, H]
            B, T, H_in, H_out,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight.transpose(0, 1).stride(0), process_weight.transpose(0, 1).stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into streams
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
