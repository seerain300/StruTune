import torch
import triton
import triton.language as tl


# Triton kernel: computes out[b, p, d] = sum_{k=0..D-1} A[b, p, k] * B[k, d]
# A: [B, M, D], B: [D, D], out: [B, M, D]
@triton.jit
def _gemm_row_reduce_ABD_kernel(A_ptr, B_ptr, Out_ptr,
                                 B, M, D,
                                 A_s0, A_s1, A_s2,
                                 B_s0, B_s1,
                                 Out_s0, Out_s1, Out_s2,
                                 BLOCK_N: tl.constexpr):
    # program ids
    b = tl.program_id(0)
    p = tl.program_id(1)
    d_block = tl.program_id(2)

    # compute d indices for this block
    d = d_block * BLOCK_N + tl.arange(0, BLOCK_N)
    # mask for valid d range
    mask_d = d < D

    # accumulator for this vector of d
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # loop over k in 0..D-1
    # Note: This is robust but not as fast as blocked matmul.
    for k in range(0, D):
        # load A[b, p, k] as scalar
        a_ptr = A_ptr + b * A_s0 + p * A_s1 + k * A_s2
        a_val = tl.load(a_ptr)
        a_val = a_val.to(tl.float32)

        # load B[k, d] as vector over d
        b_ptr = B_ptr + k * B_s0 + d * B_s1
        b_vec = tl.load(b_ptr, mask=mask_d, other=0.0)
        b_vec = b_vec.to(tl.float32)

        # accumulate
        acc += a_val * b_vec

    # store results
    out_ptr = Out_ptr + b * Out_s0 + p * Out_s1 + d * Out_s2
    tl.store(out_ptr, acc, mask=mask_d)


# Triton kernel: computes out[b, p, d] = sum_{k=0..D-1} A[b, p, k] * B[k, d]
# A: [B, I, D], B: [D, D], out: [B, I, D]
@triton.jit
def _gemm_row_reduce_AI_D_kernel(A_ptr, B_ptr, Out_ptr,
                                 B2, I2, D2,
                                 A_s0, A_s1, A_s2,
                                 B_s0, B_s1,
                                 Out_s0, Out_s1, Out_s2,
                                 BLOCK_N: tl.constexpr):
    # program ids
    b = tl.program_id(0)
    p = tl.program_id(1)
    d_block = tl.program_id(2)

    d = d_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_d = d < D2

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # loop over k in 0..D-1
    for k in range(0, D2):
        a_ptr = A_ptr + b * A_s0 + p * A_s1 + k * A_s2
        a_val = tl.load(a_ptr)
        a_val = a_val.to(tl.float32)

        b_ptr = B_ptr + k * B_s0 + d * B_s1
        b_vec = tl.load(b_ptr, mask=mask_d, other=0.0)
        b_vec = b_vec.to(tl.float32)

        acc += a_val * b_vec

    out_ptr = Out_ptr + b * Out_s0 + p * Out_s1 + d * Out_s2
    tl.store(out_ptr, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes:
        # hidden_states: [B, I, D]
        # encoder_hidden_states: [B, T, D]
        # process_weight: [D, D]
        # Outputs:
        # processed_encoder: [B, T, D]
        # processed_hidden: [B, I, D]

        B, I, D = hidden_states.shape
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == D, "Shape mismatch"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # Ensure contiguous for predictable strides
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        WT = process_weight.contiguous()  # [D, D], note: process_weight.T would be [D, D] already

        processed_encoder = torch.empty((B, enc.shape[1], D), dtype=enc.dtype, device=enc.device)
        processed_hidden = torch.empty((B, hst.shape[1], D), dtype=hst.dtype, device=hst.device)

        # Launch kernel for encoder stream: [B, T, D] @ [D, D] -> [B, T, D]
        T = enc.shape[1]
        grid_enc = (B, T, triton.cdiv(D, 64))  # BLOCK_N is 64 here; safe default
        _gemm_row_reduce_ABD_kernel[grid_enc](
            enc, WT, processed_encoder,
            B, T, D,
            enc.stride(0), enc.stride(1), enc.stride(2),
            WT.stride(0), WT.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_N=64,
            num_warps=4,  # modest parallelism
            num_stages=2,
        )

        # Launch kernel for hidden stream: [B, I, D] @ [D, D] -> [B, I, D]
        I2 = hst.shape[1]
        grid_hid = (B, I2, triton.cdiv(D, 64))
        _gemm_row_reduce_AI_D_kernel[grid_hid](
            hst, WT, processed_hidden,
            B, I2, D,
            hst.stride(0), hst.stride(1), hst.stride(2),
            WT.stride(0), WT.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
