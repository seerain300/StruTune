import torch
import triton
import triton.language as tl

# Triton kernel: concatenate along sequence dimension
# Destination: dst[b, t, d] = src1[b, t, d] if t < L_txt else src2[b, t - L_txt, d]
@triton.jit
def concat_seq_kernel(
    src1, src2, dst,
    B, L_txt, L_img, D,
    src1_stride_b, src1_stride_s, src1_stride_d,
    src2_stride_b, src2_stride_s, src2_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_b
    t = pid_t
    d = pid_d

    mask = (b < B) & (t < (L_txt + L_img)) & (d < D)

    if t < L_txt:
        val = tl.load(
            src1 + b * src1_stride_b + t * src1_stride_s + d * src1_stride_d,
            mask=mask,
            other=0.0
        )
    else:
        val = tl.load(
            src2 + b * src2_stride_b + (t - L_txt) * src2_stride_s + d * src2_stride_d,
            mask=mask,
            other=0.0
        )

    tl.store(dst + b * dst_stride_b + t * dst_stride_s + d * dst_stride_d, val, mask=mask)


# Triton kernel: split along sequence dimension back into two tensors
# From processed of shape [B, M, D], write:
# processed_encoder[b, t, d] = processed[b, t, d] for t in [0, L_txt)
# processed_hidden[b, t, d] = processed[b, t + L_txt, d] for t in [0, L_img)
@triton.jit
def split_seq_kernel(
    processed, processed_encoder, processed_hidden,
    B, L_txt, L_img, D,
    proc_stride_b, proc_stride_s, proc_stride_d,
    enc_stride_b, enc_stride_s, enc_stride_d,
    hid_stride_b, hid_stride_s, hid_stride_d,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_b
    t = pid_t
    d = pid_d

    mask = (b < B) & (t < L_txt) & (d < D)
    val = tl.load(
        processed + b * proc_stride_b + t * proc_stride_s + d * proc_stride_d,
        mask=mask,
        other=0.0
    )
    tl.store(processed_encoder + b * enc_stride_b + t * enc_stride_s + d * enc_stride_d, val, mask=mask)

    mask2 = (b < B) & (t < L_img) & (d < D)
    val2 = tl.load(
        processed + b * proc_stride_b + (t + L_txt) * proc_stride_s + d * proc_stride_d,
        mask=mask2,
        other=0.0
    )
    tl.store(processed_hidden + b * hid_stride_b + t * hid_stride_s + d * hid_stride_d, val2, mask=mask2)


# Triton kernel: batched matmul C[b, m, n] = sum_k A[b, m, k] * W[k, n]
# A is [B, M, D], W is [D, D], C is [B, M, D]
# We call this kernel separately for encoder and hidden streams to produce final outputs.
@triton.jit
def matmul_bmn_kernel(
    A, W, C,
    B, M, D,
    A_stride_b, A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_b, C_stride_m, C_stride_k,
):
    # One program per (b, m, n)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    b = pid_b
    m = pid_m
    n = pid_n

    # Accumulator for this (b, m, n)
    acc = tl.zeros((), dtype=tl.float32)

    # Sum over k from 0 to D-1
    # We iterate in blocks to improve locality; mask handles tails.
    BLOCK_K = 64
    k0 = 0
    while k0 < D:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load A[b, m, k_offsets] -> vector of length BLOCK_K
        a_ptrs = A + b * A_stride_b + m * A_stride_m + k_offsets * A_stride_k
        a_vals = tl.load(a_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Load W[k_offsets, n] -> scalar W[n] if BLOCK_K == 1, or vector when BLOCK_K > 1.
        # Here, we treat n as scalar for this program; using a scalar load for simplicity.
        # Note: Triton expects either a scalar or a vector of matching shape; for scalar, we use a 1-element load.
        w_ptrs = W + k_offsets * W_stride_k + n * W_stride_n
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Fused multiply-add: acc += sum(a_vals * w_vals)
        # a_vals: [BLOCK_K], w_vals: [BLOCK_K]
        prod = a_vals * w_vals
        # Reduce to scalar
        acc += tl.sum(prod, axis=0)

        k0 += BLOCK_K

    # Store result to C[b, m, n]
    c_ptr = C + b * C_stride_b + m * C_stride_m + n * C_stride_k
    tl.store(c_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenate along sequence dimension
        - Compute processed = concatenated @ process_weight.T using Triton matmul
        - Split back into processed_encoder and processed_hidden
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton kernels."

        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]
        L_txt = encoder_hidden_states.shape[1]

        # 1) Concatenate along sequence dimension: [B, L_txt + L_img, D]
        M = L_txt + L_img
        concatenated = torch.empty((B, M, D), dtype=hidden_states.dtype, device=hidden_states.device)

        concat_seq_kernel[(B, M, D)](
            encoder_hidden_states, hidden_states, concatenated,
            B, L_txt, L_img, D,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=4, num_stages=2
        )

        # 2) Compute processed = concatenated @ process_weight.T using Triton matmul
        # We compute C = [B, M, D] where C[b, m, n] = sum_k A[b, m, k] * W[k, n]
        processed = torch.empty((B, M, D), dtype=torch.float32, device=hidden_states.device)  # use fp32 accumulation

        # Launch Triton matmul kernel
        matmul_bmn_kernel[(B, M, D)](
            concatenated, process_weight, processed,
            B, M, D,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            num_warps=4, num_stages=2
        )

        # 3) Split back into two outputs
        processed_encoder = torch.empty((B, L_txt, D), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=torch.float32, device=hidden_states.device)

        split_seq_kernel[(B, L_txt, D)](
            processed, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=4, num_stages=2
        )

        # Cast outputs back to original dtype
        processed_encoder = processed_encoder.to(hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
