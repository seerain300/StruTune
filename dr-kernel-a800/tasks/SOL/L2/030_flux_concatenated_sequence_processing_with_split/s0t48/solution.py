import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    encoder_ptr, hidden_ptr, dst_ptr,
    B, L_txt, L_img, D,
    enc_stride_b, enc_stride_s, enc_stride_d,
    hdn_stride_b, hdn_stride_s, hdn_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
):
    # 2D grid: (b, t), where t in [0, L_txt + L_img)
    b = tl.program_id(0)
    t = tl.program_id(1)

    # Compute length
    M = L_txt + L_img

    # Only operate if b < B and t < M
    if (b >= B) or (t >= M):
        return

    # Determine source
    if t < L_txt:
        src_ptr = encoder_ptr + b * enc_stride_b + t * enc_stride_s
    else:
        t_src = t - L_txt
        src_ptr = hidden_ptr + b * hdn_stride_b + t_src * hdn_stride_s

    # Destination
    dst_ptr_out = dst_ptr + b * dst_stride_b + t * dst_stride_s

    # Copy D elements
    # We assume D is reasonably small; loop over d
    for d in range(0, D):
        val = tl.load(src_ptr + d * enc_stride_d)  # enc_stride_d/ hdn_stride_d matches d index
        tl.store(dst_ptr_out + d * dst_stride_d, val)


@triton.jit
def per_elem_matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    B, M, D,
    A_stride_b, A_stride_m, A_stride_k,
    W_stride_i, W_stride_j,
    C_stride_b, C_stride_m, C_stride_n,
):
    # 3D grid: (b, m, n)
    b = tl.program_id(0)
    m = tl.program_id(1)
    n = tl.program_id(2)

    # Accumulator in float32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over K dimension
    k = 0
    while k < D:
        # Load A[b, m, k] and W[k, n]
        a_val = tl.load(A_ptr + b * A_stride_b + m * A_stride_m + k * A_stride_k)
        w_val = tl.load(W_ptr + k * W_stride_i + n * W_stride_j)
        acc += a_val * w_val
        k += 1

    # Store result
    tl.store(C_ptr + b * C_stride_b + m * C_stride_m + n * C_stride_n, acc)


@triton.jit
def split_seqs_kernel(
    C_ptr, out1_ptr, out2_ptr,
    B, L_txt, L_img, D,
    C_stride_b, C_stride_s, C_stride_d,
    out1_stride_b, out1_stride_s, out1_stride_d,
    out2_stride_b, out2_stride_s, out2_stride_d,
):
    # Grid: (b, t, n)
    b = tl.program_id(0)
    t = tl.program_id(1)
    n = tl.program_id(2)

    # Only copy if within bounds
    if (b >= B) or (t >= (L_txt + L_img)) or (n >= D):
        return

    src_ptr = C_ptr + b * C_stride_b + t * C_stride_s + n * C_stride_d

    # Destination index for split:
    # processed_encoder has shape [B, L_txt, D], processed_hidden [B, L_img, D]
    if t < L_txt:
        dst_ptr = out1_ptr + b * out1_stride_b + t * out1_stride_s + n * out1_stride_d
    else:
        t2 = t - L_txt
        dst_ptr = out2_ptr + b * out2_stride_b + t2 * out2_stride_s + n * out2_stride_d

    val = tl.load(src_ptr)
    tl.store(dst_ptr, val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version that performs:
        1) Concatenation along sequence dimension,
        2) Linear projection using Triton per-element GEMM,
        3) Splitting into two streams.

        Returns:
            processed_encoder: [B, L_txt, D]
            processed_hidden:  [B, L_img, D]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        assert hidden_states.dtype == encoder_hidden_states.dtype == process_weight.dtype, "All tensors must have the same dtype"

        B = hidden_states.shape[0]
        D = hidden_states.shape[2]  # feature dimension
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        M = L_txt + L_img

        # Allocate and perform concatenation: [B, M, D]
        concatenated = torch.empty((B, M, D), dtype=hidden_states.dtype, device=hidden_states.device)

        grid_concat = (B, M)
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, L_txt, L_img, D,
            *encoder_hidden_states.stride(),
            *hidden_states.stride(),
            *concatenated.stride(),
            num_warps=1, num_stages=1,
        )

        # Ensure process_weight is [D, D]; original code uses process_weight [D, D] which matches
        # Now perform matmul: C = concatenated @ process_weight.T
        # concatenated shape: [B, M, D], weight: [D, D], result: [B, M, D]
        C = torch.empty((B, M, D), dtype=torch.float32, device=hidden_states.device)

        grid_matmul = (B, M, D)
        per_elem_matmul_kernel[grid_matmul](
            concatenated, process_weight, C,
            B, M, D,
            *concatenated.stride(),
            process_weight.stride(0), process_weight.stride(1),
            *C.stride(),
            num_warps=1, num_stages=1,
        )

        # Split into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, L_txt, D), dtype=hidden_states.dtype, device=hidden_states.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=hidden_states.dtype, device=hidden_states.device)

        grid_split = (B, L_txt, D)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            *C.stride(),
            *processed_encoder.stride(),
            *processed_hidden.stride(),
            num_warps=1, num_stages=1,
        )

        # If desired, cast back to original dtype; typically C is float32. The original outputs are dtype of inputs.
        # Here we keep outputs as float32 (the matmul result). In many pipelines, final output dtype can be inferred.
        # Since the original function returns tensors of the same dtype as inputs, we can cast:
        if processed_encoder.dtype != hidden_states.dtype:
            processed_encoder = processed_encoder.to(hidden_states.dtype)
        if processed_hidden.dtype != hidden_states.dtype:
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
