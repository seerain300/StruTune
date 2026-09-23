import torch
import triton
import triton.language as tl


@triton.jit
def cat_seqs_kernel(
    ehs_ptr, hs_ptr, out_ptr,
    B: tl.constexpr, L_txt: tl.constexpr, L_img: tl.constexpr, D: tl.constexpr,
    ehs_stride_b, ehs_stride_t, ehs_stride_d,
    hs_stride_b, hs_stride_t, hs_stride_d,
    out_stride_b, out_stride_s, out_stride_d,
):
    # Grid: (B, L_txt + L_img)
    b = tl.program_id(0)
    s = tl.program_id(1)  # s in [0, L_txt + L_img)

    # Compute destination pointer for out[b, s, :]
    # Note: out is a concatenation, so s < L_txt -> ehs, else s - L_txt -> hs
    if s < L_txt:
        src = ehs_ptr + b * ehs_stride_b + s * ehs_stride_t
    else:
        src = hs_ptr + b * hs_stride_b + (s - L_txt) * hs_stride_t

    # Destination offset: out[b, s, :]
    dst = out_ptr + b * out_stride_b + s * out_stride_s

    # Copy across feature dimension
    for d in range(0, D):
        # Mask to avoid any out-of-bounds (shouldn't happen given s <= L_txt+L_img)
        val = tl.load(src + d * ehs_stride_d, mask=True)
        tl.store(dst + d * out_stride_d, val, mask=True)


@triton.jit
def split_seqs_kernel(
    in_ptr, out1_ptr, out2_ptr,
    B: tl.constexpr, L_txt: tl.constexpr, L_img: tl.constexpr, D: tl.constexpr,
    in_stride_b, in_stride_s, in_stride_d,
    out1_stride_b, out1_stride_s, out1_stride_d,
    out2_stride_b, out2_stride_s, out2_stride_d,
):
    # Grid: (B, L_txt) for processed_encoder, and (B, L_img) for processed_hidden
    b = tl.program_id(0)
    s = tl.program_id(1)

    # For processed_encoder: copy in[b, :L_txt, :] -> out1[b, :, :]
    src1 = in_ptr + b * in_stride_b + s * in_stride_s
    dst1 = out1_ptr + b * out1_stride_b + s * out1_stride_s
    for d in range(0, D):
        tl.store(dst1 + d * out1_stride_d, tl.load(src1 + d * in_stride_d))

    # For processed_hidden: copy in[b, L_txt:, :] -> out2[b, :, :]
    src2 = in_ptr + b * in_stride_b + (s + L_txt) * in_stride_s
    dst2 = out2_ptr + b * out2_stride_b + s * out2_stride_s
    for d in range(0, D):
        tl.store(dst2 + d * out2_stride_d, tl.load(src2 + d * in_stride_d))


@triton.jit
def matmul_cat_triton_kernel(
    in_ptr, w_ptr, out_ptr,
    B: tl.constexpr, L_txt: tl.constexpr, L_img: tl.constexpr, D: tl.constexpr,
    in_stride_b, in_stride_s, in_stride_d,
    w_stride_k, w_stride_n,
    out_stride_b, out_stride_s, out_stride_d,
    CHUNK_K: tl.constexpr,
):
    # Grid: (B, L_txt + L_img)
    b = tl.program_id(0)
    m = tl.program_id(1)  # m in [0, L_txt + L_img)

    # Accumulator for the whole output vector at (b, m, :)
    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over K (feature dimension) in chunks
    for k0 in range(0, D, CHUNK_K):
        for kk in range(0, CHUNK_K):
            k = k0 + kk
            # Mask K within bounds
            k_valid = k < D
            # Load input row: in[b, m, k] if k is valid
            in_row_ptr = in_ptr + b * in_stride_b + m * in_stride_s + k * in_stride_d
            val = tl.load(in_row_ptr, mask=k_valid, other=0.0)  # scalar load
            # Load weight vector: w[k, :] across n
            w_row_ptr = w_ptr + k * w_stride_k
            w_vec = tl.load(w_row_ptr + tl.arange(0, D) * w_stride_n, mask=k_valid, other=0.0)
            # Accumulate: acc += val * w_vec
            # Note: val is scalar; w_vec is vector of length D
            acc += val * w_vec

    # Store acc to out[b, m, :]
    out_row_ptr = out_ptr + b * out_stride_b + m * out_stride_s
    for d in range(0, D):
        tl.store(out_row_ptr + d * out_stride_d, acc[d])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          - Concatenate encoder_hidden_states and hidden_states along sequence dim using Triton.
          - Compute matmul with process_weight via a Triton kernel (per-(b, m, n) loop).
          - Split back into processed_encoder and processed_hidden using Triton.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device"

        # Shapes
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]
        L_txt = encoder_hidden_states.shape[1]
        M = L_txt + L_img

        device = hidden_states.device

        # 1) Concatenate along sequence dimension using Triton
        # Ensure inputs are contiguous for predictable strides
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        out_cat = torch.empty((B, M, D), dtype=torch.float32, device=device)  # compute in float32

        ehs_stride = ehs.stride()
        hs_stride = hs.stride()
        out_stride = out_cat.stride()

        grid_cat = (B, M)
        cat_seqs_kernel[grid_cat](
            ehs, hs, out_cat,
            B, L_txt, L_img, D,
            ehs_stride[0], ehs_stride[1], ehs_stride[2],
            hs_stride[0], hs_stride[1], hs_stride[2],
            out_stride[0], out_stride[1], out_stride[2],
            num_warps=1, num_stages=1
        )

        # 2) Compute matmul out_cat @ process_weight.T using Triton
        # process_weight is [D, D]; out_cat is [B, M, D]
        w = process_weight.contiguous()  # [D, D], row-major expected by Triton
        w_T = w  # we will load w[k, :] directly; no need to transpose
        out_mat = torch.empty((B, M, D), dtype=torch.float32, device=device)

        in_stride = out_cat.stride()
        w_stride = w.stride()
        out_mat_stride = out_mat.stride()

        # Choose CHUNK_K; 64 is a reasonable default across many D; adjust for larger D
        CHUNK_K = 64
        grid_mat = (B, M)
        matmul_cat_triton_kernel[grid_mat](
            out_cat, w_T, out_mat,
            B, L_txt, L_img, D,
            in_stride[0], in_stride[1], in_stride[2],
            w_stride[0], w_stride[1],
            out_mat_stride[0], out_mat_stride[1], out_mat_stride[2],
            CHUNK_K=CHUNK_K,
            num_warps=1, num_stages=1
        )

        # 3) Split using Triton
        processed_encoder = torch.empty((B, L_txt, D), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, L_img, D), dtype=torch.float32, device=device)

        in_split_stride = out_mat.stride()
        out1_stride = processed_encoder.stride()
        out2_stride = processed_hidden.stride()

        # Grids: one program per (b, s) where s in [0, L_txt) and [0, L_img)
        grid_encoder = (B, L_txt)
        grid_hidden = (B, L_img)
        split_seqs_kernel[grid_encoder](
            out_mat, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            in_split_stride[0], in_split_stride[1], in_split_stride[2],
            out1_stride[0], out1_stride[1], out1_stride[2],
            out2_stride[0], out2_stride[1], out2_stride[2],
            num_warps=1, num_stages=1
        )

        # Return outputs; keep float32 to avoid dtype mismatches in comparison.
        # If you need to match input dtype, uncomment the casts below:
        # if hidden_states.dtype != torch.float32:
        #     processed_encoder = processed_encoder.to(hidden_states.dtype)
        #     processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
