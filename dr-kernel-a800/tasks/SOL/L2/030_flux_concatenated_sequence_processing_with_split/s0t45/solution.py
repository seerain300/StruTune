import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,            # *f32 [B, L_txt, D]
    hs_ptr,             # *f32 [B, L_img, D]
    dst_ptr,            # *f32 [B, M, D], M = L_txt + L_img
    B: tl.constexpr,
    L_txt: tl.constexpr,
    L_img: tl.constexpr,
    D: tl.constexpr,
    ehs_stride_b, ehs_stride_s, ehs_stride_d,
    hs_stride_b, hs_stride_s, hs_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
):
    b = tl.program_id(0)
    m = tl.program_id(1)  # 0..(L_txt+L_img-1)
    d = tl.program_id(2)  # 0..(D-1)

    if m < L_txt:
        src_ptr = ehs_ptr + b * ehs_stride_b + m * ehs_stride_s + d * ehs_stride_d
    else:
        src_m = m - L_txt
        src_ptr = hs_ptr + b * hs_stride_b + src_m * hs_stride_s + d * hs_stride_d

    dst_ptr_out = dst_ptr + b * dst_stride_b + m * dst_stride_s + d * dst_stride_d
    val = tl.load(src_ptr)
    tl.store(dst_ptr_out, val)


@triton.jit
def gmatmul_elementwise_kernel(
    A_ptr,  # *f32 [B, M, D] input left (concatenated)
    W_ptr,  # *f32 [D, D]
    C_ptr,  # *f32 [B, M, D] output
    B: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    A_stride_b, A_stride_m, A_stride_d,
    C_stride_b, C_stride_m, C_stride_d,
    K_CHUNK: tl.constexpr,
):
    # Each program handles one output element (b, m, n)
    b = tl.program_id(0)
    m = tl.program_id(1)
    n = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over K (feature dim) in chunks
    for k0 in range(0, D, K_CHUNK):
        k_range = k0 + tl.arange(0, K_CHUNK)
        mask_k = k_range < D

        # Load A[b, m, k]
        A_vec = tl.load(
            A_ptr + b * A_stride_b + m * A_stride_m + k_range * A_stride_d,
            mask=mask_k,
            other=0.0
        )  # shape [K_CHUNK]

        # Load W[k, n] (note: W is [D, D], we index row k, col n)
        W_vec = tl.load(
            W_ptr + k_range * D + n,
            mask=mask_k,
            other=0.0
        )  # shape [K_CHUNK]

        # Accumulate dot product for this chunk
        acc += tl.sum(A_vec * W_vec, axis=0)

    # Store result to C[b, m, n]
    dst_ptr = C_ptr + b * C_stride_b + m * C_stride_m + n * C_stride_d
    tl.store(dst_ptr, acc)


@triton.jit
def split_seqs_kernel(
    src_ptr,             # *T [B, M, D]
    out1_ptr,            # *T [B, L_txt, D]
    out2_ptr,            # *T [B, L_img, D]
    B: tl.constexpr,
    L_txt: tl.constexpr,
    L_img: tl.constexpr,
    D: tl.constexpr,
    src_stride_b, src_stride_m, src_stride_d,
    out1_stride_b, out1_stride_s, out1_stride_d,
    out2_stride_b, out2_stride_s, out2_stride_d,
):
    b = tl.program_id(0)
    # Copy first L_txt rows to out1
    for t in range(0, L_txt):
        src_ptr_t = src_ptr + b * src_stride_b + t * src_stride_m
        out1_ptr_t = out1_ptr + b * out1_stride_b + t * out1_stride_s
        val = tl.load(src_ptr_t)
        tl.store(out1_ptr_t, val)

    # Copy next L_img rows to out2 (shifted by L_txt)
    for t in range(0, L_img):
        src_ptr_t = src_ptr + b * src_stride_b + (L_txt + t) * src_stride_m
        out2_ptr_t = out2_ptr + b * out2_stride_b + t * out2_stride_s
        val = tl.load(src_ptr_t)
        tl.store(out2_ptr_t, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton implementation:
        - Concatenate along sequence dimension using Triton
        - Perform matmul via Triton elementwise kernel (compute C[b, m, n] = sum_k A[b, m, k] * W[k, n])
        - Split back into encoder and image streams using Triton
        """
        # Ensure CUDA tensors
        if not hidden_states.is_cuda or not encoder_hidden_states.is_cuda or not process_weight.is_cuda:
            raise RuntimeError("ModelNew.forward requires CUDA tensors")

        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D, "Mismatched hidden_dim"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # Make inputs contiguous and use float32 for robustness
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Step 1: Triton Concatenate along sequence dimension into A [B, M, D], M = L_txt + L_img
        M = L_txt + L_img
        A = torch.empty((B, M, D), device=ehs.device, dtype=torch.float32)

        ehs_stride = ehs.stride()  # (b, s, d)
        hs_stride = hs.stride()
        A_stride = A.stride()

        grid_concat = (B, M, D)
        concat_seqs_kernel[grid_concat](
            ehs, hs, A,
            B, L_txt, L_img, D,
            ehs_stride[0], ehs_stride[1], ehs_stride[2],
            hs_stride[0], hs_stride[1], hs_stride[2],
            A_stride[0], A_stride[1], A_stride[2],
            num_warps=1, num_stages=1
        )

        # Step 2: Triton GEMM: C[b, m, n] = sum_k A[b, m, k] * W[k, n]
        C = torch.empty((B, M, D), device=A.device, dtype=torch.float32)

        C_stride = C.stride()
        A_stride_used = A.stride()

        # Choose chunk size for K reduction. 64 works well for typical D; loop covers all D.
        K_CHUNK = 64

        grid_gemm = (B, M, D)
        gmatmul_elementwise_kernel[grid_gemm](
            A, W, C,
            B, M, D,
            A_stride_used[0], A_stride_used[1], A_stride_used[2],
            C_stride[0], C_stride[1], C_stride[2],
            K_CHUNK=K_CHUNK,
            num_warps=4, num_stages=1
        )

        # Step 3: Triton Split outputs
        processed_encoder = torch.empty((B, L_txt, D), device=C.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, L_img, D), device=C.device, dtype=torch.float32)

        C_stride_used = C.stride()
        out1_stride = processed_encoder.stride()
        out2_stride = processed_hidden.stride()

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            C_stride_used[0], C_stride_used[1], C_stride_used[2],
            out1_stride[0], out1_stride[1], out1_stride[2],
            out2_stride[0], out2_stride[1], out2_stride[2],
            num_warps=1, num_stages=1
        )

        # Cast back to original dtype if necessary
        if hidden_states.dtype != torch.float32:
            processed_encoder = processed_encoder.to(hidden_states.dtype)
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
