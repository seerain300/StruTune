import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton GEMM: C = A @ W
# A: [B, M, D] (concatenated), W: [D, D] (process_weight.T), C: [B, M, D]
@triton.jit
def matmul_gemm_kernel(
    A_ptr, W_ptr, C_ptr,
    B, M, D,
    A_stride_b, A_stride_m, A_stride_d,
    W_stride_k, W_stride_d,
    C_stride_b, C_stride_m, C_stride_d,
    BLOCK_M: tl.constexpr,  # tile along M (sequence length)
    BLOCK_N: tl.constexpr,  # tile along N (hidden dim)
    BLOCK_K: tl.constexpr,  # reduction tile along K (hidden dim)
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    valid_m = m_offsets < M
    valid_n = n_offsets < D
    mask_mn = valid_m[:, None] & valid_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, D, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        valid_k = k_offsets < D

        # Load A[b, m, k]
        A_addr = A_ptr + pid_b * A_stride_b + (m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_d)
        A_mask = valid_m[:, None] & valid_k[None, :]
        A_vals = tl.load(A_addr, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        A_vals = A_vals.to(tl.float32)

        # Load W[k, n] (process_weight.T)
        W_addr = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_d  # [BLOCK_K, BLOCK_N]
        W_mask = valid_k[:, None] & valid_n[None, :]
        W_vals = tl.load(W_addr, mask=W_mask, other=0.0)  # [BLOCK_K, BLOCK_N]
        W_vals = W_vals.to(tl.float32)

        # Accumulate: acc += A_vals @ W_vals
        acc += tl.dot(A_vals, W_vals)

    # Store result
    C_addr = C_ptr + pid_b * C_stride_b + (m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_d)
    tl.store(C_addr, acc, mask=mask_mn)


# Triton kernel: concatenate encoder_hidden_states and hidden_states along sequence dimension.
# dst[b, t, d] = encoder_hidden_states[b, t, d] if t < L_txt else hidden_states[b, t - L_txt, d]
@triton.jit
def concat_kernel(
    ehs_ptr, hs_ptr, dst_ptr,
    B, L_txt, L_img, D,
    ehs_stride_b, ehs_stride_s, ehs_stride_d,
    hs_stride_b, hs_stride_s, hs_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)
    if b >= B:
        return
    is_encoder = t < L_txt
    if is_encoder:
        ehs_off = b * ehs_stride_b + t * ehs_stride_s + d * ehs_stride_d
        val = tl.load(ehs_ptr + ehs_off)
        dst_off = b * dst_stride_b + t * dst_stride_s + d * dst_stride_d
        tl.store(dst_ptr + dst_off, val)
    else:
        hs_idx = t - L_txt
        hs_off = b * hs_stride_b + hs_idx * hs_stride_s + d * hs_stride_d
        val = tl.load(hs_ptr + hs_off)
        dst_off = b * dst_stride_b + t * dst_stride_s + d * dst_stride_d
        tl.store(dst_ptr + dst_off, val)


# Triton kernel: copy src[:, :L_txt, :] to dst_encoder
@triton.jit
def split_encoder_kernel(
    src_ptr, dst_ptr,
    B, L_txt, D,
    src_stride_b, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)
    if b >= B or s >= L_txt:
        return
    src_off = b * src_stride_b + s * src_stride_s + d * src_stride_d
    dst_off = b * dst_stride_b + s * dst_stride_s + d * dst_stride_d
    val = tl.load(src_ptr + src_off)
    tl.store(dst_ptr + dst_off, val)


# Triton kernel: copy src[:, L_txt:, :] to dst_hidden
@triton.jit
def split_hidden_kernel(
    src_ptr, dst_ptr,
    B, L_img, D,
    src_stride_b, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)
    if b >= B or s >= L_img:
        return
    # src index along sequence is L_txt + s
    src_off = b * src_stride_b + (L_txt + s) * src_stride_s + d * src_stride_d
    dst_off = b * dst_stride_b + s * dst_stride_s + d * dst_stride_d
    val = tl.load(src_ptr + src_off)
    tl.store(dst_ptr + dst_off, val)


@torch.no_grad()
def run_triton(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-optimized version of the original run:
    1) Concatenate encoder_hidden_states and hidden_states along sequence dimension.
    2) Apply linear projection via Triton GEMM: concatenated @ process_weight.T.
    3) Split back into separate encoder and image streams.
    """
    # Ensure inputs are CUDA tensors for Triton
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."

    B = hidden_states.shape[0]
    D = hidden_states.shape[2]
    L_txt = encoder_hidden_states.shape[1]
    L_img = hidden_states.shape[1]
    M = L_txt + L_img

    # 1) Concatenate in Triton
    dst = torch.empty((B, M, D), dtype=hidden_states.dtype, device=hidden_states.device)

    # Ensure inputs are contiguous for simpler indexing
    ehs = encoder_hidden_states.contiguous()
    hs = hidden_states.contiguous()

    grid_concat = (B, M, D)
    concat_kernel[grid_concat](
        ehs, hs, dst,
        B, L_txt, L_img, D,
        ehs.stride(0), ehs.stride(1), ehs.stride(2),
        hs.stride(0), hs.stride(1), hs.stride(2),
        dst.stride(0), dst.stride(1), dst.stride(2),
        num_warps=1, num_stages=1
    )

    # 2) Triton GEMM: C = dst @ process_weight.T
    # Make W contiguous and ensure dtype is float32 for stable accumulation
    W = process_weight.t().contiguous()  # [D, D]
    C = torch.empty((B, M, D), dtype=torch.float32, device=hidden_states.device)  # accumulate in fp32

    # Choose tile sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))
    matmul_gemm_kernel[grid_gemm](
        dst, W, C,
        B, M, D,
        dst.stride(0), dst.stride(1), dst.stride(2),
        W.stride(0), W.stride(1),  # W is [D, D], contiguous
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )

    # Cast output back to original dtype
    processed = C.to(hidden_states.dtype)

    # 3) Split in Triton
    processed_encoder = torch.empty((B, L_txt, D), dtype=hidden_states.dtype, device=hidden_states.device)
    processed_hidden = torch.empty((B, L_img, D), dtype=hidden_states.dtype, device=hidden_states.device)

    # Ensure processed is contiguous
    processed = processed.contiguous()

    grid_encoder = (B, L_txt, D)
    split_encoder_kernel[grid_encoder](
        processed, processed_encoder,
        B, L_txt, D,
        processed.stride(0), processed.stride(1), processed.stride(2),
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        num_warps=1, num_stages=1
    )

    grid_hidden = (B, L_img, D)
    split_hidden_kernel[grid_hidden](
        processed, processed_hidden,
        B, L_img, D,
        processed.stride(0), processed.stride(1), processed.stride(2),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        num_warps=1, num_stages=1
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Triton path: ensure tensors are on CUDA
        if not hidden_states.is_cuda or not encoder_hidden_states.is_cuda or not process_weight.is_cuda:
            # Fallback to original PyTorch path if CUDA tensors are not provided
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            processed = torch.matmul(concatenated, process_weight.t())
            processed_encoder = processed[:, :encoder_hidden_states.shape[1], :]
            processed_hidden = processed[:, encoder_hidden_states.shape[1]:, :]
            return processed_encoder, processed_hidden
        # Triton path
        return run_triton(hidden_states, encoder_hidden_states, process_weight)


def run(*args):
    return ModelNew()(*args)
