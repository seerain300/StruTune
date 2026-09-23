import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


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

    if (b >= B) or (t >= (L_txt + L_img)) or (d >= D):
        return

    is_encoder = t < L_txt
    if is_encoder:
        off = b * ehs_stride_b + t * ehs_stride_s + d * ehs_stride_d
        val = tl.load(ehs_ptr + off)
        dst_off = b * dst_stride_b + t * dst_stride_s + d * dst_stride_d
        tl.store(dst_ptr + dst_off, val)
    else:
        off = b * hs_stride_b + (t - L_txt) * hs_stride_s + d * hs_stride_d
        val = tl.load(hs_ptr + off)
        dst_off = b * dst_stride_b + t * dst_stride_s + d * dst_stride_d
        tl.store(dst_ptr + dst_off, val)


# Triton GEMM: C = A @ W, where
# A: [B, M, D] (concatenated), W: [D, D] (process_weight.T), C: [B, M, D]
# Each program computes one tile C[b, m_block, n_block].
@triton.jit
def matmul_gemm_kernel(
    A_ptr, W_ptr, C_ptr,
    B, M, D,
    A_stride_b, A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr,  # tile along sequence length
    BLOCK_N: tl.constexpr,  # tile along hidden dim
    BLOCK_K: tl.constexpr,  # reduction tile along K
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    valid_m = m_offsets < M
    valid_n = n_offsets < D
    mask_out = valid_m[:, None] & valid_n[None, :]  # [BLOCK_M, BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, D, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        valid_k = k_offsets < D

        # Load A[b, m, k]
        A_addr = A_ptr + pid_b * A_stride_b + (m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k)
        A_mask = valid_m[:, None] & valid_k[None, :]
        A_vals = tl.load(A_addr, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K], compute in f32

        # Load W[k, n] as [BLOCK_K, BLOCK_N]
        W_addr = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        W_mask = valid_k[:, None] & valid_n[None, :]
        W_vals = tl.load(W_addr, mask=W_mask, other=0.0)  # [BLOCK_K, BLOCK_N], f32

        # Accumulate in f32
        acc += tl.dot(A_vals, W_vals)  # [BLOCK_M, BLOCK_N], f32

    # Store results to C[b, m, n] (cast happens implicitly based on C_ptr dtype)
    C_addr = C_ptr + pid_b * C_stride_b + (m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n)
    tl.store(C_addr, acc, mask=mask_out)


# Triton kernel: split processed into two outputs along sequence dim.
# processed_encoder = processed[:, :L_txt, :], processed_hidden = processed[:, L_txt:, :]
@triton.jit
def split_kernel(
    processed_ptr, out1_ptr, out2_ptr,
    B, L_txt, L_img, D,
    processed_stride_b, processed_stride_s, processed_stride_d,
    out1_stride_b, out1_stride_s, out1_stride_d,
    out2_stride_b, out2_stride_s, out2_stride_d,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)

    if (b >= B) or (s >= L_txt) or (d >= D):
        return
    # Copy encoder part
    off = b * processed_stride_b + s * processed_stride_s + d * processed_stride_d
    val1 = tl.load(processed_ptr + off)
    out_off1 = b * out1_stride_b + s * out1_stride_s + d * out1_stride_d
    tl.store(out1_ptr + out_off1, val1)

    if (b >= B) or (s >= L_img) or (d >= D):
        return
    # Copy hidden part
    off2 = b * processed_stride_b + (s + L_txt) * processed_stride_s + d * processed_stride_d
    val2 = tl.load(processed_ptr + off2)
    out_off2 = b * out2_stride_b + s * out2_stride_s + d * out2_stride_d
    tl.store(out2_ptr + out_off2, val2)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Use Triton kernels; do not use torch ops on tensors in forward.
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available, but the environment should provide Triton.")

        # Ensure inputs are on same device and dtype
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        M = L_txt + L_img

        # Make inputs contiguous for simpler strides and better performance
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        W_T = process_weight.t().contiguous()  # [D, D]

        # 1) Concatenate in Triton: dst [B, M, D]
        dst = torch.empty((B, M, D), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_concat = (B, M, D)
        concat_kernel[grid_concat](
            ehs, hs, dst,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            dst.stride(0), dst.stride(1), dst.stride(2),
            num_warps=2, num_stages=1
        )

        # 2) Matmul in Triton: processed = dst @ W_T
        processed = torch.empty((B, M, D), device=hidden_states.device, dtype=hidden_states.dtype)
        # Tile sizes chosen for robustness across varied shapes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))
        matmul_gemm_kernel[grid_gemm](
            dst, W_T, processed,
            B, M, D,
            dst.stride(0), dst.stride(1), dst.stride(2),  # A strides
            W_T.stride(0), W_T.stride(1),                # W strides: [D, D]
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split in Triton: processed_encoder [B, L_txt, D], processed_hidden [B, L_img, D]
        processed_encoder = torch.empty((B, L_txt, D), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=hidden_states.device, dtype=hidden_states.dtype)

        grid_split = (B, L_txt, D)
        split_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=2, num_stages=1
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
