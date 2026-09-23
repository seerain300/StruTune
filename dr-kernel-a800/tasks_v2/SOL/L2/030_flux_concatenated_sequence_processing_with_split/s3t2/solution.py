import torch

# Triton import guarded; evaluation runs on GPU
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Concatenate encoder_hidden_states [B, T, D] and hidden_states [B, I, D] into out [B, P, D], P=T+I
@triton.jit
def _concatenate_seq_kernel(
    enc_ptr,         # *const T, [B, T, D]
    img_ptr,         # *const T, [B, I, D]
    out_ptr,         # *T,       [B, P, D]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    stride_eb, stride_et, stride_ed,
    stride_ib, stride_it, stride_id,
    stride_ob, stride_op, stride_od,
    BLOCK_P: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    p_offsets = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = p_offsets < P

    from_encoder = p_offsets < T
    p_in_img = p_offsets - T

    for d in range(0, D):
        enc_addrs = enc_ptr + pid_b * stride_eb + p_offsets[None, :] * stride_et + d * stride_ed
        vals_enc = tl.load(enc_addrs, mask=mask_p[None, :] & from_encoder[None, :], other=0.0)

        img_addrs = img_ptr + pid_b * stride_ib + p_in_img[None, :] * stride_it + d * stride_id
        vals_img = tl.load(img_addrs, mask=mask_p[None, :] & ~from_encoder[None, :], other=0.0)

        vals = tl.where(from_encoder[None, :], vals_enc, vals_img)

        out_addrs = out_ptr + pid_b * stride_ob + p_offsets[None, :] * stride_op + d * stride_od
        tl.store(out_addrs, vals, mask=mask_p[None, :])


# Kernel 2: GEMM out = X [B, P, D] @ W_T [D, D], result Y [B, P, D]
@triton.jit
def _gemm_kernel(
    X_ptr,           # *const T, [B, P, D]
    W_ptr,           # *const T, [D, D]
    Y_ptr,           # *T,       [B, P, D]
    B: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xp, stride_xd,
    stride_wk, stride_wd,
    stride_yb, stride_yp, stride_yd,
    BLOCK_N: tl.constexpr,  # feature tile
    BLOCK_P: tl.constexpr,  # sequence tile
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    p_offsets = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)

    for k in range(0, D):
        x_ptrs = X_ptr + pid_b * stride_xb + p_offsets[:, None] * stride_xp + k * stride_xd
        x_mask = (p_offsets[:, None] < P)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + k * stride_wk + n_offsets[None, :] * stride_wd
        w_mask = (n_offsets[None, :] < D)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += x * w

    y_ptrs = Y_ptr + pid_b * stride_yb + p_offsets[:, None] * stride_yp + n_offsets[None, :] * stride_yd
    y_mask = (p_offsets[:, None] < P) & (n_offsets[None, :] < D)
    tl.store(y_ptrs, acc, mask=y_mask)


# Kernel 3: Copy slice from Y [B, P, D] into processed_encoder [B, T, D] at columns [0:T]
@triton.jit
def _copy_slice_kernel(
    src_ptr,         # *const T, [B, P, D]
    dst_ptr,         # *T,       [B, T, D]
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    stride_sb, stride_sp, stride_sd,
    stride_db, stride_dt, stride_dd,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    d_offsets = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    t = pid_t
    mask_d = d_offsets < D
    src_addrs = src_ptr + pid_b * stride_sb + t * stride_sp + d_offsets * stride_sd
    dst_addrs = dst_ptr + pid_b * stride_db + t * stride_dt + d_offsets * stride_dd
    vals = tl.load(src_addrs, mask=mask_d, other=0.0)
    tl.store(dst_addrs, vals, mask=mask_d)


# Kernel 4: Copy slice from Y [B, P, D] into processed_hidden [B, I, D] at columns [T:T+I]
@triton.jit
def _copy_slice_kernel_img(
    src_ptr,         # *const T, [B, P, D]
    dst_ptr,         # *T,       [B, I, D]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    stride_sb, stride_sp, stride_sd,
    stride_ib, stride_it, stride_id,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    d_offsets = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    i = pid_i
    src_col = T + i
    mask_d = d_offsets < D
    src_addrs = src_ptr + pid_b * stride_sb + src_col * stride_sp + d_offsets * stride_sd
    dst_addrs = dst_ptr + pid_b * stride_ib + i * stride_it + d_offsets * stride_id
    vals = tl.load(src_addrs, mask=mask_d, other=0.0)
    tl.store(dst_addrs, vals, mask=mask_d)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenation along the sequence dimension is done via Triton.
        - Linear projection (matmul) is done via a Triton GEMM kernel.
        - Slicing into processed_encoder and processed_hidden is done via Triton copy kernels.
        Forward does only shape/stride/grid/allocations and launches Triton kernels; no torch ops in the hot path.
        """
        # Fallback if Triton not available or tensors not on CUDA
        if (not TRITON_AVAILABLE) or (not hidden_states.is_cuda) or (not encoder_hidden_states.is_cuda) or (not process_weight.is_cuda):
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            processed = torch.matmul(concatenated, process_weight.t())
            processed_encoder = processed[:, :encoder_hidden_states.shape[1], :]
            processed_hidden = processed[:, encoder_hidden_states.shape[1]:, :]
            return processed_encoder, processed_hidden

        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        P = T + I

        # 1) Concatenate into concatenated_in [B, P, D] using Triton
        enc = encoder_hidden_states.contiguous()  # [B, T, D]
        img = hidden_states.contiguous()         # [B, I, D]
        concatenated_in = torch.empty((B, P, D), device=enc.device, dtype=enc.dtype)

        BLOCK_P = 128
        grid_concat = (B, triton.cdiv(P, BLOCK_P))
        _concatenate_seq_kernel[grid_concat](
            enc, img, concatenated_in,
            B, T, I, D, P,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            concatenated_in.stride(0), concatenated_in.stride(1), concatenated_in.stride(2),
            BLOCK_P=BLOCK_P,
        )

        # 2) GEMM: out = concatenated_in @ process_weight.T, result [B, P, D]
        # Ensure weight is transposed: W_T [D, D]
        W_T = process_weight.t().contiguous()  # [D, D]
        out = torch.empty((B, P, D), device=enc.device, dtype=enc.dtype)

        BLOCK_N = 128
        BLOCK_P_T = 128
        grid_gemm = (B, triton.cdiv(P, BLOCK_P_T), triton.cdiv(D, BLOCK_N))
        _gemm_kernel[grid_gemm](
            concatenated_in, W_T, out,
            B, P, D,
            concatenated_in.stride(0), concatenated_in.stride(1), concatenated_in.stride(2),
            W_T.stride(0), W_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_N=BLOCK_N, BLOCK_P=BLOCK_P_T,
            num_warps=4,
        )

        # 3) Split into processed_encoder [B, T, D] and processed_hidden [B, I, D] using Triton copy kernels
        processed_encoder = torch.empty((B, T, D), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, D), device=enc.device, dtype=enc.dtype)

        BLOCK_D = 64
        grid_e = (B, T, triton.cdiv(D, BLOCK_D))
        _copy_slice_kernel[grid_e](
            out, processed_encoder,
            B, T, D, P,
            out.stride(0), out.stride(1), out.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )

        grid_i = (B, I, triton.cdiv(D, BLOCK_D))
        _copy_slice_kernel_img[grid_i](
            out, processed_hidden,
            B, T, I, D, P,
            out.stride(0), out.stride(1), out.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
