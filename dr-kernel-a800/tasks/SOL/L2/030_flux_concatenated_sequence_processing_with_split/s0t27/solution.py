import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,   # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,    # *hidden_states [B, L_img, D]
    dst_ptr,   # *output concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)  # tile along sequence (concatenated) dimension

    S_total = L_txt + L_img
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S_total
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Load and store for the first L_txt rows (encoder part)
    ehs_base = ehs_ptr + pid_b * ehs_stride_b
    ehs_ptrs = ehs_base + offs_s[:, None] * ehs_stride_s + offs_d[None, :] * ehs_stride_d
    val = tl.load(ehs_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

    dst_base = dst_ptr + pid_b * dst_stride_b
    dst_ptrs_e = dst_base + offs_s[:, None] * dst_stride_s + offs_d[None, :] * dst_stride_d
    tl.store(dst_ptrs_e, val, mask=mask_s[:, None] & mask_d[None, :])

    # Load and store for the next L_img rows (image part)
    hs_base = hs_ptr + pid_b * hs_stride_b
    hs_ptrs = hs_base + (offs_s - L_txt)[:, None] * hs_stride_s + offs_d[None, :] * hs_stride_d
    mask_hs = (offs_s >= L_txt) & (offs_s < S_total) & mask_d[None, :]
    val2 = tl.load(hs_ptrs, mask=mask_hs, other=0.0)

    dst_ptrs_h = dst_base + (offs_s - L_txt)[:, None] * dst_stride_s + offs_d[None, :] * dst_stride_d
    tl.store(dst_ptrs_h, val2, mask=mask_hs)


@triton.jit
def split_outputs_kernel(
    C_ptr,            # *C: [B, L_txt + L_img, D]
    out_encoder_ptr,  # *out_encoder: [B, L_txt, D]
    out_hidden_ptr,   # *out_hidden: [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    C_stride_b: tl.int32, C_stride_s: tl.int32, C_stride_d: tl.int32,
    out_e_stride_b: tl.int32, out_e_stride_s: tl.int32, out_e_stride_d: tl.int32,
    out_h_stride_b: tl.int32, out_h_stride_s: tl.int32, out_h_stride_d: tl.int32,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Encoder part: s in [0, L_txt)
    S_e = L_txt
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S_e
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    C_base = C_ptr + pid_b * C_stride_b
    C_ptrs = C_base + offs_s[:, None] * C_stride_s + offs_d[None, :] * C_stride_d
    val_e = tl.load(C_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

    out_e_base = out_encoder_ptr + pid_b * out_e_stride_b
    out_e_ptrs = out_e_base + offs_s[:, None] * out_e_stride_s + offs_d[None, :] * out_e_stride_d
    tl.store(out_e_ptrs, val_e, mask=mask_s[:, None] & mask_d[None, :])

    # Hidden part: s in [L_txt, L_txt + L_img)
    S_h = L_img
    hs_base = C_ptr + pid_b * C_stride_b
    hs_ptrs = hs_base + (offs_s + L_txt)[:, None] * C_stride_s + offs_d[None, :] * C_stride_d
    mask_hs = (offs_s < S_h) & mask_d[None, :]
    val_h = tl.load(hs_ptrs, mask=mask_hs, other=0.0)

    out_h_base = out_hidden_ptr + pid_b * out_h_stride_b
    out_h_ptrs = out_h_base + offs_s[:, None] * out_h_stride_s + offs_d[None, :] * out_h_stride_d
    tl.store(out_h_ptrs, val_h, mask=mask_hs)


@triton.jit
def batched_matmul_kernel(
    A_ptr,    # *A: [B, M, K], A = concatenated input
    Wt_ptr,   # *Wt: [K, N], where process_weight is [D, D], Wt = process_weight.T
    C_ptr,    # *C: [B, M, N] output
    B: tl.int32, M: tl.int32, K: tl.int32, N: tl.int32,
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    Wt_stride_k: tl.int32, Wt_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (batch, tiles along M, tiles along N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for partial tiles
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        mask_k = k_idx < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_base = A_ptr + pid_b * A_stride_b
        A_ptrs = A_base + offs_m[:, None] * A_stride_m + k_idx[None, :] * A_stride_k
        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load Wt tile: [BLOCK_K, BLOCK_N]
        Wt_ptrs = Wt_ptr + k_idx[:, None] * Wt_stride_k + offs_n[None, :] * Wt_stride_n
        w = tl.load(Wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # FMA
        acc += tl.dot(a, w)

    # Store result in C
    C_base = C_ptr + pid_b * C_stride_b
    C_ptrs = C_base + offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n
    # Cast back to input dtype if needed (C should match A's dtype; here we assume fp32 for simplicity)
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton implementation:
        - Concatenate [B, L_txt, D] and [B, L_img, D] along seq dim in Triton.
        - Apply linear projection in Triton using a custom batched matmul kernel.
        - Split outputs in Triton.
        All tensors must be on CUDA device; kernels are launched from forward.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA tensors."

        # Shapes
        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D and process_weight.shape[0] == D and process_weight.shape[1] == D

        # 1) Concatenate in Triton: dst [B, L_txt + L_img, D]
        dst = torch.empty((B, L_txt + L_img, D), device=hidden_states.device, dtype=torch.float32)
        grid_concat = (B, triton.cdiv(L_txt + L_img, 64))  # tile along sequence dim; 64 is a safe default
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, dst,
            B, L_txt, L_img, D,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            dst.stride(0), dst.stride(1), dst.stride(2),
            BLOCK_S=64, BLOCK_D=64
        )

        # 2) Matmul in Triton: C = dst @ process_weight.T, C: [B, L_txt + L_img, D]
        # Wt: [D, D] = process_weight.T (make contiguous for simple stride)
        Wt = process_weight.transpose(0, 1).contiguous()
        C = torch.empty((B, L_txt + L_img, D), device=hidden_states.device, dtype=torch.float32)
        grid_matmul = (B, triton.cdiv(L_txt + L_img, 64), triton.cdiv(D, 64))
        batched_matmul_kernel[grid_matmul](
            dst, Wt, C,
            B, L_txt + L_img, D, D,  # M = L_txt + L_img, K = D, N = D
            dst.stride(0), dst.stride(1), dst.stride(2),
            Wt.stride(0), Wt.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # 3) Split in Triton: out_encoder [B, L_txt, D], out_hidden [B, L_img, D]
        out_encoder = torch.empty((B, L_txt, D), device=hidden_states.device, dtype=torch.float32)
        out_hidden = torch.empty((B, L_img, D), device=hidden_states.device, dtype=torch.float32)
        grid_split = (B, triton.cdiv(L_txt, 64), triton.cdiv(D, 64))
        split_outputs_kernel[grid_split](
            C, out_encoder, out_hidden,
            B, L_txt, L_img, D,
            C.stride(0), C.stride(1), C.stride(2),
            out_encoder.stride(0), out_encoder.stride(1), out_encoder.stride(2),
            out_hidden.stride(0), out_hidden.stride(1), out_hidden.stride(2),
            BLOCK_S=64, BLOCK_D=64
        )

        # Cast outputs to the same dtype as inputs for consistency (original code likely uses float32)
        out_encoder = out_encoder.to(encoder_hidden_states.dtype)
        out_hidden = out_hidden.to(hidden_states.dtype)

        return out_encoder, out_hidden


def run(*args):
    return ModelNew()(*args)
