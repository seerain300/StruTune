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
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)  # tile along sequence dimension

    # Copy encoder_hidden_states: s in [0, L_txt)
    S_e = L_txt
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S_e
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    ehs_base = ehs_ptr + pid_b * ehs_stride_b
    ehs_ptrs = ehs_base + offs_s[:, None] * ehs_stride_s + offs_d[None, :] * ehs_stride_d
    val_e = tl.load(ehs_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

    dst_e_base = dst_ptr + pid_b * dst_stride_b
    dst_e_ptrs = dst_e_base + offs_s[:, None] * dst_stride_s + offs_d[None, :] * dst_stride_d
    tl.store(dst_e_ptrs, val_e, mask=mask_s[:, None] & mask_d[None, :])

    # Copy hidden_states: s in [L_txt, L_txt + L_img)
    S_h = L_img
    hs_base = hs_ptr + pid_b * hs_stride_b
    hs_ptrs = hs_base + (offs_s + L_txt)[:, None] * hs_stride_s + offs_d[None, :] * hs_stride_d
    val_h = tl.load(hs_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

    dst_h_base = dst_ptr + pid_b * dst_stride_b
    dst_h_ptrs = dst_h_base + (offs_s + L_txt)[:, None] * dst_stride_s + offs_d[None, :] * dst_stride_d
    tl.store(dst_h_ptrs, val_h, mask=mask_s[:, None] & mask_d[None, :])


@triton.jit
def batched_matmul_kernel(
    A_ptr,     # *A: [B, M, K], where M = L_txt + L_img, K = D
    Wt_ptr,    # *Wt: [K, N], N = D
    C_ptr,     # *C: [B, M, N]
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

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k  # [BLOCK_K]
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_k = k_idx < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + pid_b * A_stride_b + offs_m[:, None] * A_stride_m + k_idx[None, :] * A_stride_k
        a_mask = (mask_m[:, None] & mask_k[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # float32 by default

        # Load Wt tile: [BLOCK_K, BLOCK_N]
        wt_ptrs = Wt_ptr + k_idx[:, None] * Wt_stride_k + offs_n[None, :] * Wt_stride_n
        wt_mask = (mask_k[:, None] & mask_n[None, :])
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)  # float32

        # Accumulate
        acc += tl.dot(a, wt)

    # Store C tile
    c_ptrs = C_ptr + pid_b * C_stride_b + offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n
    c_mask = (mask_m[:, None] & mask_n[None, :])
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def split_outputs_kernel(
    C_ptr,           # *processed [B, M, D], M = L_txt + L_img
    out_e_ptr,       # *processed_encoder [B, L_txt, D]
    out_h_ptr,       # *processed_hidden [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_d: tl.int32,
    out_e_stride_b: tl.int32, out_e_stride_s: tl.int32, out_e_stride_d: tl.int32,
    out_h_stride_b: tl.int32, out_h_stride_s: tl.int32, out_h_stride_d: tl.int32,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)  # tile along sequence dimension

    # Copy encoder part: s in [0, L_txt)
    S_e = L_txt
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S_e
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    C_e_base = C_ptr + pid_b * C_stride_b
    C_e_ptrs = C_e_base + offs_s[:, None] * C_stride_m + offs_d[None, :] * C_stride_d
    val_e = tl.load(C_e_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

    out_e_base = out_e_ptr + pid_b * out_e_stride_b
    out_e_ptrs = out_e_base + offs_s[:, None] * out_e_stride_s + offs_d[None, :] * out_e_stride_d
    tl.store(out_e_ptrs, val_e, mask=mask_s[:, None] & mask_d[None, :])

    # Copy hidden part: s in [L_txt, L_txt + L_img)
    S_h = L_img
    C_h_base = C_ptr + pid_b * C_stride_b
    C_h_ptrs = C_h_base + (offs_s + L_txt)[:, None] * C_stride_m + offs_d[None, :] * C_stride_d
    val_h = tl.load(C_h_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

    out_h_base = out_h_ptr + pid_b * out_h_stride_b
    out_h_ptrs = out_h_base + offs_s[:, None] * out_h_stride_s + offs_d[None, :] * out_h_stride_d
    tl.store(out_h_ptrs, val_h, mask=mask_s[:, None] & mask_d[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton implementation:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim (Triton).
        2) Apply linear projection using Triton GEMM: C = A @ process_weight.T.
        3) Split outputs into processed_encoder and processed_hidden (Triton).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device for Triton."

        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, L_txt, D), "encoder_hidden_states shape must be [B, L_txt, D]"
        assert hidden_states.shape == (B, L_img, D), "hidden_states shape must be [B, L_img, D]"
        assert process_weight.shape == (D, D), "process_weight shape must be [D, D]"

        # Ensure contiguous for simpler stride handling
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Step 1: Concatenate in Triton
        M = L_txt + L_img
        A = torch.empty((B, M, D), dtype=hs.dtype, device=hs.device)

        grid_concat = (B, triton.cdiv(M, 64), triton.cdiv(D, 64))
        concat_seqs_kernel[grid_concat](
            ehs, hs, A,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            BLOCK_S=64, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # Step 2: Batched matmul in Triton: C = A @ W^T
        K = D
        N = D
        C = torch.empty((B, M, N), dtype=hs.dtype, device=hs.device)

        grid_gemm = (B, triton.cdiv(M, 64), triton.cdiv(N, 64))
        # Wt = W.T is [K, N]
        Wt = W.transpose(0, 1).contiguous()  # [D, D]

        batched_matmul_kernel[grid_gemm](
            A, Wt, C,
            B, M, K, N,
            A.stride(0), A.stride(1), A.stride(2),
            Wt.stride(0), Wt.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Step 3: Split outputs in Triton
        processed_encoder = torch.empty((B, L_txt, D), dtype=hs.dtype, device=hs.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=hs.dtype, device=hs.device)

        grid_split = (B, triton.cdiv(L_txt, 64), triton.cdiv(D, 64))
        split_outputs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_S=64, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


# For local testing (optional)
if __name__ == "__main__":
    # Example inputs (ensure on CUDA)
    device = "cuda"
    B, D = 2, 64
    L_txt, L_img = 128, 256
    x = torch.randn(B, L_img, D, device=device, dtype=torch.float32)
    y = torch.randn(B, L_txt, D, device=device, dtype=torch.float32)
    W = torch.randn(D, D, device=device, dtype=torch.float32)

    model = ModelNew().to(device)
    out_e, out_h = model(x, y, W)
    print("Output encoder shape:", out_e.shape)  # [B, L_txt, D]
    print("Output hidden shape:", out_h.shape)   # [B, L_img, D]


def run(*args):
    return ModelNew()(*args)
