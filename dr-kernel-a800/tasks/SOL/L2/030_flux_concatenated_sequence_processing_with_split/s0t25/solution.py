import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,        # *encoder_hidden_states: [B, L_txt, D]
    hs_ptr,         # *hidden_states: [B, L_img, D]
    out_ptr,        # *output concatenated: [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    out_stride_b: tl.int32, out_stride_s: tl.int32, out_stride_d: tl.int32,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    S = L_txt + L_img
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Output base for this batch
    out_base = out_ptr + pid_b * out_stride_b
    out_ptrs = out_base + offs_s[:, None] * out_stride_s + offs_d[None, :] * out_stride_d

    # Load from encoder_hidden_states for first L_txt positions
    mask_txt = (offs_s < L_txt) & mask_s
    ehs_base = ehs_ptr + pid_b * ehs_stride_b
    ehs_ptrs = ehs_base + offs_s[:, None] * ehs_stride_s + offs_d[None, :] * ehs_stride_d
    val_txt = tl.load(ehs_ptrs, mask=mask_txt[:, None] & mask_d[None, :], other=0.0)

    # Load from hidden_states for remaining positions
    mask_img = (offs_s >= L_txt) & mask_s
    hs_idx = offs_s - L_txt  # valid only where mask_img is True
    hs_base = hs_ptr + pid_b * hs_stride_b
    hs_ptrs = hs_base + hs_idx[:, None] * hs_stride_s + offs_d[None, :] * hs_stride_d
    val_img = tl.load(hs_ptrs, mask=mask_img[:, None] & mask_d[None, :], other=0.0)

    # Select appropriate value per position
    val = tl.where(mask_txt[:, None], val_txt, val_img)

    # Store
    tl.store(out_ptrs, val, mask=mask_s[:, None] & mask_d[None, :])


@triton.jit
def split_seqs_kernel(
    in_ptr,         # *input: [B, L, D] processed tensor
    out_encoder_ptr,  # *output encoder: [B, L_txt, D]
    out_hidden_ptr,   # *output hidden: [B, L_img, D]
    B: tl.int32,
    L: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    in_stride_b: tl.int32, in_stride_s: tl.int32, in_stride_d: tl.int32,
    out_e_stride_b: tl.int32, out_e_stride_s: tl.int32, out_e_stride_d: tl.int32,
    out_h_stride_b: tl.int32, out_h_stride_s: tl.int32, out_h_stride_d: tl.int32,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)  # tile along sequence dimension (encoder part)

    S_e = L_txt
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S_e
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    in_base = in_ptr + pid_b * in_stride_b
    in_ptrs = in_base + offs_s[:, None] * in_stride_s + offs_d[None, :] * in_stride_d
    val_e = tl.load(in_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

    out_e_base = out_encoder_ptr + pid_b * out_e_stride_b
    out_e_ptrs = out_e_base + offs_s[:, None] * out_e_stride_s + offs_d[None, :] * out_e_stride_d
    tl.store(out_e_ptrs, val_e, mask=mask_s[:, None] & mask_d[None, :])

    # Hidden part: copy from in at offsets [L_txt + offs_s]
    S_h = L_img
    hs_base = in_base + (L_txt + offs_s)[:, None] * in_stride_s + offs_d[None, :] * in_stride_d
    val_h = tl.load(hs_base, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

    out_h_base = out_hidden_ptr + pid_b * out_h_stride_b
    out_h_ptrs = out_h_base + offs_s[:, None] * out_h_stride_s + offs_d[None, :] * out_h_stride_d
    tl.store(out_h_ptrs, val_h, mask=mask_s[:, None] & mask_d[None, :])


@triton.jit
def batched_matmul_kernel(
    A_ptr,  # *A: [B, M, K]
    Wt_ptr, # *Wt: [K, N]
    C_ptr,  # *C: [B, M, N]
    B: tl.int32, M: tl.int32, K: tl.int32, N: tl.int32,
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    Wt_stride_k: tl.int32, Wt_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (batch, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + pid_b * A_stride_b + (offs_m[:, None] * A_stride_m) + (offs_k[None, :] * A_stride_k)
        A_mask = (mask_m[:, None] & mask_k[None, :])
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W^T tile: shape [BLOCK_K, BLOCK_N]
        Wt_ptrs = Wt_ptr + (offs_k[:, None] * Wt_stride_k) + (offs_n[None, :] * Wt_stride_n)
        Wt_mask = (mask_k[:, None] & mask_n[None, :])
        w = tl.load(Wt_ptrs, mask=Wt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store result tile
    C_ptrs = C_ptr + pid_b * C_stride_b + (offs_m[:, None] * C_stride_m) + (offs_n[None, :] * C_stride_n)
    C_mask = (mask_m[:, None] & mask_n[None, :])
    # Cast back to original dtype of C if needed (assume C dtype matches A / Wt)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, L_img, D]
        encoder_hidden_states: [B, L_txt, D]
        process_weight: [D, D]
        returns (processed_encoder: [B, L_txt, D], processed_hidden: [B, L_img, D])
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"

        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D
        assert process_weight.shape[0] == D and process_weight.shape[1] == D

        # 1) Triton: concatenate along sequence dim
        S = L_txt + L_img
        concatenated = torch.empty((B, S, D), dtype=hidden_states.dtype, device=hidden_states.device)

        BLOCK_S = 128
        BLOCK_D = 128
        grid_concat = (B, triton.cdiv(S, BLOCK_S))
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, L_txt, L_img, D,
            *encoder_hidden_states.stride(), *hidden_states.stride(), *concatenated.stride(),
            BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # 2) Triton: batched matmul processed = concatenated @ process_weight.T
        # Make inputs contiguous for robust indexing
        A = concatenated.contiguous()  # [B, M, K] with M=S, K=D
        Wt = process_weight.transpose(0, 1).contiguous()  # [K, N] with K=D, N=D

        processed = torch.empty((B, S, D), dtype=A.dtype, device=A.device)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_matmul = (B, triton.cdiv(S, BLOCK_M), triton.cdiv(D, BLOCK_N))
        batched_matmul_kernel[grid_matmul](
            A, Wt, processed,
            B, S, D, D,  # M=S, K=D, N=D
            *A.stride(), *Wt.stride(), *processed.stride(),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Triton: split into encoder and hidden streams
        processed_encoder = torch.empty((B, L_txt, D), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=processed.dtype, device=processed.device)

        grid_split = (B, triton.cdiv(L_txt, BLOCK_S))
        split_seqs_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, S, L_txt, L_img, D,
            *processed.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
