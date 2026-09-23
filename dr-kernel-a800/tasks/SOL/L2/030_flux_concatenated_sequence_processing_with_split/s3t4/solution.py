import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out_kernel(
    src_ptr,  # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,  # *ptr to out: [B, P, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Grid: (B, ceil(P/BLOCK_P))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)

    # We write into out[:, :T, :]
    p_start = pid_p * BLOCK_P
    p_offsets = p_start + tl.arange(0, BLOCK_P)
    mask_p = p_offsets < T

    # Loop over feature dimension D
    for d in range(0, D):
        src_row_ptr = src_ptr + b * T * D + p_offsets * D + d
        out_row_ptr = out_ptr + b * P * D + p_offsets * D + d
        src_vals = tl.load(src_row_ptr, mask=mask_p, other=0.0)
        tl.store(out_row_ptr, src_vals, mask=mask_p)


@triton.jit
def _copy_img_to_out_kernel(
    src_ptr,  # *ptr to hidden_states: [B, I, D]
    out_ptr,  # *ptr to out: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Grid: (B, ceil(P/BLOCK_P))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)

    # We write into out[:, T:, :]
    p_start = pid_p * BLOCK_P
    p_offsets = p_start + tl.arange(0, BLOCK_P)
    mask_p = p_offsets < I  # indices in [T, T+I)

    for d in range(0, D):
        src_row_ptr = src_ptr + b * I * D + p_offsets * D + d
        out_row_ptr = out_ptr + b * P * D + (p_offsets + T) * D + d
        src_vals = tl.load(src_row_ptr, mask=mask_p, other=0.0)
        tl.store(out_row_ptr, src_vals, mask=mask_p)


@triton.jit
def _gemm_kernel(
    X_ptr,        # *ptr to concatenated: [B, P, D]
    WT_ptr,       # *ptr to W_T: [D, D] (transpose of process_weight)
    Y_ptr,        # *ptr to output: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(P/BLOCK_P), ceil(D/BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < P
    mask_n = n_offsets < D

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dim in chunks
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load X tile [BP, BK]: X[b, p, k]
        x_ptrs = X_ptr + b * P * D + p_offsets[:, None] * D + k_offsets[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_p[:, None] & mask_k[None, :], other=0.0)  # fp32

        # Load W_T tile [BK, BN]: W_T[k, n]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # fp32

        acc += tl.dot(x_tile, wt_tile)

    # Store results to Y [B, P, D]
    y_ptrs = Y_ptr + b * P * D + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_encoder_kernel(
    src_ptr,  # *ptr to processed: [B, P, D]
    dst_ptr,  # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Grid: (B, ceil(T/BLOCK_P), 1)
    b = tl.program_id(0)
    pid_p = tl.program_id(1)

    p_start = pid_p * BLOCK_P
    p_offsets = p_start + tl.arange(0, BLOCK_P)
    mask_p = p_offsets < T

    for d in range(0, D):
        src_row_ptr = src_ptr + b * P * D + p_offsets * D + d
        dst_row_ptr = dst_ptr + b * T * D + p_offsets * D + d
        src_vals = tl.load(src_row_ptr, mask=mask_p, other=0.0)
        tl.store(dst_row_ptr, src_vals, mask=mask_p)


@triton.jit
def _copy_slice_img_kernel(
    src_ptr,  # *ptr to processed: [B, P, D]
    dst_ptr,  # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Grid: (B, ceil(I/BLOCK_P), 1)
    b = tl.program_id(0)
    pid_p = tl.program_id(1)

    p_start = pid_p * BLOCK_P
    p_offsets = p_start + tl.arange(0, BLOCK_P)
    mask_p = p_offsets < I  # indices in [T, T+I)

    for d in range(0, D):
        src_row_ptr = src_ptr + b * P * D + (p_offsets + T) * D + d
        dst_row_ptr = dst_ptr + b * I * D + p_offsets * D + d
        src_vals = tl.load(src_row_ptr, mask=mask_p, other=0.0)
        tl.store(dst_row_ptr, src_vals, mask=mask_p)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation of:
          concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          processed_encoder = processed[:, :T, :]
          processed_hidden = processed[:, T:, :]
        All torch ops are avoided in forward; Triton kernels are launched instead.
        """
        # Ensure inputs are CUDA tensors and contiguous; cast to float32 for compute
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        P = T + I

        assert encoder_hidden_states.shape == (B, T, D), "encoder_hidden_states must be [B, T, D]"
        assert hidden_states.shape == (B, I, D), "hidden_states must be [B, I, D]"
        assert process_weight.shape == (D, D), "process_weight must be [D, D]"

        device = hidden_states.device
        assert device.type == "cuda", "Inputs must be on CUDA device"

        # Allocate concatenated and processed outputs
        out = torch.empty((B, P, D), device=device, dtype=torch.float32)
        processed = torch.empty((B, P, D), device=device, dtype=torch.float32)

        # Copy encoder part to out[:, :T, :]
        BLOCK_P = 128
        grid_concat_enc = (B, triton.cdiv(P, BLOCK_P))
        _copy_encoder_to_out_kernel[grid_concat_enc](
            encoder_hidden_states.contiguous().float(),
            out,
            B=B, T=T, D=D, P=P, BLOCK_P=BLOCK_P,
            num_warps=4, num_stages=2
        )

        # Copy image part to out[:, T:, :]
        grid_concat_img = (B, triton.cdiv(P, BLOCK_P))
        _copy_img_to_out_kernel[grid_concat_img](
            hidden_states.contiguous().float(),
            out,
            B=B, I=I, D=D, P=P, BLOCK_P=BLOCK_P,
            num_warps=4, num_stages=2
        )

        # Prepare weight as [D, D] (transpose of original [D, D]); cast to fp32
        WT = process_weight.contiguous().float()  # [D, D]

        # Launch GEMM: processed = out @ WT
        BLOCK_P_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid_gemm = (B, triton.cdiv(P, BLOCK_P_M), triton.cdiv(D, BLOCK_N))
        _gemm_kernel[grid_gemm](
            out, WT, processed,
            B=B, P=P, D=D,
            BLOCK_P=BLOCK_P_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3
        )

        # Allocate outputs for the two streams and copy slices
        processed_encoder = torch.empty((B, T, D), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=device, dtype=torch.float32)

        grid_slice_enc = (B, triton.cdiv(T, BLOCK_P))
        _copy_slice_encoder_kernel[grid_slice_enc](
            processed, processed_encoder,
            B=B, T=T, D=D, BLOCK_P=BLOCK_P,
            num_warps=4, num_stages=2
        )

        grid_slice_img = (B, triton.cdiv(I, BLOCK_P))
        _copy_slice_img_kernel[grid_slice_img](
            processed, processed_hidden,
            B=B, I=I, D=D, BLOCK_P=BLOCK_P,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
