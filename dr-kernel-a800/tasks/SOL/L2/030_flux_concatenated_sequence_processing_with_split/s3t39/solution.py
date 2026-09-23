import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_per_tile_encoder_kernel(
    A,  # pointer to encoder_hidden_states [B, T, D]
    W,  # pointer to process_weight.T [D, D]
    C,  # pointer to processed_encoder [B, T, D]
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    stride_Ab, stride_At, stride_Ad,
    stride_Wm, stride_Wn,
    stride_Cb, stride_Ct, stride_Cd,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # Grid: (B, ceil(T/BM), ceil(D/BN))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Row and column indices for this program tile
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)

    # Bounds mask
    mask_m = rows < T
    mask_n = cols < D

    # Initialize accumulator [BM, BN] in fp32
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Reduction over K in chunks of BK
    for k0 in range(0, D, BK):
        ks = k0 + tl.arange(0, BK)
        mask_k = ks < D

        # Load A tile: shape [BM, BK]
        A_ptrs = A + pid_b * stride_Ab + rows[:, None] * stride_At + ks[None, :] * stride_Ad
        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W tile transposed as [BK, BN] where W is [D, D]
        # We want W_T[k, d] -> W[ks, cols]
        W_ptrs = W + ks[:, None] * stride_Wm + cols[None, :] * stride_Wn
        w = tl.load(W_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a.to(tl.float32), w.to(tl.float32))

    # Store results C[b, rows, cols] = acc
    C_ptrs = C + pid_b * stride_Cb + rows[:, None] * stride_Ct + cols[None, :] * stride_Cd
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _gemm_per_tile_hidden_kernel(
    A,  # pointer to hidden_states [B, I, D]
    W,  # pointer to process_weight.T [D, D]
    C,  # pointer to processed_hidden [B, I, D]
    B: tl.constexpr, I: tl.constexpr, D: tl.constexpr,
    stride_Ab, stride_Ai, stride_Ad,
    stride_Wm, stride_Wn,
    stride_Cb, stride_Ci, stride_Cd,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # Grid: (B, ceil(I/BM), ceil(D/BN))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    rows = pid_m * BM + tl.arange(0, BM)  # indices over I
    cols = pid_n * BN + tl.arange(0, BN)  # indices over D

    mask_m = rows < I
    mask_n = cols < D

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, D, BK):
        ks = k0 + tl.arange(0, BK)
        mask_k = ks < D

        # Load A tile [BM, BK]
        A_ptrs = A + pid_b * stride_Ab + rows[:, None] * stride_Ai + ks[None, :] * stride_Ad
        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W tile [BK, BN]
        W_ptrs = W + ks[:, None] * stride_Wm + cols[None, :] * stride_Wn
        w = tl.load(W_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a.to(tl.float32), w.to(tl.float32))

    # Store
    C_ptrs = C + pid_b * stride_Cb + rows[:, None] * stride_Ci + cols[None, :] * stride_Cd
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - No torch.cat or torch.matmul usage.
        - Computes both outputs via Triton GEMM kernels.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        # Ensure contiguity for predictable strides
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        WT = process_weight.t().contiguous()  # [D, D]

        # Output tensors
        processed_encoder = torch.empty((B, T, D), dtype=torch.float32, device=enc.device)
        processed_hidden = torch.empty((B, I, D), dtype=torch.float32, device=enc.device)

        # Choose modest tile sizes to balance correctness and performance
        BM = 8
        BN = 16
        BK = 32

        # Launch kernel for encoder stream: [B, T, D] @ [D, D] -> [B, T, D]
        grid_enc = (B, triton.cdiv(T, BM), triton.cdiv(D, BN))
        _gemm_per_tile_encoder_kernel[grid_enc](
            enc, WT, processed_encoder,
            B, T, D,
            enc.stride(0), enc.stride(1), enc.stride(2),
            WT.stride(0), WT.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BM=BM, BN=BN, BK=BK,
            num_warps=2, num_stages=2,
        )

        # Launch kernel for hidden stream: [B, I, D] @ [D, D] -> [B, I, D]
        grid_hid = (B, triton.cdiv(I, BM), triton.cdiv(D, BN))
        _gemm_per_tile_hidden_kernel[grid_hid](
            hst, WT, processed_hidden,
            B, I, D,
            hst.stride(0), hst.stride(1), hst.stride(2),
            WT.stride(0), WT.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BM=BM, BN=BN, BK=BK,
            num_warps=2, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
