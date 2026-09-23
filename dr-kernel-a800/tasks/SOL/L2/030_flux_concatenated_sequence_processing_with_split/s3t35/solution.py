import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_encoder_blocked(
    X,           # *ptr to encoder_hidden_states: [B, T, D]
    WT,          # *ptr to process_weight.T: [D, D]
    Out,         # *ptr to processed_encoder: [B, T, D]
    B, T, D,     # sizes
    stride_xb, stride_xt, stride_xd,   # strides for X
    stride_wtk, stride_wtd,            # strides for WT
    stride_ob, stride_ot, stride_od,   # strides for Out
    BLOCK_M: tl.constexpr,             # tile size over M (T)
    BLOCK_N: tl.constexpr,             # tile size over N (D)
    BLOCK_K: tl.constexpr,             # tile size over K (D)
):
    # Grid over (batch, tiles of T, tiles of D)
    b = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)

    # Compute tile indices
    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)  # sequence positions in [0..T)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)  # output hidden dims in [0..D)
    mask_m = offs_m < T
    mask_n = offs_n < D

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K (D) in chunks of BLOCK_K
    for k in range(0, D, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D

        # Load X tile: shape (BLOCK_M, BLOCK_K), X[b, offs_m, offs_k]
        x_ptrs = X + b * stride_xb + offs_m[:, None] * stride_xt + offs_k[None, :] * stride_xd
        x_mask = mask_m[:, None] & mask_k[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        # Load WT tile: shape (BLOCK_K, BLOCK_N), WT[offs_k, offs_n]
        wt_ptrs = WT + offs_k[:, None] * stride_wtk + offs_n[None, :] * stride_wtd
        wt_mask = mask_k[:, None] & mask_n[None, :]
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        # Accumulate (cast to fp32 for numerical stability)
        acc += tl.dot(x.to(tl.float32), wt.to(tl.float32))

    # Store result: Out[b, offs_m, offs_n]
    out_ptrs = Out + b * stride_ob + offs_m[:, None] * stride_ot + offs_n[None, :] * stride_od
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _gemm_hidden_blocked(
    X,           # *ptr to hidden_states: [B, I, D]
    WT,          # *ptr to process_weight.T: [D, D]
    Out,         # *ptr to processed_hidden: [B, I, D]
    B, I, D,     # sizes
    stride_xb, stride_xi, stride_xd,   # strides for X
    stride_wtk, stride_wtd,            # strides for WT
    stride_ob, stride_oi, stride_od,   # strides for Out
    BLOCK_M: tl.constexpr,             # tile size over M (I)
    BLOCK_N: tl.constexpr,             # tile size over N (D)
    BLOCK_K: tl.constexpr,             # tile size over K (D)
):
    # Grid over (batch, tiles of I, tiles of D)
    b = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)

    # Compute tile indices
    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)  # image sequence positions in [0..I)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)  # output hidden dims in [0..D)
    mask_m = offs_m < I
    mask_n = offs_n < D

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K (D) in chunks of BLOCK_K
    for k in range(0, D, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D

        # Load X tile: shape (BLOCK_M, BLOCK_K), X[b, offs_m, offs_k]
        x_ptrs = X + b * stride_xb + offs_m[:, None] * stride_xi + offs_k[None, :] * stride_xd
        x_mask = mask_m[:, None] & mask_k[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        # Load WT tile: shape (BLOCK_K, BLOCK_N), WT[offs_k, offs_n]
        wt_ptrs = WT + offs_k[:, None] * stride_wtk + offs_n[None, :] * stride_wtd
        wt_mask = mask_k[:, None] & mask_n[None, :]
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        # Accumulate (cast to fp32 for numerical stability)
        acc += tl.dot(x.to(tl.float32), wt.to(tl.float32))

    # Store result: Out[b, offs_m, offs_n]
    out_ptrs = Out + b * stride_ob + offs_m[:, None] * stride_oi + offs_n[None, :] * stride_od
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA and contiguity
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Make inputs contiguous
        enc = encoder_hidden_states.contiguous()  # [B, T, D]
        hst = hidden_states.contiguous()         # [B, I, D]
        WT = process_weight.t().contiguous()     # [D, D]

        # Allocate outputs
        processed_encoder = torch.empty((B, T, D), dtype=enc.dtype, device=enc.device)
        processed_hidden = torch.empty((B, I, D), dtype=hst.dtype, device=hst.device)

        # Dynamic block sizes based on problem dimensions
        # Larger BLOCK_N helps vectorization along hidden dim; BLOCK_K 64/128 is standard
        BLOCK_M = 64 if (T >= 64 or I >= 64) else 32
        BLOCK_N = 128 if D >= 128 else 64
        BLOCK_K = 64

        # Launch GEMM for encoder stream: [B, T, D] @ [D, D] -> [B, T, D]
        grid_enc = (B, triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _gemm_encoder_blocked[grid_enc](
            enc, WT, processed_encoder,
            B, T, D,
            enc.stride(0), enc.stride(1), enc.stride(2),
            WT.stride(0), WT.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Launch GEMM for hidden stream: [B, I, D] @ [D, D] -> [B, I, D]
        grid_hid = (B, triton.cdiv(I, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _gemm_hidden_blocked[grid_hid](
            hst, WT, processed_hidden,
            B, I, D,
            hst.stride(0), hst.stride(1), hst.stride(2),
            WT.stride(0), WT.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
