import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_encoder_elementwise_vec(
    enc_ptr, wt_ptr, out_ptr,
    B, T, D,
    enc_bs, enc_ts, enc_ds,
    wt_ds0, wt_ds1,  # strides for [K, D] i.e., wt[k, d]
    out_bs, out_ts, out_ds,
    BLOCK: tl.constexpr,  # vector width along output hidden dim
    BLOCK_K: tl.constexpr  # reduction chunk along K (keep small to minimize order impact)
):
    # program ids: batch, row (sequence position), col tile along hidden dim
    b = tl.program_id(0)
    p = tl.program_id(1)  # p in [0, T)
    col_block = tl.program_id(2)

    # compute output hidden indices for this program
    offs_n = col_block * BLOCK + tl.arange(0, BLOCK)
    mask_n = offs_n < D

    # accumulator for this (b, p, offs_n) vector
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # loop over K (hidden) dimension in chunks
    for k_start in range(0, D, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_range < D

        # load W^T[k, d] chunk as [BLOCK_K, BLOCK]
        # wt_ptr is [K, D], so stride0 is along K, stride1 along D
        wt_ptrs = wt_ptr + k_range[:, None] * wt_ds0 + offs_n[None, :] * wt_ds1
        # mask for 2D load: valid k and d
        wt_mask = mask_k[:, None] & mask_n[None, :]
        wt_tile = tl.load(wt_ptrs, mask=wt_mask, other=0.0)  # [BLOCK_K, BLOCK], fp32 or fp16 loaded

        # load enc[b, p, k] vector [BLOCK_K]
        enc_ptrs = enc_ptr + b * enc_bs + p * enc_ts + k_range * enc_ds
        enc_mask = mask_k
        enc_vec = tl.load(enc_ptrs, mask=enc_mask, other=0.0)  # [BLOCK_K], accumulate in fp32

        # accumulate outer product: acc += sum_k enc_vec[k] * wt_tile[k, :]
        # Broadcast enc_vec [BLOCK_K] to [BLOCK_K, BLOCK] and multiply
        prod = enc_vec[:, None] * wt_tile  # fp32
        acc += tl.sum(prod, axis=0)  # reduce over K chunk

    # store results
    out_ptrs = out_ptr + b * out_bs + p * out_ts + offs_n * out_ds
    tl.store(out_ptrs, acc, mask=mask_n)


@triton.jit
def _gemm_hidden_elementwise_vec(
    hst_ptr, wt_ptr, out_ptr,
    B, I, D,
    hst_bs, hst_is, hst_ds,
    wt_ds0, wt_ds1,  # strides for [K, D]
    out_bs, out_is, out_ds,
    BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    p = tl.program_id(1)  # p in [0, I)
    col_block = tl.program_id(2)

    offs_n = col_block * BLOCK + tl.arange(0, BLOCK)
    mask_n = offs_n < D

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for k_start in range(0, D, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_range < D

        wt_ptrs = wt_ptr + k_range[:, None] * wt_ds0 + offs_n[None, :] * wt_ds1
        wt_mask = mask_k[:, None] & mask_n[None, :]
        wt_tile = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        hst_ptrs = hst_ptr + b * hst_bs + p * hst_is + k_range * hst_ds
        hst_mask = mask_k
        hst_vec = tl.load(hst_ptrs, mask=hst_mask, other=0.0)

        prod = hst_vec[:, None] * wt_tile
        acc += tl.sum(prod, axis=0)

    out_ptrs = out_ptr + b * out_bs + p * out_is + offs_n * out_ds
    tl.store(out_ptrs, acc, mask=mask_n)


def _run_triton_gemm_vec(x: torch.Tensor, wt: torch.Tensor, out: torch.Tensor, BLOCK=128, BLOCK_K=8):
    """
    x: [B, M, D] (either encoder or hidden)
    wt: [D, D] (process_weight)
    out: [B, M, D] (output)
    """
    assert x.is_cuda and wt.is_cuda and out.is_cuda
    B, M, D = x.shape
    x = x.contiguous()
    wt = wt.contiguous()
    out = out.contiguous()

    grid = (B, M, triton.cdiv(D, BLOCK))
    _gemm_elementwise_vec = _gemm_encoder_elementwise_vec if x.shape[1] == M else _gemm_hidden_elementwise_vec  # placeholder to help type inference
    _gemm_elementwise_vec = _gemm_encoder_elementwise_vec  # explicit use of encoder kernel
    _gemm_encoder_elementwise_vec[grid](
        x, wt, out,
        B, M, D,
        x.stride(0), x.stride(1), x.stride(2),
        wt.stride(0), wt.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK=BLOCK, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward without torch.cat or torch.matmul.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        Shapes:
          hidden_states: [batch, img_seq_len, hidden_dim]
          encoder_hidden_states: [batch, text_seq_len, hidden_dim]
          process_weight: [hidden_dim, hidden_dim]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton"
        B = hidden_states.shape[0]
        D = hidden_states.shape[2]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]

        # Ensure process_weight is contiguous [D, D]
        wt = process_weight.contiguous()

        # Output tensors
        processed_encoder = torch.empty((B, T, D), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=hidden_states.device, dtype=torch.float32)

        # Run Triton GEMM for encoder stream
        _run_triton_gemm_vec(encoder_hidden_states, wt, processed_encoder, BLOCK=128, BLOCK_K=8)
        # Run Triton GEMM for hidden stream
        _run_triton_gemm_vec(hidden_states, wt, processed_hidden, BLOCK=128, BLOCK_K=8)

        # Cast back to original dtype if needed (evaluation expects fp32 outputs; keep fp32 for correctness)
        # If original inputs were fp32, we already match. If not, the original run returns fp32 by default.
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
