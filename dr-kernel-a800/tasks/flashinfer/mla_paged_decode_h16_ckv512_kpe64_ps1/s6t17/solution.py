import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_qn_qp_kernel(
    qn_ptr,         # *fp32, [H, D], contiguous
    qp_ptr,         # *fp32, [H, Dp], contiguous
    KcT_ptr,        # *fp32, [D, L_tokens], contiguous (Kc.T)
    KpT_ptr,        # *fp32, [Dp, L_tokens], contiguous (Kp.T)
    logits_qn_ptr,  # *fp32, [H, L_tokens], contiguous
    logits_qp_ptr,  # *fp32, [H, L_tokens], contiguous
    H,              # int32 (runtime)
    D: tl.constexpr,          # compile-time constant: 512
    Dp: tl.constexpr,         # compile-time constant: 64
    L_tokens,       # int32 (runtime per-batch)
):
    # Grid: (h,)
    h = tl.program_id(0)

    # For q_nope: compute logits_qn[h, t] = sum_d qn[h, d] * Kc[t, d]
    for t in range(0, L_tokens):
        acc = 0.0
        for d in range(0, D):
            q = tl.load(qn_ptr + h * D + d)
            K = tl.load(KcT_ptr + d * L_tokens + t)  # Kc.T row at t, column d
            acc += q * K
        tl.store(logits_qn_ptr + h * L_tokens + t, acc)

    # For q_pe: compute logits_qp[h, t] = sum_dp qp[h, dp] * Kp[t, dp]
    for t in range(0, L_tokens):
        acc = 0.0
        for dp in range(0, Dp):
            q = tl.load(qp_ptr + h * Dp + dp)
            K = tl.load(KpT_ptr + dp * L_tokens + t)  # Kp.T row at t, column dp
            acc += q * K
        tl.store(logits_qp_ptr + h * L_tokens + t, acc)


@triton.jit
def compute_lse_reduction_kernel(
    logits_ptr,     # *fp32, [H*L_tokens] flattened
    lse_ptr,        # *fp32, [H]
    H,              # int32
    L_tokens,       # int32
    BLOCK: tl.constexpr,  # e.g., 128
):
    # Grid: (h,)
    h = tl.program_id(0)
    start = h * L_tokens

    # Compute max
    m = -float("inf")
    for offs in range(0, L_tokens, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < L_tokens
        vals = tl.load(logits_ptr + start + idx, mask=mask, other=-float("inf"))
        local_max = tl.max(vals, axis=0)
        if local_max > m:
            m = local_max

    # Compute sum of exp(x - m)
    sum_exp = 0.0
    for offs in range(0, L_tokens, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < L_tokens
        vals = tl.load(logits_ptr + start + idx, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(vals - m), axis=0)

    lse = tl.log(sum_exp) / math.log(2.0)
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_matvec_kernel(
    logits_ptr,     # *fp32, [H*L_tokens] flattened
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    output_ptr,     # *fp32, [H*D] flattened
    H,              # int32
    D: tl.constexpr,          # 512
    L_tokens,       # int32
    BLOCK: tl.constexpr,      # vectorization over tokens, e.g., 128
):
    # Grid: (h,)
    h = tl.program_id(0)
    start_logits = h * L_tokens

    # Compute m and sum_exp for softmax
    m = -float("inf")
    for offs in range(0, L_tokens, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < L_tokens
        vals = tl.load(logits_ptr + start_logits + idx, mask=mask, other=-float("inf"))
        local_max = tl.max(vals, axis=0)
        if local_max > m:
            m = local_max

    sum_exp = 0.0
    for offs in range(0, L_tokens, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < L_tokens
        vals = tl.load(logits_ptr + start_logits + idx, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(vals - m), axis=0)

    # Compute output[h, :]
    for d in range(0, D):
        out_val = 0.0
        for offs in range(0, L_tokens, BLOCK):
            idx = offs + tl.arange(0, BLOCK)
            mask = idx < L_tokens
            vals = tl.load(logits_ptr + start_logits + idx, mask=mask, other=-float("inf"))
            attn = tl.exp(vals - m) / sum_exp
            Kd = tl.load(Kc_ptr + idx * D + d, mask=mask, other=0.0)
            out_val += tl.sum(attn * Kd, axis=0)
        tl.store(output_ptr + h * D + d, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0):
        super().__init__()
        self.sm_scale = float(sm_scale)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale=None):
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"

        # Shapes
        B, H, D = q_nope.shape
        _, _, Dp = q_pe.shape
        num_pages = ckv_cache.shape[0]
        # We only use the tokens within kv_indptr per batch; no need to assume num_pages matches

        # Cast to fp32 for computation
        qn = q_nope.to(torch.float32).contiguous()        # [B, H, D]
        qp = q_pe.to(torch.float32).contiguous()          # [B, H, Dp]
        Kc = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        output = torch.empty((B, H, D), dtype=torch.float32, device=device)  # computed buffer
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Compute valid token count and slice indices for this batch
            if kv_indptr.numel() <= 1:
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            L_tokens = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            tok_idx = kv_indices[kv_indptr[b].item():kv_indptr[b + 1].item()].to(torch.int64)  # [L_tokens]

            # Prepare transposed Kc/Kp for dot: Kc.T and Kp.T
            KcT = Kc[tok_idx].transpose(0, 1).contiguous()  # [D, L_tokens]
            KpT = Kp[tok_idx].transpose(0, 1).contiguous()  # [Dp, L_tokens]

            # Compute logits_qn and logits_qp via scalar kernels (GEMV-like but Triton-friendly)
            logits_qn = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)
            logits_qp = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)

            grid_h = (H,)
            compute_logits_qn_qp_kernel[grid_h](
                qn[b],           # [H, D]
                qp[b],           # [H, Dp]
                KcT,             # [D, L_tokens]
                KpT,             # [Dp, L_tokens]
                logits_qn,       # [H*L_tokens]
                logits_qp,       # [H*L_tokens]
                H,
                D=512,
                Dp=64,
                L_tokens=L_tokens,
                num_warps=4,
            )

            # Scale and combine
            logits_scaled = (logits_qn + logits_qp) * (self.sm_scale if sm_scale is None else float(sm_scale))

            # Compute lse per head
            compute_lse_reduction_kernel[grid_h](
                logits_scaled,   # [H*L_tokens]
                lse[b],          # [H]
                H,
                L_tokens,
                BLOCK=128,
                num_warps=2,
            )

            # Compute output per head
            output_b = torch.empty((H * D,), dtype=torch.float32, device=device)
            compute_output_matvec_kernel[grid_h](
                logits_scaled,   # [H*L_tokens]
                Kc[tok_idx],     # [L_tokens, D]
                output_b,        # [H*D]
                H,
                D=512,
                L_tokens=L_tokens,
                BLOCK=128,
                num_warps=4,
            )

            output[b] = output_b.view(H, D)

        # Return in original expected dtypes: output in bfloat16, lse in float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
