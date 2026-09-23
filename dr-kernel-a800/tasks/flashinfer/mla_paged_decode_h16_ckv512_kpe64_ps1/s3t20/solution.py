import torch
import triton
import triton.language as tl


# Fused logits: logits[h, t] = qn[h, :] @ Kc[t, :] + qp[h, :] @ Kp[t, :]
@triton.jit
def fused_logits_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits_ptr,
    H: tl.constexpr,  # 16
    Dq: tl.constexpr, # 512
    Dp: tl.constexpr, # 64
    T: tl.constexpr,  # number of tokens
    BLOCK_T: tl.constexpr,  # power-of-two, e.g., 128
):
    h = tl.program_id(0)  # one program per head
    off_dq = tl.arange(0, Dq)  # [Dq]
    off_dp = tl.arange(0, Dp)  # [Dp]

    qnh = tl.load(qn_ptr + h * Dq + off_dq)  # [Dq]
    qph = tl.load(qp_ptr + h * Dp + off_dp)  # [Dp]

    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)  # [BLOCK_T], power-of-two
        mask_t = offs_t < T

        Kc_sub = tl.load(
            Kc_ptr + offs_t[:, None] * Dq + off_dq[None, :],
            mask=mask_t[:, None],
            other=0.0
        )  # [BLOCK_T, Dq]
        Kp_sub = tl.load(
            Kp_ptr + offs_t[:, None] * Dp + off_dp[None, :],
            mask=mask_t[:, None],
            other=0.0
        )  # [BLOCK_T, Dp]

        dot1 = tl.sum(qnh[None, :] * Kc_sub, axis=1)  # [BLOCK_T]
        dot2 = tl.sum(qph[None, :] * Kp_sub, axis=1)  # [BLOCK_T]

        logits_row = dot1 + dot2
        tl.store(logits_ptr + h * T + offs_t, logits_row, mask=mask_t)


# Row-wise softmax: attn[h, :] = softmax(logits[h, :] * sm_scale)
@triton.jit
def softmax_row_kernel(
    logits_ptr, attn_ptr, sm_scale,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,  # power-of-two, e.g., 128
):
    h = tl.program_id(0)
    offs = tl.arange(0, BLOCK_T)
    logits_row = tl.load(logits_ptr + h * T + offs, mask=offs < T, other=-float('inf'))
    scaled = logits_row * sm_scale
    max_val = tl.max(scaled, axis=0)
    scaled = scaled - max_val
    exps = tl.exp(scaled)
    sum_val = tl.sum(exps, axis=0)
    attn_row = exps / sum_val
    tl.store(attn_ptr + h * T + offs, attn_row, mask=offs < T)


# Row-wise logsumexp: lse[h] = logsumexp(logits[h, :] * sm_scale) / ln(2)
@triton.jit
def lse_row_kernel(
    logits_ptr, lse_ptr, sm_scale,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,  # power-of-two, e.g., 128
):
    h = tl.program_id(0)
    offs = tl.arange(0, BLOCK_T)
    logits_row = tl.load(logits_ptr + h * T + offs, mask=offs < T, other=-float('inf'))
    scaled = logits_row * sm_scale
    max_val = tl.max(scaled, axis=0)
    scaled = scaled - max_val
    sum_exp = tl.sum(tl.exp(scaled), axis=0)
    lse_val = tl.log(sum_exp) + max_val  # logsumexp
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse_val = lse_val / ln2
    tl.store(lse_ptr + h, lse_val)


# Per-head matmul: out[h, :] = attn[h, :] @ Kc[:, :]
@triton.jit
def matmul_row_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H: tl.constexpr,
    T: tl.constexpr,
    Dq: tl.constexpr,
    BLOCK_T: tl.constexpr,  # power-of-two, e.g., 128
):
    h = tl.program_id(0)
    off_dq = tl.arange(0, Dq)
    out_row = tl.zeros((Dq,), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        attn_sub = tl.load(attn_ptr + h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
        Kc_sub = tl.load(
            Kc_ptr + offs_t[:, None] * Dq + off_dq[None, :],
            mask=mask_t[:, None],
            other=0.0
        )  # [BLOCK_T, Dq]
        partial = tl.sum(Kc_sub * attn_sub[:, None], axis=0)  # [Dq]
        out_row += partial
    tl.store(out_ptr + h * Dq + off_dq, out_row)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.H = 16
        self.Dq = 512
        self.Dp = 64

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, use_triton=None):
        # Ignore use_triton; keep 8 args to satisfy harness
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device

        B = q_nope.shape[0]
        assert q_nope.shape[1] == self.H and q_nope.shape[2] == self.Dq
        assert q_pe.shape[1] == self.H and q_pe.shape[2] == self.Dp

        output = torch.empty((B, self.H, self.Dq), dtype=torch.float32, device=device)
        lse = torch.empty((B, self.H), dtype=torch.float32, device=device)

        for b in range(B):
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                output[b].zero_()
                lse[b].zero_()
                continue

            L_tokens = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end]  # [L_tokens]

            # Load Kc and Kp for these tokens
            Kc_b = ckv_cache[tok_idx, 0].to(torch.float32)  # [L_tokens, 512]
            Kp_b = kpe_cache[tok_idx, 0].to(torch.float32)  # [L_tokens, 64]

            # q_nope[b] and q_pe[b]
            qn = q_nope[b].to(torch.float32)  # [16, 512]
            qp = q_pe[b].to(torch.float32)    # [16, 64]

            # Allocate logits [H, T]
            logits = torch.empty((self.H, L_tokens), dtype=torch.float32, device=device)

            # Fused logits
            fused_logits_kernel[(self.H,)](
                qn, qp, Kc_b, Kp_b, logits,
                H=self.H, Dq=self.Dq, Dp=self.Dp, T=L_tokens,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Softmax per row
            attn = torch.empty_like(logits, dtype=torch.float32, device=device)
            softmax_row_kernel[(self.H,)](
                logits, attn, float(sm_scale),
                T=L_tokens, BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Logsumexp per row
            lse_row_kernel[(self.H,)](
                logits, lse[b], float(sm_scale),
                T=L_tokens, BLOCK_T=128,
                num_warps=1, num_stages=1
            )

            # Per-head output matmul
            for h in range(self.H):
                out_row = torch.empty((self.Dq,), dtype=torch.float32, device=device)
                matmul_row_kernel[(1,)](
                    attn[h], Kc_b, out_row,
                    H=self.H, T=L_tokens, Dq=self.Dq, BLOCK_T=128,
                    num_warps=4, num_stages=2
                )
                output[b, h] = out_row

        return output, lse


def run(*args):
    return ModelNew()(*args)
