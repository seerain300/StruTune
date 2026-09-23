import torch
import math
import triton
import triton.language as tl


@triton.jit
def matvec_logits_kernel(
    qn_ptr,         # *float32, [H, Kc_dim]
    Kc_ptr,         # *float32, [M, Kc_dim]
    Kp_ptr,         # *float32, [M, Kp_dim]
    qp_ptr,         # *float32, [H, Kp_dim]
    logits_ptr,     # *float32, [H, M]
    H: tl.constexpr,            # num heads
    M,                          # number of tokens (runtime int)
    Kc_dim: tl.constexpr,       # 512
    Kp_dim: tl.constexpr,       # 64
    sm_scale,                   # float32 scalar
):
    # one program per head
    h = tl.program_id(0)
    for i in range(0, M):
        # accumulate score over Kc_dim
        score = 0.0
        for k in range(0, Kc_dim):
            q = tl.load(qn_ptr + h * Kc_dim + k)  # qn[h, k]
            kc = tl.load(Kc_ptr + i * Kc_dim + k)  # Kc[i, k]
            score += q * kc
        # accumulate p_score over Kp_dim
        p_score = 0.0
        for k in range(0, Kp_dim):
            qp_elem = tl.load(qp_ptr + h * Kp_dim + k)  # qp[h, k]
            kp_elem = tl.load(Kp_ptr + i * Kp_dim + k)  # Kp[i, k]
            p_score += qp_elem * kp_elem
        # store logits_scaled[h, i] = score + p_score * sm_scale
        tl.store(logits_ptr + h * M + i, score + p_score * sm_scale)


@triton.jit
def attention_and_out_kernel(
    logits_ptr,   # *float32, [H, M]
    Kc_ptr,       # *float32, [M, Kc_dim]
    out_ptr,      # *float32, [H, Kc_dim]
    H: tl.constexpr,
    M,                          # runtime int
    Kc_dim: tl.constexpr,       # 512
    sm_scale,                   # float32 scalar
):
    # one program per head
    h = tl.program_id(0)
    # load logits for this head
    logits_row = tl.load(logits_ptr + h * M + tl.arange(0, M))
    # compute max for numerical stability
    max_val = tl.max(logits_row, axis=0)
    x = (logits_row - max_val) * sm_scale
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    inv_sum = 1.0 / sum_exp

    # compute out[h, :] = sum_i exp((logits[h,i]-max)*sm_scale) * Kc[i,:] * inv_sum
    out_vec = tl.zeros((Kc_dim,), dtype=tl.float32)

    BLOCK_M = 128
    for start in range(0, M, BLOCK_M):
        idx = start + tl.arange(0, BLOCK_M)
        mask = idx < M
        logits_chunk = tl.load(logits_ptr + h * M + idx, mask=mask, other=-float('inf'))
        x_chunk = (logits_chunk - max_val) * sm_scale
        exp_chunk = tl.exp(x_chunk)
        out_contrib = tl.zeros((BLOCK_M, Kc_dim), dtype=tl.float32)
        for kk in range(0, Kc_dim):
            kc_col = tl.load(Kc_ptr + idx * Kc_dim + kk, mask=mask, other=0.0)
            out_contrib[:, kk] = exp_chunk * kc_col * inv_sum
        out_vec += tl.sum(out_contrib, axis=0)

    tl.store(out_ptr + h * Kc_dim + tl.arange(0, Kc_dim), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Shape checks to mirror original assumptions
        H = q_nope.shape[1]
        Kc_dim = q_nope.shape[2]
        Kp_dim = q_pe.shape[2]
        assert H == 16, "num_qo_heads must be 16"
        assert Kc_dim == 512, "head_dim_ckv must be 512"
        assert Kp_dim == 64, "head_dim_kpe must be 64"

        batch_size = q_nope.shape[0]
        assert kv_indptr.numel() == batch_size + 1, "kv_indptr length must be batch_size + 1"

        output = torch.empty((batch_size, H, Kc_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            tok_idx = kv_indices[start:end].to(torch.long)
            Kc = ckv_cache[tok_idx, 0, :].to(torch.float32)  # [M, 512]
            Kp = kpe_cache[tok_idx, 0, :].to(torch.float32)  # [M, 64]
            qn = q_nope[b].to(torch.float32)                 # [16, 512]
            qp = q_pe[b].to(torch.float32)                  # [16, 64]

            logits = torch.empty((H, M), dtype=torch.float32, device=device)

            matvec_logits_kernel[(H,)](
                qn, Kc, Kp, qp, logits,
                H=H, M=M, Kc_dim=Kc_dim, Kp_dim=Kp_dim, sm_scale=sm_scale
            )

            # Recompute lse from unscaled logits to match original: lse = logsumexp((qn@Kc.T + qp@Kp.T)) / ln(2)
            logits_unscaled = (qn @ Kc.T) + (qp @ Kp.T)  # [H, M]
            max_val = torch.max(logits_unscaled, dim=1, keepdim=True).values
            lse_sub = torch.logsumexp((logits_unscaled - max_val) * sm_scale, dim=1) / math.log(2.0)
            lse[b] = lse_sub

            out = torch.empty((H, Kc_dim), dtype=torch.float32, device=device)
            attention_and_out_kernel[(H,)](
                logits, Kc, out,
                H=H, M=M, Kc_dim=Kc_dim, sm_scale=sm_scale
            )
            output[b] = out

        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
