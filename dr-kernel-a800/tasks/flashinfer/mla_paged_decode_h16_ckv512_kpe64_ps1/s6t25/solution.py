import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_scaled_per_batch_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, Triton output buffer [H*L_tokens], contiguous
    H,              # int32
    D: tl.constexpr,          # 512
    Dp: tl.constexpr,         # 64
    L_tokens,       # int32
    sm_scale,       # float32
):
    # For each head h, compute logits for all tokens t
    for h in range(0, H):
        base = h * L_tokens
        # Precompute qn and qp vectors
        qn_vec = [0.0] * D
        for kk in range(0, D):
            qn_vec[kk] = tl.load(qn_ptr + h * D + kk)
        qp_vec = [0.0] * Dp
        for kk in range(0, Dp):
            qp_vec[kk] = tl.load(qp_ptr + h * Dp + kk)
        for t in range(0, L_tokens):
            acc1 = 0.0
            for kk in range(0, D):
                acc1 += qn_vec[kk] * tl.load(Kc_ptr + t * D + kk)
            acc2 = 0.0
            for kk in range(0, Dp):
                acc2 += qp_vec[kk] * tl.load(Kp_ptr + t * Dp + kk)
            logit = (acc1 + acc2) * sm_scale
            tl.store(logits_ptr + base + t, logit)


@triton.jit
def compute_lse_per_batch_kernel(
    logits_ptr,     # *fp32, contiguous [H*L_tokens]
    lse_ptr,        # *fp32, contiguous [H]
    H,              # int32
    L_tokens,       # int32
):
    # One program per head
    h = tl.program_id(0)
    base = h * L_tokens
    m = -float("inf")
    for t in range(0, L_tokens):
        x = tl.load(logits_ptr + base + t)
        m = tl.maximum(m, x)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        x = tl.load(logits_ptr + base + t)
        sum_exp += tl.exp(x - m)
    lse_val = m + tl.log(sum_exp)  # natural logsumexp
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def compute_output_per_batch_kernel(
    logits_ptr,     # *fp32, contiguous [H*L_tokens]
    Kc_ptr,         # *fp32, contiguous [L_tokens, D]
    out_ptr,        # *fp32, Triton output buffer [H*D], contiguous
    H,              # int32
    D: tl.constexpr,          # 512
    L_tokens,       # int32
):
    # One program per head; compute output[h, :] = softmax(logits[h, :]) @ Kc[:, :]
    h = tl.program_id(0)
    base_logits = h * L_tokens
    m = -float("inf")
    for t in range(0, L_tokens):
        x = tl.load(logits_ptr + base_logits + t)
        m = tl.maximum(m, x)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        x = tl.load(logits_ptr + base_logits + t)
        sum_exp += tl.exp(x - m)
    for j in range(0, D):
        acc = 0.0
        for t in range(0, L_tokens):
            x = tl.load(logits_ptr + base_logits + t)
            p = tl.exp(x - m) / sum_exp
            kv = tl.load(Kc_ptr + t * D + j)
            acc += p * kv
        tl.store(out_ptr + h * D + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure all inputs are CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton kernels."

        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"

        head_dim_kpe = q_pe.shape[-1]
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Extract Kc_all and Kp_all: [num_pages, head_dim]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

        # Prepare outputs
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=q_nope.device)

        # Process each batch element in the host to match PyTorch behavior exactly
        for b in range(batch_size):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                lse[b] = torch.full((num_qo_heads,), -float("inf"), dtype=torch.float32, device=q_nope.device)
                output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
                continue

            # Token indices for this batch element
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].contiguous()  # [L_tokens]
            Kc = Kc_all[tok_idx].contiguous()  # [L_tokens, 512], fp32
            Kp = Kp_all[tok_idx].contiguous()  # [L_tokens, 64], fp32

            # Prepare qn and qp for this batch: fp32 for Triton math
            qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

            # Allocate Triton buffers (fp32) for intermediate results
            logits_flat = torch.empty(L_tokens * num_qo_heads, dtype=torch.float32, device=q_nope.device)
            lse_buf = torch.empty(num_qo_heads, dtype=torch.float32, device=q_nope.device)
            out_flat = torch.empty(num_qo_heads * head_dim_ckv, dtype=torch.float32, device=q_nope.device)

            # Launch Triton kernel to compute logits_scaled[h, t]
            compute_logits_scaled_per_batch_kernel[(1,)](
                qn, qp, Kc, Kp, logits_flat,
                num_qo_heads, 512, 64, L_tokens, sm_scale
            )

            # Compute lse[h] = logsumexp(logits_scaled[h, :]) / ln(2)
            compute_lse_per_batch_kernel[(num_qo_heads,)](logits_flat, lse_buf, num_qo_heads, L_tokens)
            lse[b] = lse_buf / math.log(2.0)

            # Compute output[h, :] = softmax(logits_scaled[h, :]) @ Kc[:, :]
            compute_output_per_batch_kernel[(num_qo_heads,)](logits_flat, Kc, out_flat, num_qo_heads, 512, L_tokens)

            # Cast output to bfloat16 for final return
            out_mat = out_flat.view(num_qo_heads, head_dim_ckv)
            output[b] = out_mat.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
