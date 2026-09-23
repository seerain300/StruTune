import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          L_tokens: tl.constexpr, Dc: tl.constexpr):
    # Each program handles one token row for CKV cache
    pid = tl.program_id(0)
    if pid >= L_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    # Copy the row of length Dc into out_ptr at offset pid*Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


@triton.jit
def gather_rows_p_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          L_tokens: tl.constexpr, Dp: tl.constexpr):
    # Each program handles one token row for KPE cache
    pid = tl.program_id(0)
    if pid >= L_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr, L: tl.constexpr):
    # Compute softmax over L tokens for a single "row" (conceptually per head).
    m = -float("inf")
    for t in range(0, L):
        val = tl.load(logits_ptr + t)
        m = tl.maximum(m, val)
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logits_ptr + t)
        sum_exp += tl.exp(val - m)
    for t in range(0, L):
        val = tl.load(logits_ptr + t)
        attn = tl.exp(val - m) / sum_exp
        tl.store(attn_ptr + t, attn)


@triton.jit
def matvec_kernel(attn_ptr, K_ptr, out_ptr,
                   H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr):
    # Grid over heads and blocks of Dc; compute out[head, :] = attn[head, :] @ K[:, :]
    pid_h = tl.program_id(0)  # head index
    pid_d = tl.program_id(1)  # block index over Dc
    d_start = pid_d * 64
    # Accumulator for this block
    acc = 0.0
    for k in range(0, L):
        a = tl.load(attn_ptr + k)  # scalar attention for token k
        base_k = k * Dc
        for j in range(d_start, d_start + 64):
            if j >= Dc:
                break
            k_val = tl.load(K_ptr + base_k + j)  # scalar key vector element
            acc += a * k_val
    # Store to out_ptr at offsets for this head
    for j in range(d_start, d_start + 64):
        if j >= Dc:
            break
        tl.store(out_ptr + pid_h * Dc + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device and contiguity
        device = q_nope.device

        # Preprocess caches: squeeze size-1 dims and move to float32
        ckv_cache_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [P, Dc]
        kpe_cache_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [P, Dp]
        assert ckv_cache_all.shape[1] == 512, "ckv_cache second dim must be 512"
        assert kpe_cache_all.shape[1] == 64, "kpe_cache second dim must be 64"

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                for i in range(num_qo_heads):
                    output[b, i] = torch.zeros((head_dim_ckv,), dtype=torch.bfloat16, device=device)
                lse[b] = torch.zeros((num_qo_heads,), dtype=torch.float32, device=device)
                continue

            # Token indices for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather rows from caches into Kc_flat and Kp_flat (float32)
            Kc_flat = torch.empty((L_tokens * head_dim_ckv,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * head_dim_kpe,), dtype=torch.float32, device=device)

            grid_gather = (L_tokens,)
            gather_rows_c_kernel[grid_gather](ckv_cache_all, tok_idx, Kc_flat, L_tokens, head_dim_ckv)
            Kc = Kc_flat.view(L_tokens, head_dim_ckv)  # [L_tokens, 512]

            gather_rows_p_kernel[grid_gather](kpe_cache_all, tok_idx, Kp_flat, L_tokens, head_dim_kpe)
            Kp = Kp_flat.view(L_tokens, head_dim_kpe)  # [L_tokens, 64]

            # 2) For each head i:
            for i in range(num_qo_heads):
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [512]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [64]
                # Compute logits for this head: qn @ Kc.T + qp @ Kp.T
                logits_qn = qn @ Kc.T                         # [1, L_tokens]
                logits_qp = qp @ Kp.T                         # [1, L_tokens]
                logits = (logits_qn + logits_qp).squeeze(0)   # [L_tokens]
                logits_scaled = logits * sm_scale             # [L_tokens]

                # Compute lse per head using torch for robustness: base-2 logsumexp
                lse[b, i] = torch.logsumexp(logits_scaled, dim=0) / math.log(2.0)

                # 3) Compute attention weights via Triton softmax: attn over L_tokens
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_row_kernel[(1,)](logits_scaled, attn, L_tokens)

                # 4) Final projection: attn @ Kc -> [512], use Triton matvec
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                # Prepare 1D attn and Kc for matvec
                attn_1d = attn.contiguous().view(-1)               # [L_tokens]
                Kc_1d = Kc.contiguous().view(-1)                  # [L_tokens * Dc]
                grid_m = (1, triton.cdiv(head_dim_ckv, 64))      # grid over heads=1, blocks over Dc
                matvec_kernel[grid_m](attn_1d, Kc_1d, out_vec, 1, head_dim_ckv, L_tokens)
                # Store output[b, i] as bfloat16
                output[b, i] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
