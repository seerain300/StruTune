import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.constexpr, Dc: tl.constexpr):
    # Each program handles one token row: copy ckv_cache[tok_idx[pid], :] into out[pid, :]
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


@triton.jit
def gather_rows_p_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.constexpr, Dp: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


@triton.jit
def lse_base2_kernel(logits_scaled_ptr, lse_ptr,
                     H: tl.constexpr, L: tl.constexpr):
    # One program per head i; compute lse[i] = logsumexp(logits_scaled[i, :]) / ln(2)
    i = tl.program_id(0)
    m = -float("inf")
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logits_scaled_ptr + i * L + t)
        m = tl.maximum(m, val)
        sum_exp += tl.exp(val - m)
    lse_val = m + math.log(sum_exp) / math.log(2.0)
    tl.store(lse_ptr + i, lse_val)


@triton.jit
def softmax_row_kernel(row_ptr_in, row_ptr_out,
                        L: tl.constexpr):
    # One program per row: compute softmax over L elements
    i = tl.program_id(0)
    # Load row
    row = tl.zeros([L], dtype=tl.float32)
    for t in range(0, L):
        row[t] = tl.load(row_ptr_in + i * L + t)
    # Max for numerical stability
    m = tl.max(row, axis=0)
    exp_row = tl.exp(row - m)
    sum_exp = tl.sum(exp_row, axis=0)
    for t in range(0, L):
        tl.store(row_ptr_out + i * L + t, exp_row[t] / sum_exp)


@triton.jit
def matvec_kernel(attn_ptr, Kc_ptr, out_ptr,
                  H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr):
    # One program per head
    i = tl.program_id(0)
    # Initialize output vector
    for d in range(0, Dc):
        tl.store(out_ptr + i * Dc + d, 0.0)
    # Compute out[i, d] = sum_{t=0..L-1} attn[i, t] * Kc[t, d]
    for d in range(0, Dc):
        acc_d = 0.0
        for t in range(0, L):
            attn_t = tl.load(attn_ptr + i * L + t)
            Kc_t_d = tl.load(Kc_ptr + t * Dc + d)
            acc_d += attn_t * Kc_t_d
        tl.store(out_ptr + i * Dc + d, acc_d)


@triton.jit
def matmul_qn_KcT_kernel(qn_ptr, Kc_ptr, out_ptr,
                         H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr):
    # Compute logits_qn[i, t] = dot(qn[i, :], Kc[t, :]) for t in [0..L-1]
    i = tl.program_id(0)
    for t in range(0, L):
        acc = 0.0
        for k in range(0, Dc):
            qn_k = tl.load(qn_ptr + i * Dc + k)
            Kc_t_k = tl.load(Kc_ptr + t * Dc + k)
            acc += qn_k * Kc_t_k
        tl.store(out_ptr + i * L + t, acc)


@triton.jit
def matmul_qp_KpT_kernel(qp_ptr, Kp_ptr, out_ptr,
                         H: tl.constexpr, Dp: tl.constexpr, L: tl.constexpr):
    # Compute logits_qp[i, t] = dot(qp[i, :], Kp[t, :]) for t in [0..L-1]
    i = tl.program_id(0)
    for t in range(0, L):
        acc = 0.0
        for k in range(0, Dp):
            qp_k = tl.load(qp_ptr + i * Dp + k)
            Kp_t_k = tl.load(Kp_ptr + t * Dp + k)
            acc += qp_k * Kp_t_k
        tl.store(out_ptr + i * L + t, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton kernels."
        device = q_nope.device

        # Shapes
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        assert ckv_cache.shape == (num_pages, 1, 512), "ckv_cache must have shape [num_pages, 1, 512]"
        assert kpe_cache.shape == (num_pages, 1, 64), "kpe_cache must have shape [num_pages, 1, 64]"
        assert kv_indptr.shape[0] == batch_size + 1, "kv_indptr length must be batch_size + 1"

        # Prepare caches: squeeze size-1 dim and cast to float32 for compute
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [P, Dp]

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # For each batch element
        for b in range(batch_size):
            # Derive token range using kv_indptr
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg

            if L_tokens <= 0:
                lse[b].zero_()
                output[b].zero_()
                continue

            # Gather token indices and corresponding cache rows
            tok_idx = kv_indices[page_beg:page_end].contiguous().to(torch.int32)  # [L_tokens]

            # Triton gather for Kc rows and Kp rows
            Kc_flat = torch.empty((L_tokens * head_dim_ckv,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * head_dim_kpe,), dtype=torch.float32, device=device)

            grid_g = (L_tokens,)
            gather_rows_c_kernel[grid_g](Kc_all.view(-1), tok_idx, Kc_flat, num_tokens=L_tokens, Dc=head_dim_ckv)
            Kc = Kc_flat.view(L_tokens, head_dim_ckv)  # [L_tokens, 512]

            gather_rows_p_kernel[grid_g](Kp_all.view(-1), tok_idx, Kp_flat, num_tokens=L_tokens, Dp=head_dim_kpe)
            Kp = Kp_flat.view(L_tokens, head_dim_kpe)  # [L_tokens, 64]

            # q vectors per head: cast to float32 for compute
            qn = q_nope[b].contiguous().to(torch.float32)  # [H, 512]
            qp = q_pe[b].contiguous().to(torch.float32)    # [H, 64]

            # Allocate logits outputs for each head
            logits_qn_flat = torch.empty((num_qo_heads * L_tokens,), dtype=torch.float32, device=device)
            logits_qp_flat = torch.empty((num_qo_heads * L_tokens,), dtype=torch.float32, device=device)

            # Compute qn @ Kc.T and qp @ Kp.T per head using Triton
            grid_m = (num_qo_heads,)
            for i in range(num_qo_heads):
                matmul_qn_KcT_kernel[grid_m](
                    qn[i].contiguous().view(-1), Kc.contiguous().view(-1),
                    logits_qn_flat[i * L_tokens:], H=num_qo_heads, Dc=head_dim_ckv, L=L_tokens
                )
                matmul_qp_KpT_kernel[grid_m](
                    qp[i].contiguous().view(-1), Kp.contiguous().view(-1),
                    logits_qp_flat[i * L_tokens:], H=num_qo_heads, Dp=head_dim_kpe, L=L_tokens
                )

            # Sum to get full logits per head: shape [H, L_tokens] flattened to [H*L_tokens]
            logits_flat = logits_qn_flat + logits_qp_flat  # [H*L_tokens]
            logits = logits_flat.view(num_qo_heads, L_tokens)  # [H, L_tokens]

            # Scale logits
            logits_scaled = logits * sm_scale  # [H, L_tokens]

            # Compute lse per head in base-2: use Triton kernel
            lse[b] = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
            lse_base2_kernel[grid_m](logits_scaled.view(-1), lse[b], H=num_qo_heads, L=L_tokens)

            # Compute attention weights via Triton softmax
            attn = torch.empty((num_qo_heads, L_tokens), dtype=torch.float32, device=device)
            softmax_row_kernel[grid_m](
                logits_scaled.view(-1), attn.view(-1), L=L_tokens
            )

            # Final projection: attn @ Kc -> [H, 512], use Triton matvec
            out_flat = torch.empty((num_qo_heads * head_dim_ckv,), dtype=torch.float32, device=device)
            matvec_kernel[grid_m](
                attn.contiguous().view(-1), Kc.contiguous().view(-1), out_flat,
                H=num_qo_heads, Dc=head_dim_ckv, L=L_tokens
            )
            out_vec = out_flat.view(num_qo_heads, head_dim_ckv)  # [H, 512]

            # Store to output[b, i] as bfloat16
            for i in range(num_qo_heads):
                output[b, i] = out_vec[i].to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
