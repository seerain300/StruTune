import math
import torch
import triton
import triton.language as tl


# Triton kernel: gather rows from a flattened cache into out
# cache: [num_pages * D], tok_idx: [num_tokens], out: [num_tokens * D]
@triton.jit
def gather_rows_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                        num_tokens: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * D
    for k in range(0, D):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * D + k, val)


# Triton kernel: compute per-head lse = logsumexp(logit_row) / ln(2), one program per head
# logit_row_ptr: base pointer for this head; we pass it as the address for head i.
# lse_ptr: [H], float32, output lse per head.
@triton.jit
def lse_base2_row_kernel(logit_row_ptr, lse_ptr,
                         L: tl.constexpr):
    i = tl.program_id(0)  # head index
    m = -float("inf")
    # Pass 1: compute max over L tokens
    for t in range(0, L):
        val = tl.load(logit_row_ptr + t)
        m = tl.maximum(m, val)
    # Pass 2: compute sum exp
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logit_row_ptr + t)
        sum_exp += tl.exp(val - m)
    # lse_base2 = m + log(sum_exp) / ln(2)
    lse_val = m + (math.log(sum_exp) / math.log(2.0))
    tl.store(lse_ptr + i, lse_val)


# Triton kernel: softmax over a row, one program per head
# logit_row_ptr: base pointer for this head
# attn_row_ptr: [H*L_tokens], float32, output softmax
@triton.jit
def softmax_row_kernel(logit_row_ptr, attn_row_ptr,
                        L: tl.constexpr):
    i = tl.program_id(0)  # head index
    m = -float("inf")
    # pass 1: max
    for t in range(0, L):
        val = tl.load(logit_row_ptr + t)
        m = tl.maximum(m, val)
    # pass 2: sum of exp
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logit_row_ptr + t)
        sum_exp += tl.exp(val - m)
    # pass 3: write normalized
    for t in range(0, L):
        val = tl.load(logit_row_ptr + t)
        attn = tl.exp(val - m) / sum_exp
        tl.store(attn_row_ptr + i * L + t, attn)


# Triton kernel: matvec y = x @ A where x is [L], A is [L, D], y is [D], one program per head
# attn_flat: [H*L], K_flat: [L*D], out_flat: [H*D]
@triton.jit
def matvec_kernel(attn_flat_ptr, K_flat_ptr, out_flat_ptr,
                  H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                  BLOCK_D: tl.constexpr):
    i = tl.program_id(0)  # head index
    # y[i, :] = sum_{t=0..L-1} attn[i, t] * K[t, :]
    for d in range(0, D, BLOCK_D):
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for t in range(0, L):
            base_x = i * L + t  # attn[i, t]
            x = tl.load(attn_flat_ptr + base_x)
            base_K = t * D + d  # K[t, d:d+BLOCK_D]
            k = tl.load(K_flat_ptr + base_K + tl.arange(0, BLOCK_D), mask=tl.arange(0, BLOCK_D) < D - d, other=0.0)
            acc += x * k
        out_base = i * D + d
        tl.store(out_flat_ptr + out_base + tl.arange(0, BLOCK_D), acc, mask=tl.arange(0, BLOCK_D) < D - d)


# Triton kernel: compute logits for a single head i:
# Inputs: qn_flat: [Dc], qp_flat: [Dp], Kc_flat: [L*Dc], Kp_flat: [L*Dp]
# Outputs: logits_out: [L], out_vec: [Dc]
# Assumes grid=(1,) — one program per head. It loops over tokens to compute dot products.
@triton.jit
def compute_logits_scaled_and_matvec_kernel(qn_ptr, qp_ptr, Kc_flat_ptr, Kp_flat_ptr, logits_out_ptr, out_vec_ptr,
                                            H: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr, L: tl.constexpr, sm_scale: tl.float32):
    # This kernel is intended to run with grid=(1,), i.e., a single program instance.
    # We will compute logits per token and then use a separate Triton softmax kernel,
    # and then compute out_vec via a matvec kernel. Here we just compute logits.
    # However, Triton requires a grid, so we emulate a single head by using H as the head id.
    # Note: In the forward, we will launch this kernel with grid=(1,) and pass q_nope[b, i] and q_pe[b, i] as qn_ptr, qp_ptr.
    # We'll compute logits vector:
    for t in range(0, L):
        # Compute dot(qn, Kc[t, :]) and dot(qp, Kp[t, :])
        dot_qn = 0.0
        dot_qp = 0.0
        # qn is length Dc, Kc_flat is length L*Dc, stride Dc
        base_qn = 0
        base_Kc = t * Dc
        for k in range(0, Dc):
            qvk = tl.load(qn_ptr + base_qn + k)
            kvk = tl.load(Kc_flat_ptr + base_Kc + k)
            dot_qn += qvk * kvk
        # Similarly for qp and Kp
        base_qp = 0
        base_Kp = t * Dp
        for k in range(0, Dp):
            qvk = tl.load(qp_ptr + base_qp + k)
            kvk = tl.load(Kp_flat_ptr + base_Kp + k)
            dot_qp += qvk * kvk
        logits_val = dot_qn + dot_qp
        # Scale
        logits_val = logits_val * sm_scale
        tl.store(logits_out_ptr + t, logits_val)

    # After computing logits_out, we would normally call softmax_row_kernel and matvec_kernel here.
    # However, since this is a single-program kernel and we need to return out_vec, we need to
    # compute softmax and matvec. Triton lacks multi-program coordination, so we instead
    # return control to forward which will use separate kernels. For correctness, we won't
    # write out_vec here. The forward will compute attn via softmax and then call matvec.
    # Thus, this kernel only computes logits_out. out_vec will be computed by forward using
    # softmax_row_kernel and matvec_kernel.


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # All tensors must be on CUDA for Triton
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be on CUDA for Triton kernels."
        device = q_nope.device

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        assert num_qo_heads == self.num_qo_heads, "num_qo_heads must be 16."

        head_dim_ckv = self.head_dim_ckv
        head_dim_kpe = self.head_dim_kpe
        num_pages = ckv_cache.shape[0]
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1

        # Squeeze caches to remove size-1 dim and cast to float32 for compute
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Dp]

        # Output tensors
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Determine number of tokens for this batch item
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                for i in range(num_qo_heads):
                    output[b, i] = torch.zeros((head_dim_ckv,), dtype=torch.bfloat16, device=device)
                lse[b] = torch.zeros((num_qo_heads,), dtype=torch.float32, device=device)
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens], int32

            # 1) Gather rows from caches into Kc_flat and Kp_flat (float32)
            Kc_flat = torch.empty((L_tokens * head_dim_ckv,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * head_dim_kpe,), dtype=torch.float32, device=device)

            grid_gather = (L_tokens,)
            # Gather CKV rows
            gather_rows_kernel[grid_gather](
                Kc_all, tok_idx, Kc_flat,
                num_tokens=L_tokens, D=head_dim_ckv
            )
            Kc = Kc_flat.view(L_tokens, head_dim_ckv)  # [L_tokens, Dc]

            # Gather KPE rows
            gather_rows_kernel[grid_gather](
                Kp_all, tok_idx, Kp_flat,
                num_tokens=L_tokens, D=head_dim_kpe
            )
            Kp = Kp_flat.view(L_tokens, head_dim_kpe)  # [L_tokens, Dp]

            # For each head i: compute logits_scaled, attn, and out_vec using Triton
            for i in range(num_qo_heads):
                # 2) Compute logits_scaled for head i using Triton kernel:
                # Prepare qn, qp as 1D tensors
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]

                # Allocate logits_out
                logits_out = torch.empty((L_tokens,), dtype=torch.float32, device=device)

                # Launch compute_logits_scaled_and_matvec_kernel: grid=(1,) — one head per launch
                compute_logits_scaled_and_matvec_kernel[(1,)](
                    qn, qp, Kc_flat, Kp_flat, logits_out,
                    H=num_qo_heads, Dc=head_dim_ckv, Dp=head_dim_kpe, L=L_tokens, sm_scale=float(sm_scale)
                )

                # 3) Triton softmax over logits_out to get attn
                attn_flat = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_row_kernel[(num_qo_heads,)](logits_out, attn_flat, L=L_tokens)

                # 4) Triton lse in base-2 for this head (per-row kernel expects row of length L)
                lse[b, i] = torch.empty((), dtype=torch.float32, device=device)  # will be overwritten
                lse_base2_row_kernel[(1,)](logits_out, lse[b], L=L_tokens)

                # 5) Final projection: attn @ Kc -> [Dc], Triton matvec
                out_flat = torch.empty((num_qo_heads * head_dim_ckv,), dtype=torch.float32, device=device)
                matvec_kernel[(num_qo_heads,)](
                    attn_flat, Kc_flat, out_flat,
                    H=num_qo_heads, D=head_dim_ckv, L=L_tokens,
                    BLOCK_D=128
                )
                out_vec = out_flat.view(num_qo_heads, head_dim_ckv)  # [H, Dc]
                # Store to output[b, i] as bfloat16
                output[b, i] = out_vec[i].to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
