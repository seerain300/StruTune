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


# Triton kernel: per-head logsumexp in base-2, one program per head
# logit_row_ptr: [H * L], lse_ptr: [H], float32
@triton.jit
def lse_base2_row_kernel(logit_row_ptr, lse_ptr,
                         H: tl.constexpr, L: tl.constexpr):
    i = tl.program_id(0)  # head index
    m = -float("inf")
    # Pass 1: compute max of logit_row[i, :]
    for t in range(0, L):
        val = tl.load(logit_row_ptr + i * L + t)
        m = tl.maximum(m, val)
    # Pass 2: compute sum exp
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logit_row_ptr + i * L + t)
        sum_exp += tl.exp(val - m)
    # logsumexp / ln(2)
    lse_val = m + (math.log(sum_exp) / math.log(2.0))
    tl.store(lse_ptr + i, lse_val)


# Triton kernel: softmax for a single head over L tokens, one program per head
# attn_ptr: [H * L], logit_row_ptr: [H * L], L: constexpr
@triton.jit
def softmax_row_kernel(attn_ptr, logit_row_ptr,
                       H: tl.constexpr, L: tl.constexpr):
    i = tl.program_id(0)  # head index
    # Load logits for this head
    m = -float("inf")
    for t in range(0, L):
        val = tl.load(logit_row_ptr + i * L + t)
        m = tl.maximum(m, val)
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logit_row_ptr + i * L + t)
        sum_exp += tl.exp(val - m)
    inv_sum = 1.0 / sum_exp
    for t in range(0, L):
        val = tl.load(logit_row_ptr + i * L + t)
        # attn[t] = exp(val - m) / sum_exp
        at = tl.exp(val - m) * inv_sum
        tl.store(attn_ptr + i * L + t, at)


# Triton kernel: matvec for a single head: out_vec[i, :] = attn_row[i, :] @ Kc
# attn_row_ptr: [L], Kc_ptr: [L * D], out_vec_ptr: [H * D], H, D, L, strides
@triton.jit
def matvec_kernel(out_vec_ptr, attn_row_ptr, Kc_ptr,
                  H: tl.constexpr, D: tl.constexpr, L: tl.constexpr):
    i = tl.program_id(0)  # head index
    for d in range(0, D):
        acc = 0.0
        for t in range(0, L):
            at = tl.load(attn_row_ptr + i * L + t)  # attn for this head, t-th token
            Kd = tl.load(Kc_ptr + t * D + d)        # Kc[t, d]
            acc += at * Kd
        # Write out_vec[i, d]
        tl.store(out_vec_ptr + i * D + d, acc)


# Triton kernel: per-head row-wise matmul (row of Q times Kc.T): logits_row[i, t] = sum_d Q[i,d] * Kc[t,d]
# q_ptr: [H * Dq], Kc_ptr: [L * D], logits_ptr: [H * L]
@triton.jit
def matmul_row_kernel(logits_ptr, q_ptr, Kc_ptr,
                      H: tl.constexpr, Dq: tl.constexpr, D: tl.constexpr, L: tl.constexpr):
    i = tl.program_id(0)  # head index
    for t in range(0, L):
        acc = 0.0
        for d in range(0, Dq):
            q_val = tl.load(q_ptr + i * Dq + d)      # q[i, d]
            K_val = tl.load(Kc_ptr + t * D + d)     # Kc[t, d]
            acc += q_val * K_val
        tl.store(logits_ptr + i * L + t, acc)


# Triton kernel: per-head row-wise matmul for Kp: logits_row[i, t] += sum_p Qp[i,p] * Kp[t,p]
# qp_ptr: [H * Dp], Kp_ptr: [L * Dp], logits_ptr: [H * L]
@triton.jit
def matmul_row_kernel_p(logits_ptr, qp_ptr, Kp_ptr,
                        H: tl.constexpr, Dp: tl.constexpr, L: tl.constexpr):
    i = tl.program_id(0)  # head index
    for t in range(0, L):
        acc = 0.0
        for p in range(0, Dp):
            qp_val = tl.load(qp_ptr + i * Dp + p)   # qp[i, p]
            Kp_val = tl.load(Kp_ptr + t * Dp + p)   # Kp[t, p]
            acc += qp_val * Kp_val
        tl.store(logits_ptr + i * L + t, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original assertions
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device/dtype and contiguity
        device = q_nope.device
        B = q_nope.shape[0]
        H = self.num_qo_heads
        Dc = self.head_dim_ckv
        Dp = self.head_dim_kpe

        # Prepare caches
        # ckv_cache: [P, 1, Dc] -> [P, Dc]
        ckv_cache_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()
        # kpe_cache: [P, 1, Dp] -> [P, Dp]
        kpe_cache_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()

        # Output and lse buffers
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV, set output and lse to zeros
                for i in range(H):
                    output[b, i] = torch.zeros((Dc,), dtype=torch.bfloat16, device=device)
                lse[b] = torch.zeros((H,), dtype=torch.float32, device=device)
                continue

            # Gather token indices
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32).contiguous()  # [L_tokens]

            # Gather rows from caches into float32 buffers
            Kc_flat = torch.empty((L_tokens * Dc,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * Dp,), dtype=torch.float32, device=device)

            # Launch gather rows
            grid_gather = (L_tokens,)
            gather_rows_kernel[grid_gather](ckv_cache_all, tok_idx, Kc_flat, L_tokens, Dc)
            Kc = Kc_flat.view(L_tokens, Dc)  # [L_tokens, Dc]

            gather_rows_kernel[grid_gather](kpe_cache_all, tok_idx, Kp_flat, L_tokens, Dp)
            Kp = Kp_flat.view(L_tokens, Dp)  # [L_tokens, Dp]

            # For each head i
            for i in range(H):
                # Load qn and qp as float32
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]

                # Compute logits_row = qn @ Kc.T + qp @ Kp.T for this head i
                logits_row = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                # First term: qn @ Kc.T
                matmul_row_kernel[(1,)](logits_row, qn, Kc, H=1, Dq=Dc, D=Dc, L=L_tokens)  # 1 program per head
                # Second term: qp @ Kp.T
                matmul_row_kernel_p[(1,)](logits_row, qp, Kp, H=1, Dp=Dp, L=L_tokens)
                # Scale
                logits_scaled = logits_row * sm_scale  # [L_tokens]

                # Compute lse per head (base-2): launch Triton kernel
                lse[b, i] = torch.empty((), dtype=torch.float32, device=device)  # dummy, overwritten by kernel
                lse_base2_row_kernel[(H,)](logits_scaled, lse[b], H=H, L=L_tokens)

                # Compute attn = softmax(logits_scaled) per head
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_row_kernel[(H,)](attn, logits_scaled, H=H, L=L_tokens)

                # Compute out_vec[i] = attn @ Kc
                out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
                matvec_kernel[(1,)](out_vec, attn, Kc, H=1, D=Dc, L=L_tokens)

                # Store output[b, i, :]
                output[b, i] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
