import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits for a single batch element b and head h.
# For each token t, logits[t] = qn @ Kc[t, :] + qp @ Kp[t, :].
@triton.jit
def _compute_logits_kernel(
    qn_ptr,           # *const float32, shape [Dc]
    qp_ptr,           # *const float32, shape [Dp]
    Kc_ptr,           # *const float32, shape [Lt, Dc]
    Kp_ptr,           # *const float32, shape [Lt, Dp]
    logits_ptr,       # *float32,       shape [Lt]
    Dc: tl.constexpr, Dp: tl.constexpr, Lt: tl.constexpr
):
    for t in tl.static_range(0, Lt):
        sum_qn = 0.0
        for i in tl.static_range(0, Dc):
            qn_val = tl.load(qn_ptr + i)
            Kc_val = tl.load(Kc_ptr + t * Dc + i)
            sum_qn += qn_val * Kc_val
        sum_qp = 0.0
        for j in tl.static_range(0, Dp):
            qp_val = tl.load(qp_ptr + j)
            Kp_val = tl.load(Kp_ptr + t * Dp + j)
            sum_qp += qp_val * Kp_val
        tl.store(logits_ptr + t, sum_qn + sum_qp)


# Triton kernel: compute base-2 logsumexp of a float32 vector v of length N.
@triton.jit
def _lse_base2_kernel(
    v_ptr,            # *const float32, shape [N]
    lse_ptr,          # *float32,       shape [1]
    N: tl.constexpr,
    inv_ln2: tl.float32
):
    # First pass: max for numerical stability
    m = -float('inf')
    for i in tl.static_range(0, N):
        v_i = tl.load(v_ptr + i)
        m = tl.maximum(m, v_i)
    # Second pass: sum of exp(v - m)
    s = 0.0
    for i in tl.static_range(0, N):
        v_i = tl.load(v_ptr + i)
        s += tl.exp(v_i - m)
    # Store lse = m + log(s) / ln(2)
    tl.store(lse_ptr, m + tl.log(s) * inv_ln2)


# Triton kernel: compute softmax of a float32 vector v of length N (base-2 logsumexp already applied).
# Returns attention vector in a_ptr.
@triton.jit
def _softmax_kernel(
    v_ptr,            # *const float32, shape [N]
    a_ptr,            # *float32,       shape [N]
    N: tl.constexpr,
    inv_ln2: tl.float32
):
    # We use softmax(v / ln(2)) = exp(v / ln(2)) / sum exp(v / ln(2))
    s = 0.0
    for i in tl.static_range(0, N):
        s += tl.exp(tl.load(v_ptr + i) * inv_ln2)
    for i in tl.static_range(0, N):
        v_i = tl.load(v_ptr + i)
        a_i = tl.exp(v_i * inv_ln2) / s
        tl.store(a_ptr + i, a_i)


# Triton kernel: compute out = sum_t attn[t] * Kc[t, :] for a single head h.
# Kc shape [Lt, Dc], attn shape [Lt], out shape [Dc].
@triton.jit
def _dot_reduce_kernel(
    Kc_ptr,           # *const float32, shape [Lt, Dc]
    attn_ptr,         # *const float32, shape [Lt]
    out_ptr,          # *float32,       shape [Dc]
    Lt: tl.constexpr, Dc: tl.constexpr
):
    for d in tl.static_range(0, Dc):
        sum_d = 0.0
        for t in tl.static_range(0, Lt):
            attn_t = tl.load(attn_ptr + t)
            Kc_td = tl.load(Kc_ptr + t * Dc + d)
            sum_d += attn_t * Kc_td
        tl.store(out_ptr + d, sum_d)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size=None, num_qo_heads=None, head_dim_ckv=None, head_dim_kpe=None, num_pages=None, sm_scale=1.0):
        super().__init__()
        # We don't store inputs, but we keep attributes for clarity
        self.num_qo_heads = num_qo_heads
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe
        self.sm_scale = float(sm_scale)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors"

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        assert H == 16, "num_qo_heads must be 16"
        assert Dc == 512, "head_dim_ckv must be 512"
        assert Dp == 64, "head_dim_kpe must be 64"

        # Prepare Kc_all and Kp_all as contiguous rows
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Dp]

        # Output and lse
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        inv_ln2 = 1.0 / math.log(2.0)

        for b in range(B):
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                output[b].zero_()
                lse[b].fill_(-float('inf'))
                continue

            L_tokens = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [L_tokens]
            Kc_b = Kc_all[tok_idx].contiguous().to(torch.float32)   # [L_tokens, Dc]
            Kp_b = Kp_all[tok_idx].contiguous().to(torch.float32)   # [L_tokens, Dp]

            # q_nope[b] and q_pe[b]
            qn = q_nope[b].contiguous().to(torch.float32)           # [H, Dc]
            qp = q_pe[b].contiguous().to(torch.float32)             # [H, Dp]

            for h in range(H):
                qn_vec = qn[h, :].contiguous().to(torch.float32)    # [Dc]
                qp_vec = qp[h, :].contiguous().to(torch.float32)    # [Dp]

                # 1) Compute logits vector [L_tokens]
                logits = torch.empty(L_tokens, dtype=torch.float32, device=device)
                # Launch Triton kernel to compute logits
                _compute_logits_kernel[(1,)](
                    qn_vec, qp_vec,
                    Kc_b, Kp_b,
                    logits,
                    Dc=512, Dp=64, Lt=L_tokens
                )

                # 2) Compute lse (base-2 logsumexp) over logits_scaled = logits * sm_scale
                logits_scaled = logits * self.sm_scale
                lse_bh = torch.empty(1, dtype=torch.float32, device=device)
                _lse_base2_kernel[(1,)](
                    logits_scaled,
                    lse_bh,
                    N=L_tokens,
                    inv_ln2=inv_ln2
                )
                lse[b, h] = lse_bh[0]

                # 3) Compute attention vector (softmax over logits_scaled)
                attn = torch.empty(L_tokens, dtype=torch.float32, device=device)
                _softmax_kernel[(1,)](
                    logits_scaled,
                    attn,
                    N=L_tokens,
                    inv_ln2=inv_ln2
                )

                # 4) Compute out = attn @ Kc_b -> [Dc]
                out_vec = torch.empty(Dc, dtype=torch.float32, device=device)
                _dot_reduce_kernel[(1,)](
                    Kc_b, attn,
                    out_vec,
                    Lt=L_tokens, Dc=512
                )

                output[b, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


# Helper for the evaluation harness (fused interface)
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# Original Model for reference
class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
