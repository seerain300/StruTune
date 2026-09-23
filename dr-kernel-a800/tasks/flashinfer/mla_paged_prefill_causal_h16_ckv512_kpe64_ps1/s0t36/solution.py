import torch
import triton
import triton.language as tl


# Triton kernels: GEMV, reductions, elementwise ops

# 1) GEMV: qn_row @ Kc.T -> out[Dn]
#    qn_ptr: [Dn], Kc_ptr: [KV, Dn], out_ptr: [Dn]
@triton.jit
def gemv_qn_kc_kernel(qn_ptr, Kc_ptr, out_ptr,
                      Dn: tl.constexpr, KV: tl.constexpr):
    for d in range(0, Dn):
        acc = 0.0
        for j in range(0, KV):
            val = tl.load(Kc_ptr + j * Dn + d)
            qn_val = tl.load(qn_ptr + d)
            acc += qn_val * val
        tl.store(out_ptr + d, acc)


# 2) GEMV: qp_row @ Kp.T -> out[Dp]
@triton.jit
def gemv_qp_kp_kernel(qp_ptr, Kp_ptr, out_ptr,
                      Dp: tl.constexpr, KV: tl.constexpr):
    for d in range(0, Dp):
        acc = 0.0
        for j in range(0, KV):
            val = tl.load(Kp_ptr + j * Dp + d)
            qp_val = tl.load(qp_ptr + d)
            acc += qp_val * val
        tl.store(out_ptr + d, acc)


# 3) Elementwise add: logits_qn[Dn] + logits_qp[Dp] (only first Dp dims)
@triton.jit
def add_first_d_logits_kernel(logits_qn_ptr, logits_qp_ptr, logits_sum_ptr,
                              Dn: tl.constexpr, Dp: tl.constexpr):
    for d in range(0, Dp):
        a = tl.load(logits_qn_ptr + d)
        b = tl.load(logits_qp_ptr + d)
        tl.store(logits_sum_ptr + d, a + b)
    # For d >= Dp, logits_sum = logits_qn
    for d in range(Dp, Dn):
        a = tl.load(logits_qn_ptr + d)
        tl.store(logits_sum_ptr + d, a)


# 4) Scale logits by sm_scale
@triton.jit
def scale_logits_kernel(logits_ptr, scaled_ptr, KV: tl.constexpr, sm_scale: tl.float32):
    for j in range(0, KV):
        x = tl.load(logits_ptr + j)
        y = x * sm_scale
        tl.store(scaled_ptr + j, y)


# 5) Causal mask: apply mask over KV on scaled logits (set masked positions to -inf)
#    keep[j] = (j > (prefix_len + i))
@triton.jit
def apply_causal_mask_kernel(scaled_ptr, masked_ptr, KV: tl.constexpr, prefix_len: tl.int32, query_pos: tl.int32):
    for j in range(0, KV):
        x = tl.load(scaled_ptr + j)
        keep = (j > (prefix_len + query_pos))
        val = tl.where(keep, x, -float('inf'))
        tl.store(masked_ptr + j, val)


# 6) Reduce max over masked logits (vector of KV)
@triton.jit
def reduce_max_kernel(masked_ptr, max_ptr, KV: tl.constexpr):
    m = -float('inf')
    for j in range(0, KV):
        x = tl.load(masked_ptr + j)
        m = tl.maximum(m, x)
    tl.store(max_ptr, m)


# 7) Sum of exp(masked - max) over KV
@triton.jit
def sumexp_kernel(masked_ptr, max_ptr, sum_ptr, KV: tl.constexpr):
    m = tl.load(max_ptr)
    s = 0.0
    for j in range(0, KV):
        x = tl.load(masked_ptr + j)
        s += tl.exp(x - m)
    tl.store(sum_ptr, s)


# 8) Compute LSE = log(sum) * inv_ln2 + max
@triton.jit
def compute_lse_kernel(sum_ptr, max_ptr, lse_ptr, inv_ln2: tl.float32):
    s = tl.load(sum_ptr)
    m = tl.load(max_ptr)
    lse = tl.log(s) * inv_ln2 + m
    tl.store(lse_ptr, lse)


# 9) Masked softmax over KV: attn[j] = exp(masked[j] - m) / sum if j > (prefix_len + query_pos), else 0
@triton.jit
def masked_softmax_kernel(masked_ptr, attn_ptr, KV: tl.constexpr, prefix_len: tl.int32, query_pos: tl.int32):
    # Compute max
    m = -float('inf')
    for j in range(0, KV):
        x = tl.load(masked_ptr + j)
        m = tl.maximum(m, x)
    # Compute sum of exp
    sum_val = 0.0
    for j in range(0, KV):
        x = tl.load(masked_ptr + j)
        sum_val += tl.exp(x - m)
    # Write normalized attn
    for j in range(0, KV):
        x = tl.load(masked_ptr + j)
        keep = (j > (prefix_len + query_pos))
        val = tl.exp(x - m) / sum_val
        val = tl.where(keep, val, 0.0)
        tl.store(attn_ptr + j, val)


# 10) GEMV: out[h, :] = attn_row @ Kc -> [Dn]
@triton.jit
def gemv_attn_kc_kernel(attn_ptr, Kc_ptr, out_ptr,
                        Dn: tl.constexpr, KV: tl.constexpr):
    for d in range(0, Dn):
        acc = 0.0
        for j in range(0, KV):
            val = tl.load(Kc_ptr + j * Dn + d)
            attn_j = tl.load(attn_ptr + j)
            acc += attn_j * val
        tl.store(out_ptr + d, acc)


def _triton_only_forward(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    # Ensure CUDA
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
    device = q_nope.device

    # Shapes
    total_q, H, Dn = q_nope.shape  # [N, 16, 512]
    Dp = q_pe.shape[-1]            # 64
    # Batches
    B = qo_indptr.shape[0] - 1

    # Allocate outputs
    output = torch.empty((total_q, H, Dn), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

    inv_ln2 = 1.4426950408889634  # 1 / ln(2)

    for b in range(B):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        if q_start >= q_end:
            continue

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        if kv_start >= kv_end:
            continue

        kv_len = kv_end - kv_start
        tok_idx = kv_indices[kv_start:kv_end].to(torch.int32).contiguous()  # [KV]
        Kc = ckv_cache[tok_idx].to(torch.float32).contiguous()             # [KV, Dn]
        Kp = kpe_cache[tok_idx].to(torch.float32).contiguous()             # [KV, Dp]

        for i in range(q_end - q_start):
            i_q = q_start + i
            # prefix_len is number of previously processed tokens in this batch
            prefix_len = kv_len - (q_end - q_start)
            query_pos = i_q  # absolute position of query

            for h in range(H):
                # 1) qn_row [Dn], qp_row [Dp]
                qn_row = q_nope[i_q, h, :].to(torch.float32).contiguous()  # [Dn]
                qp_row = q_pe[i_q, h, :].to(torch.float32).contiguous()   # [Dp]

                # 2) logits_qn = qn_row @ Kc.T -> [Dn]
                logits_qn = torch.empty(Dn, dtype=torch.float32, device=device)
                gemv_qn_kc_kernel[(1,)](qn_row, Kc, logits_qn, Dn=Dn, KV=kv_len)

                # 3) logits_qp = qp_row @ Kp.T -> [Dp]
                logits_qp = torch.empty(Dp, dtype=torch.float32, device=device)
                gemv_qp_kp_kernel[(1,)](qp_row, Kp, logits_qp, Dp=Dp, KV=kv_len)

                # 4) Add only first Dp dims
                logits_sum = torch.empty(Dn, dtype=torch.float32, device=device)
                add_first_d_logits_kernel[(1,)](logits_qn, logits_qp, logits_sum, Dn=Dn, Dp=Dp)

                # 5) Scale
                scaled = torch.empty(Dn, dtype=torch.float32, device=device)
                scale_logits_kernel[(kv_len,)](logits_sum, scaled, KV=kv_len, sm_scale=sm_scale)

                # 6) Apply causal mask
                masked = torch.empty(Dn, dtype=torch.float32, device=device)
                apply_causal_mask_kernel[(kv_len,)](scaled, masked, KV=kv_len, prefix_len=prefix_len, query_pos=query_pos)

                # 7) Reduce max
                max_val = torch.empty(1, dtype=torch.float32, device=device)
                reduce_max_kernel[(1,)](masked, max_val, KV=kv_len)

                # 8) Sum exp
                sum_exp = torch.empty(1, dtype=torch.float32, device=device)
                sumexp_kernel[(1,)](masked, max_val, sum_exp, KV=kv_len)

                # 9) Compute LSE
                lse_row = torch.empty(1, dtype=torch.float32, device=device)
                compute_lse_kernel[(1,)](sum_exp, max_val, lse_row, inv_ln2)

                # 10) Masked softmax
                attn = torch.empty(Dn, dtype=torch.float32, device=device)
                masked_softmax_kernel[(kv_len,)](masked, attn, KV=kv_len, prefix_len=prefix_len, query_pos=query_pos)

                # 11) Output row: attn @ Kc -> [Dn]
                out_row = torch.empty(Dn, dtype=torch.float32, device=device)
                gemv_attn_kc_kernel[(1,)](attn, Kc, out_row, Dn=Dn, KV=kv_len)

                # Store output and lse
                output[i_q, h, :] = out_row.to(torch.bfloat16)
                lse[i_q, h] = lse_row[0]

    return output, lse


# For completeness, mirror get_inputs and fused_operator from the original snippet.
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = _triton_only_forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return fused_operator(*args)


def run(*args):
    return ModelNew()(*args)
