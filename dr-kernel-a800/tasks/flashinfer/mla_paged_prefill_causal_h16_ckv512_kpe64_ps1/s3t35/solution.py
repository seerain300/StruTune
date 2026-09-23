import torch
import math
import triton
import triton.language as tl


# Kernel 1: Compute a single-row (per head) logits_scaled of length L
# Inputs:
#   qn_ptr: *fp32, pointer to q_nope[q_abs] tensor of shape [H, K] (H=16, K=512)
#   qp_ptr: *fp32, pointer to q_pe[q_abs] tensor of shape [H, Kp] (Kp=64)
#   Kc_ptr: *fp32, base pointer to [P, K] cache
#   Kp_ptr: *fp32, base pointer to [P, Kp] cache
#   tok_idx_ptr: *int32, length L
#   logits_scaled_ptr: *fp32, output row of length L
# This kernel computes for each l:
#   logits_scaled[l] = sum_k qn[h, k] * Kc[tok_idx[l], k] + sum_kp qp[h, k'] * Kp[tok_idx[l], k']
# It assumes head index h is provided as constexpr and loops over h in host; kernels compute per-h for given h.

@triton.jit
def compute_logits_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, tok_idx_ptr, logits_scaled_ptr,
    H, K, Kp, L,
    stride_qn_h, stride_qn_k,
    stride_qp_h, stride_qp_kp,
    stride_log_l,
    head: tl.constexpr,
):
    for l in range(L):
        tok = tl.load(tok_idx_ptr + l)  # int32
        acc = 0.0
        # Accumulate over K features using q_nope[q_abs, h, :]
        for k in range(K):
            qn_off = head * stride_qn_h + k * stride_qn_k
            qn_val = tl.load(qn_ptr + qn_off)
            kc_off = tok * K + k
            kc_val = tl.load(Kc_ptr + kc_off)
            acc += qn_val * kc_val
        # Accumulate over Kp features using q_pe[q_abs, h, :]
        for kp in range(Kp):
            qp_off = head * stride_qp_h + kp * stride_qp_kp
            qp_val = tl.load(qp_ptr + qp_off)
            kp_off = tok * Kp + kp
            kp_val = tl.load(Kp_ptr + kp_off)
            acc += qp_val * kp_val
        tl.store(logits_scaled_ptr + l * stride_log_l, acc)


# Kernel 2: Compute lse = logsumexp(logits_scaled) / ln(2) for a single row
@triton.jit
def compute_lse_row_kernel(
    logits_scaled_ptr, lse_scalar_ptr,
    L,
    stride_log_l,
):
    m = -float('inf')
    for l in range(L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        if val > m:
            m = val
    sumexp = 0.0
    for l in range(L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        sumexp += tl.exp(val - m)
    lse = m + tl.log(sumexp) / 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_scalar_ptr, lse)


# Kernel 3: Compute softmax(logits_scaled) for a single row into attn_ptr
@triton.jit
def compute_softmax_row_kernel(
    logits_scaled_ptr, lse_scalar_ptr, attn_ptr,
    L,
    stride_log_l, stride_attn_l,
):
    lse = tl.load(lse_scalar_ptr)
    for l in range(L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        attn = tl.exp(val - lse)
        tl.store(attn_ptr + l * stride_attn_l, attn)


# Kernel 4: GEMV per head: out = attn @ Kc_all[tok_idx, :]
@triton.jit
def gemv_out_row_kernel(
    attn_ptr, Kc_ptr, tok_idx_ptr, out_ptr,
    L, K,
    stride_attn_l, stride_out_k,
):
    for k in range(K):
        acc = 0.0
        for l in range(L):
            attn_val = tl.load(attn_ptr + l * stride_attn_l)
            tok = tl.load(tok_idx_ptr + l)
            kc_val = tl.load(Kc_ptr + tok * K + k)
            acc += attn_val * kc_val
        tl.store(out_ptr + k * stride_out_k, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA."
        device = q_nope.device

        # Squeeze caches to [P, K] and [P, Kp]
        Kc_all = ckv_cache.squeeze(1).contiguous().float()  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().float()  # [P, 64]

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        head_dim_kpe = q_pe.shape[-1]
        assert head_dim_kpe == 64

        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        assert qo_indptr[-1].item() == total_q, "Sum of qo_indptr must equal total_q."

        # Allocate outputs
        output = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            # Compute tok_idx for this batch segment
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg
            if L <= 0:
                continue
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()  # [L]

            for i in range(q_len):
                q_abs = q_start + i

                # Cast query slices to float32 for kernel computation
                # Shapes: q_nope[q_abs] -> [16, 512], q_pe[q_abs] -> [16, 64]
                qn = q_nope[q_abs].to(torch.float32).contiguous()  # [16, 512]
                qp = q_pe[q_abs].to(torch.float32).contiguous()   # [16, 64]

                # Allocate temporaries
                logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)
                attn = torch.empty((L,), dtype=torch.float32, device=device)

                # 1) Compute logits_scaled for each head h
                for h in range(16):
                    # Launch compute_logits_row_kernel
                    compute_logits_row_kernel[(1,)](
                        qn, qp, Kc_all, Kp_all, tok_idx, logits_scaled,
                        H=16, K=512, Kp=64, L=L,
                        stride_qn_h=512, stride_qn_k=1,
                        stride_qp_h=64, stride_qp_kp=1,
                        stride_log_l=1,
                        head=h,
                    )
                    # 2) Compute lse for this head
                    lse_scalar = torch.empty((1,), dtype=torch.float32, device=device)
                    compute_lse_row_kernel[(1,)](
                        logits_scaled, lse_scalar,
                        L=L, stride_log_l=1,
                    )
                    lse[q_abs, h] = lse_scalar[0]

                    # 3) Compute attn = softmax(logits_scaled)
                    compute_softmax_row_kernel[(1,)](
                        logits_scaled, lse[q_abs, h], attn,
                        L=L, stride_log_l=1, stride_attn_l=1,
                    )

                    # 4) Compute output vector for this head via GEMV
                    out_vec = torch.empty((512,), dtype=torch.float32, device=device)
                    gemv_out_row_kernel[(1,)](
                        attn, Kc_all, tok_idx, out_vec,
                        L=L, K=512,
                        stride_attn_l=1, stride_out_k=1,
                    )

                    # Store into output (bf16)
                    output[q_abs, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


# Helpers to mirror original interface (optional)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
