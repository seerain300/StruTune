import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute output and lse for a single (batch b, head h).
@triton.jit
def _batch_elem_kernel(
    q_nope_ptr, q_pe_ptr, Kc_all_ptr, Kp_all_ptr,
    out_ptr, lse_ptr,
    # strides
    out_stride_b, out_stride_h, out_stride_d,
    q_nope_stride_b, q_nope_stride_h, q_nope_stride_d,
    q_pe_stride_b, q_pe_stride_h, q_pe_stride_d,
    Kc_stride_t, Kc_stride_d,
    Kp_stride_t, Kp_stride_d,
    # sizes
    Dc: tl.constexpr, Dp: tl.constexpr, L_tokens: tl.constexpr,
    sm_scale: tl.float32
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load qn and qp for head h
    # q_nope[b, h, :] -> pointer + offset
    qn_ptr = q_nope_ptr + b * q_nope_stride_b + h * q_nope_stride_h
    qn = tl.load(qn_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)

    # q_pe[b, h, :] -> pointer + offset
    qp_ptr = q_pe_ptr + b * q_pe_stride_b + h * q_pe_stride_h
    qp = tl.load(qp_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

    # Initialize logsumexp accumulators
    max_val = -float("inf")
    sum_exp = 0.0  # fp32 scalar

    # First pass: compute max and sum_exp for logsumexp over tokens
    for t in range(0, L_tokens):
        # Kc[t, :] and Kp[t, :]
        Kc_t = tl.load(Kc_all_ptr + t * Kc_stride_t + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        Kp_t = tl.load(Kp_all_ptr + t * Kp_stride_t + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

        # Dot products
        sum_qn_Kc = tl.sum(qn * Kc_t, axis=0)
        sum_qp_Kp = tl.sum(qp * Kp_t, axis=0)
        logits_t = sum_qn_Kc + sum_qp_Kp

        # Scaled logits
        logits_scaled_t = logits_t * sm_scale

        # Update max and sum_exp
        max_val = tl.maximum(max_val, logits_scaled_t)
        sum_exp += tl.exp(logits_scaled_t - max_val)

    # Compute lse = max_val + log(sum_exp) / ln(2)
    lse_val = (max_val + tl.log(sum_exp)) / 1.4426950408889634  # 1 / ln(2)

    # Second pass: compute attn and accumulate output vector
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    for t in range(0, L_tokens):
        Kc_t = tl.load(Kc_all_ptr + t * Kc_stride_t + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        Kp_t = tl.load(Kp_all_ptr + t * Kp_stride_t + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

        sum_qn_Kc = tl.sum(qn * Kc_t, axis=0)
        sum_qp_Kp = tl.sum(qp * Kp_t, axis=0)
        logits_t = sum_qn_Kc + sum_qp_Kp
        attn_t = tl.exp((logits_t * sm_scale) - lse_val) / 1.4426950408889634  # base-2 softmax scaling

        # out[h, :] += attn[t] * Kc[t, :]
        out_vec += attn_t * Kc_t

    # Store output[b, h, :] as bfloat16
    out_ptr_b = out_ptr + b * out_stride_b + h * out_stride_h
    out_store = out_vec.to(tl.bfloat16)
    # Store vector using a loop for indices
    for i in range(0, Dc):
        tl.store(out_ptr_b + i * out_stride_d, out_store[i])

    # Store lse[b, h] as float32
    tl.store(lse_ptr + b * lse_stride_b + h * lse_stride_h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure on GPU
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA device"

        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Constants (assert in host if needed)
        # We won't rely on Triton constants here; pass sizes as ints.

        # Prepare output and lse
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(B):
            # Compute L_tokens from kv_indptr
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # output[b].zero_(); lse[b] = -inf
                output[b].zero_()
                lse[b] = float("-inf")
                continue

            # Gather token indices
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.long)

            # Gather Kc and Kp rows
            Kc_all = ckv_cache[tok_idx].to(torch.float32)  # [L_tokens, Dc]
            Kp_all = kpe_cache[tok_idx].to(torch.float32)  # [L_tokens, Dp]

            # Prepare strides
            out_stride_b, out_stride_h, out_stride_d = output.stride()
            q_nope_stride_b, q_nope_stride_h, q_nope_stride_d = q_nope[b].stride()  # we will load via ptr arithmetic, so use shape to form pointers
            q_pe_stride_b, q_pe_stride_h, q_pe_stride_d = q_pe[b].stride()

            Kc_stride_t, Kc_stride_d = Kc_all.stride()
            Kp_stride_t, Kp_stride_d = Kp_all.stride()

            # Launch Triton kernel for each head
            for h in range(H):
                # Launch kernel: one program per (b, h)
                _batch_elem_kernel[(1, 1)](
                    q_nope[b], q_pe[b], Kc_all, Kp_all,
                    output, lse,
                    out_stride_b, out_stride_h, out_stride_d,
                    q_nope_stride_b, q_nope_stride_h, q_nope_stride_d,
                    q_pe_stride_b, q_pe_stride_h, q_pe_stride_d,
                    Kc_stride_t, Kc_stride_d,
                    Kp_stride_t, Kp_stride_d,
                    Dc=Dc, Dp=Dp, L_tokens=L_tokens,
                    sm_scale=sm_scale,
                    num_warps=4, num_stages=2
                )

        return output, lse

# Original Model for reference
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    # No strict assert on num_pages; just use as-is

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = page_end - page_beg
        if L_tokens <= 0:
            output[b].zero_()
            continue

        tok_idx = kv_indices[page_beg:page_end].to(torch.long)
        Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]
        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)    # [num_qo_heads, head_dim_kpe]

        logits = qn @ Kc.T + qp @ Kp.T    # [num_qo_heads, L_tokens]
        logits_scaled = logits * sm_scale

        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, L_tokens]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[b] = out.to(torch.bfloat16)

    return output, lse

def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
