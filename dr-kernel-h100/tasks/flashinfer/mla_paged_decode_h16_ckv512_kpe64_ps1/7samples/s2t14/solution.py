import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute output vector for a single head h for batch element b.
# Inputs:
#   qn_ptr: pointer to q_nope[b, h, :] of length Dc
#   qp_ptr: pointer to q_pe[b, h, :] of length Dp
#   Kc_ptr: pointer to Kc_all[tok_idx, :] of shape [L_tokens, Dc]
#   Kp_ptr: pointer to Kp_all[tok_idx, :] of shape [L_tokens, Dp]
#   out_ptr: pointer to output[b, h, :] of length Dc
#   lse_ptr: pointer to lse[b, h] (scalar)
# Arguments:
#   Dc: int (512)
#   Dp: int (64)
#   L_tokens: int (runtime)
#   sm_scale: float32
@triton.jit
def _head_kernel(
    qn_ptr, qp_ptr,
    Kc_ptr, Kp_ptr,
    out_ptr, lse_ptr,
    Dc: tl.constexpr, Dp: tl.constexpr, L_tokens: tl.constexpr,
    sm_scale: tl.float32
):
    # Accumulate logits per token
    # logits: [L_tokens] in fp32
    logits = tl.zeros((L_tokens,), dtype=tl.float32)

    # First pass: compute logits[t] = qn @ Kc[t, :] + qp @ Kp[t, :]
    for t in range(L_tokens):
        # qn @ Kc[t, :]
        acc_qn = 0.0
        for i in range(Dc):
            qni = tl.load(qn_ptr + i)
            Kc_val = tl.load(Kc_ptr + t * Dc + i)
            acc_qn += qni * Kc_val
        # qp @ Kp[t, :]
        acc_qp = 0.0
        for j in range(Dp):
            qpj = tl.load(qp_ptr + j)
            Kp_val = tl.load(Kp_ptr + t * Dp + j)
            acc_qp += qpj * Kp_val
        logits[t] = acc_qn + acc_qp

    # Compute base-2 logsumexp of logits_scaled = logits * sm_scale
    # logsumexp(x) = m + log(sum(exp(x - m))), where m = max(x)
    m = tl.max(logits)
    sum_exp = 0.0
    for t in range(L_tokens):
        sum_exp += tl.exp(logits[t] * sm_scale - m)
    lse_val = (m + tl.log(sum_exp)) / math.log(2.0)
    tl.store(lse_ptr, lse_val)

    # Second pass: compute attention and accumulate output
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    for t in range(L_tokens):
        attn = tl.exp(logits[t] * sm_scale - lse_val) / math.log(2.0)
        # Kc[t, :]
        for i in range(Dc):
            Kc_val = tl.load(Kc_ptr + t * Dc + i)
            out_vec[i] += attn * Kc_val

    # Store output as bfloat16
    for i in range(Dc):
        # out_ptr points to output[b, h, i]
        tl.store(out_ptr + i, out_vec[i].to(tl.bfloat16))


def _run_triton_only(q_nope, q_pe, Kc_all, Kp_all, kv_indptr, kv_indices, sm_scale):
    """
    Triton-only version: computes output and lse using Triton kernels.
    Returns (output [B, H, Dc] bfloat16, lse [B, H] float32).
    """
    assert q_nope.is_cuda and q_pe.is_cuda and Kc_all.is_cuda and Kp_all.is_cuda, "Triton requires CUDA tensors"
    B, H, Dc = q_nope.shape
    Dp = q_pe.shape[-1]
    device = q_nope.device

    output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # Process each batch element and head in Triton
    # We will launch one program per head h for batch b.
    # For b in range(B): compute tok_idx, gather Kc/Kp, then launch kernel for each h.
    for b in range(B):
        # Determine token range [page_beg, page_end)
        # Note: kv_indptr is int32; ensure long for indexing
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = max(page_end - page_beg, 0)

        # If empty, leave output zeros and lse as -inf
        if L_tokens == 0:
            # output[b] already initialized; lse[b] set via kernel to -inf, but we can skip if needed.
            lse[b].fill_(-float('inf'))
            continue

        # Gather token indices
        tok_idx = kv_indices[page_beg:page_end].to(torch.long).contiguous()

        # Gather Kc and Kp rows
        Kc = Kc_all[tok_idx].contiguous()  # [L_tokens, Dc]
        Kp = Kp_all[tok_idx].contiguous()  # [L_tokens, Dp]

        # Prepare qn and qp as contiguous float32
        qn = q_nope[b].to(torch.float32).contiguous()  # [Dc]
        qp = q_pe[b].to(torch.float32).contiguous()    # [Dp]

        # Launch Triton kernel for each head
        # We pass L_tokens as tl.constexpr; Triton will unroll.
        # Grid over heads: one program per head.
        for h in range(H):
            # Compute output and lse for this head
            # out_ptr points to output[b, h, :]
            out_ptr = output[b, h]  # 1D tensor of length Dc
            # lse_ptr points to lse[b, h]
            lse_ptr = lse[b, h]

            _head_kernel[(1,)](
                qn, qp,
                Kc, Kp,
                out_ptr, lse_ptr,
                Dc=512, Dp=64, L_tokens=L_tokens,
                sm_scale=sm_scale,
                num_warps=4, num_stages=2
            )

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA for Triton
        if not (q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda):
            raise RuntimeError("ModelNew.forward requires CUDA tensors for Triton kernels.")
        # Ensure types: q_nope, q_pe can be bfloat16; we will upcast for compute
        # squeeze caches
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)
        # Run Triton-only computation
        output, lse = _run_triton_only(q_nope, q_pe, Kc_all, Kp_all, kv_indptr, kv_indices, sm_scale)
        return output, lse


# Original helper functions (unchanged)
@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    # Checks
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
    assert kv_indptr.shape[0] == batch_size + 1
    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_beg >= page_end:
            # No KV cache for this batch element
            output[b].zero_()
            continue

        L_tokens = page_end - page_beg
        tok_idx = kv_indices[page_beg:page_end].to(torch.long)

        Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]
        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)   # [num_qo_heads, head_dim_kpe]

        for h in range(num_qo_heads):
            # Compute logits for this head: qn[h] @ Kc.T + qp[h] @ Kp.T
            logits = qn[h] @ Kc.T + qp[h] @ Kp.T  # [L_tokens]
            logits_scaled = logits * sm_scale
            lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)  # [L_tokens]
            out = attn @ Kc  # [head_dim_ckv]
            output[b, h] = out.to(torch.bfloat16)

    return output, lse

def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
