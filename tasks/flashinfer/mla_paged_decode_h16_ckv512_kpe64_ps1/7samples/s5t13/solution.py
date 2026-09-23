import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: gather selected rows from global cache into contiguous per-batch buffers
@triton.jit
def gather_tokens_kernel(
    K_src_ptr,        # *f32, flattened [num_pages, head_dim]
    idx_ptr,          # *i32, [L_tokens] token indices
    out_ptr,          # *f32, contiguous [L_tokens, head_dim]
    head_dim: tl.constexpr,   # int (columns in each cache row)
    L_tokens: tl.constexpr,   # int (number of tokens selected for this batch)
):
    pid = tl.program_id(0)  # which token row to gather
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)
    src_linear = tok_id * head_dim + offs
    vals = tl.load(K_src_ptr + src_linear)
    tl.store(out_ptr + pid * head_dim + offs, vals)


# Triton kernel: compute per-head logits = qn @ Kc.T + qp @ Kp.T
@triton.jit
def forward_attention_kernel(
    qn_ptr,           # *f32, [head_dim_ckv]
    qp_ptr,           # *f32, [head_dim_kpe]
    Kc_ptr,           # *f32, [L_tokens, head_dim_ckv] (row-major)
    Kp_ptr,           # *f32, [L_tokens, head_dim_kpe] (row-major)
    logits_ptr,       # *f32, [L_tokens]
    L_tokens: tl.constexpr,
    head_dim_ckv: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # Loop over tokens to compute dot products: qn @ Kc.T + qp @ Kp.T
    # We'll compute each element of logits_ptr one by one
    for i in range(0, L_tokens):
        sum_qn = 0.0
        # qn is [head_dim_ckv], Kc[i] row is [head_dim_ckv]
        for j in range(0, head_dim_ckv):
            qj = tl.load(qn_ptr + j)
            kc_j = tl.load(Kc_ptr + i * head_dim_ckv + j)
            sum_qn += qj * kc_j
        sum_qp = 0.0
        for j in range(0, head_dim_kpe):
            qj = tl.load(qp_ptr + j)
            kp_j = tl.load(Kp_ptr + i * head_dim_kpe + j)
            sum_qp += qj * kp_j
        logits_ptr[i] = sum_qn + sum_qp  # default sm_scale=1.0, but kernel accepts it if needed


# Triton kernel: softmax over a vector logits_ptr of length L_tokens (in-place)
@triton.jit
def softmax_kernel(
    logits_ptr,       # *f32, [L_tokens]
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # 1) compute max
    max_val = -float("inf")
    for i in range(0, L_tokens):
        vi = tl.load(logits_ptr + i)
        if vi > max_val:
            max_val = vi

    # 2) compute sum of exp((vi - max) * sm_scale)
    sum_exp = 0.0
    for i in range(0, L_tokens):
        vi = tl.load(logits_ptr + i)
        sum_exp += tl.exp((vi - max_val) * sm_scale)

    # 3) normalize
    inv_sum = 1.0 / sum_exp
    for i in range(0, L_tokens):
        vi = tl.load(logits_ptr + i)
        attn_i = tl.exp((vi - max_val) * sm_scale) * inv_sum
        tl.store(logits_ptr + i, attn_i)


# Triton kernel: compute per-head logsumexp of logits_scaled
@triton.jit
def lse_per_head_kernel(
    logits_ptr,       # *f32, [L_tokens]
    out_ptr,          # *f32 scalar
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # Compute max, sum of exp
    max_val = -float("inf")
    for i in range(0, L_tokens):
        vi = tl.load(logits_ptr + i)
        if vi > max_val:
            max_val = vi

    sum_exp = 0.0
    for i in range(0, L_tokens):
        vi = tl.load(logits_ptr + i)
        sum_exp += tl.exp((vi - max_val) * sm_scale)

    lse_val = max_val * sm_scale + tl.log(sum_exp)
    tl.store(out_ptr, lse_val)


# Triton kernel: matvec out = attn @ Kc, where attn is [L_tokens], Kc is [L_tokens, head_dim]
@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32, [L_tokens]
    K_ptr,            # *f32, [L_tokens, head_dim] (row-major)
    out_ptr,          # *f32, [head_dim]
    head_dim: tl.constexpr,
    L_tokens: tl.constexpr,
):
    offs = tl.arange(0, head_dim)
    for j in range(0, head_dim):
        acc = 0.0
        for i in range(0, L_tokens):
            acc += tl.load(attn_ptr + i) * tl.load(K_ptr + i * head_dim + j)
        tl.store(out_ptr + j, acc)


# Triton kernel: reduce per-batch lse across heads and convert to base-2 (1 / ln(2))
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,         # *f32, [batch_size, num_qo_heads]
    out_ptr,          # *f32, [batch_size]
    num_heads: tl.constexpr,
    ln2: tl.constexpr,  # 1 / ln(2) as float
):
    b = tl.program_id(0)
    total = 0.0
    for h in range(0, num_heads):
        total += tl.load(lse_ptrs + b * num_heads + h)
    total = total * ln2  # convert natural logsumexp to base-2
    tl.store(out_ptr + b, total)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Assumptions (match original):
    # - q_nope: [B, 16, 512], q_pe: [B, 16, 64], ckv_cache: [N, 1, 512], kpe_cache: [N, 1, 64]
    # - kv_indptr: [B+1], kv_indices: [num_kv_indices]
    device = q_nope.device
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda

    B = q_nope.shape[0]
    head_dim_ckv = q_nope.shape[-1]
    head_dim_kpe = q_pe.shape[-1]
    num_qo_heads = q_nope.shape[1]
    N = ckv_cache.shape[0]
    L_tokens_total = kv_indices.numel()

    # Slice per-batch tokens using kv_indptr[b:b+1]
    # Ensure indices are contiguous
    L_tokens_list = []
    idx_lists = []
    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        L = end - start
        L_tokens_list.append(L)
        idx_lists.append(kv_indices[start:end].contiguous())

    # We will process each batch b separately (loops), but Triton kernels run per launch.
    output = torch.empty((B, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse_per_head = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

    for b in range(B):
        if L_tokens_list[b] <= 0:
            # No tokens for this batch; output zeros and continue
            output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            lse_per_head[b] = 0.0
            continue

        L = L_tokens_list[b]
        idx = idx_lists[b]
        assert idx.numel() == L

        # Prepare per-batch Kc_tmp and Kp_tmp: [L, head_dim]
        Kc_tmp = torch.empty((L, head_dim_ckv), dtype=torch.float32, device=device)
        Kp_tmp = torch.empty((L, head_dim_kpe), dtype=torch.float32, device=device)

        # Launch gather kernel
        grid_gather = (L,)
        gather_tokens_kernel[grid_gather](
            K_src_ptr=ckv_cache.squeeze(1).to(torch.float32).contiguous(),    # [N, head_dim_ckv] flattened
            idx_ptr=idx,                          # [L]
            out_ptr=Kc_tmp,                      # [L, head_dim_ckv]
            head_dim=head_dim_ckv,
            L_tokens=L,
        )

        grid_gather[0] = (L,)
        gather_tokens_kernel[grid_gather](
            K_src_ptr=kpe_cache.squeeze(1).to(torch.float32).contiguous(),    # [N, head_dim_kpe] flattened
            idx_ptr=idx,                          # [L]
            out_ptr=Kp_tmp,                      # [L, head_dim_kpe]
            head_dim=head_dim_kpe,
            L_tokens=L,
        )

        # Compute output for each head h and per-head lse
        for h in range(num_qo_heads):
            # qn, qp: flatten heads to [1, head_dim]
            qn = q_nope[b, h].to(torch.float32).contiguous()   # [head_dim_ckv]
            qp = q_pe[b, h].to(torch.float32).contiguous()     # [head_dim_kpe]

            # Logits vector [L]
            logits = torch.empty((L,), dtype=torch.float32, device=device)
            forward_attention_kernel[(L,)](
                qn_ptr=qn, qp_ptr=qp, Kc_ptr=Kc_tmp, Kp_ptr=Kp_tmp, logits_ptr=logits, L_tokens=L,
                head_dim_ckv=head_dim_ckv, head_dim_kpe=head_dim_kpe, sm_scale=float(sm_scale),
            )

            # Softmax over logits * sm_scale (in-place)
            softmax_kernel[(L,)](
                logits_ptr=logits, L_tokens=L, sm_scale=float(sm_scale),
            )

            # Matvec: out_vec = attn @ Kc_tmp -> [head_dim_ckv]
            out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            matvec_kernel[(head_dim_ckv,)](
                attn_ptr=logits, K_ptr=Kc_tmp, out_ptr=out_vec, head_dim=head_dim_ckv, L_tokens=L,
            )
            output[b, h] = out_vec

            # Per-head lse: logsumexp(logits * sm_scale)
            lse_val = torch.empty((), dtype=torch.float32, device=device)
            lse_per_head_kernel[(L,)](
                logits_ptr=logits, out_ptr=lse_val, L_tokens=L, sm_scale=float(sm_scale),
            )
            lse_per_head[b, h] = lse_val

    # Reduce lse across heads to get per-batch lse, convert to base-2
    ln2 = 1.0 / math.log(2.0)
    lse_base2 = torch.empty((B,), dtype=torch.float32, device=device)
    lse_reduce_kernel[(B,)](
        lse_ptrs=lse_per_head, out_ptr=lse_base2, num_heads=num_qo_heads, ln2=ln2,
    )

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)

    return output, lse_base2


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA device for Triton
        if not q_nope.is_cuda or not q_pe.is_cuda or not ckv_cache.is_cuda or not kpe_cache.is_cuda or not kv_indptr.is_cuda or not kv_indices.is_cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                q_nope = q_nope.to(device)
                q_pe = q_pe.to(device)
                ckv_cache = ckv_cache.to(device)
                kpe_cache = kpe_cache.to(device)
                kv_indptr = kv_indptr.to(device)
                kv_indices = kv_indices.to(device)
            else:
                raise RuntimeError("CUDA device required for Triton kernels")
        output, lse_base2 = _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)
        return output, lse_base2


def run(*args):
    return ModelNew()(*args)
