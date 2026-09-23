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
    head_dim: tl.constexpr,   # int
    L_tokens: tl.constexpr,   # int
):
    pid = tl.program_id(0)  # row id in [0, L_tokens)
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)
    src_linear = tok_id * head_dim + offs
    vals = tl.load(K_src_ptr + src_linear)
    tl.store(out_ptr + pid * head_dim + offs, vals)


# Triton kernel: softmax over a 1D vector (input length L)
@triton.jit
def softmax_kernel(
    inp_ptr,          # *f32, [L_tokens]
    out_ptr,          # *f32, [L_tokens]
    L: tl.constexpr,  # int
    scale: tl.constexpr,  # sm_scale (float)
):
    pid = tl.program_id(0)
    offs = tl.arange(0, L)
    x = tl.load(inp_ptr + offs)
    x = x * scale
    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    sum_e = tl.sum(e, axis=0)
    out = e / sum_e
    tl.store(out_ptr + offs, out)


# Triton kernel: matvec out[i] = sum_j attn[i] * K[j], i in [0, head_dim), j in [0, L_tokens)
@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32, [L_tokens]
    K_ptr,            # *f32, [L_tokens]
    out_ptr,          # *f32, [head_dim]
    head_dim: tl.constexpr,   # int
    L: tl.constexpr,          # int
):
    i = tl.program_id(0)  # row in [0, head_dim)
    acc = 0.0
    for j in range(0, L):
        a = tl.load(attn_ptr + j)
        k = tl.load(K_ptr + j)
        acc += a * k
    tl.store(out_ptr + i, acc)


# Triton kernel: per-element logsumexp of a 1D vector (input length L), writes (max, sum_exp)
@triton.jit
def lse_per_head_kernel(
    x_ptr,            # *f32, [L_tokens] logits_scaled
    out_max_ptr,      # *f32, [L_tokens] per-element max
    out_sum_ptr,      # *f32, [L_tokens] per-element sum_exp
    L: tl.constexpr,  # int
    scale: tl.constexpr,  # sm_scale (float)
):
    pid = tl.program_id(0)  # index along L
    offs = tl.arange(0, 1)  # scalar
    x = tl.load(x_ptr + pid)
    x = x * scale
    m = tl.max(x, axis=0)  # scalar max of vector
    y = x - m
    e = tl.exp(y)
    sum_e = tl.sum(e, axis=0)  # scalar sum
    tl.store(out_max_ptr + pid, m)
    tl.store(out_sum_ptr + pid, sum_e)


# Triton kernel: reduce per-batch per-head lse across num_heads, atomic add into per-batch scalar
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,         # *f32, [batch_size, num_qo_heads]
    out_ptr,          # *f32, [batch_size]
    num_heads: tl.constexpr,  # int
):
    b = tl.program_id(0)
    total = 0.0
    for h in range(0, num_heads):
        total += lse_ptrs[b * num_heads + h]
    tl.atomic_add(out_ptr + b, total)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Check shapes
    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    assert num_qo_heads == 16 and head_dim_ckv == 512 and head_dim_kpe == 64

    num_pages = ckv_cache.shape[0]
    assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
    assert kv_indptr.shape[0] == batch_size + 1
    assert kv_indices.shape[0] > 0

    device = q_nope.device
    assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
    assert ckv_cache.dtype == torch.bfloat16 and kpe_cache.dtype == torch.bfloat16

    # Output buffers
    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    # Per-batch reduction for lse
    lse_base2 = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    # Loop over batches
    for b in range(batch_size):
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L_tokens <= 0:
            output[b].zero_()
            continue

        # Gather token indices
        tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.int32).contiguous()  # [L_tokens]

        # Gather Kc and Kp rows
        Kc_tmp = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
        Kp_tmp = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

        # Flatten caches for easy indexing
        Kc_flat = ckv_cache.to(torch.float32).view(num_pages, head_dim_ckv)
        Kp_flat = kpe_cache.to(torch.float32).view(num_pages, head_dim_kpe)

        grid_g = (L_tokens,)
        gather_tokens_kernel[grid_g](
            K_src_ptr=Kc_flat, idx_ptr=tok_idx, out_ptr=Kc_tmp,
            head_dim=head_dim_ckv, L_tokens=L_tokens,
        )
        gather_tokens_kernel[grid_g](
            K_src_ptr=Kp_flat, idx_ptr=tok_idx, out_ptr=Kp_tmp,
            head_dim=head_dim_kpe, L_tokens=L_tokens,
        )

        # Process each head
        for h in range(num_qo_heads):
            qn = q_nope[b, h].to(torch.float32).contiguous()   # [head_dim_ckv]
            qp = q_pe[b, h].to(torch.float32).contiguous()    # [head_dim_kpe]

            # Forward attention: logits = qn @ Kc_tmp.T + qp @ Kp_tmp.T
            logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            # Use Triton matvec to compute logits: we need to compute sum_j (qn[j] * Kc_tmp[:, j])
            # Implement via a simple loop over j; Triton kernel matvec computes out[j] = sum_i attn[i] * K[j] but we need out[j] here.
            # To keep Triton-only, we implement logits as torch operations here (safe) since they are small.
            logits = torch.matmul(qn, Kc_tmp.transpose(0, 1)) + torch.matmul(qp, Kp_tmp.transpose(0, 1))

            # Scale and softmax in Triton
            logits_scaled = logits * sm_scale  # sm_scale is float
            attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            softmax_kernel[(L_tokens,)](
                inp_ptr=logits_scaled, out_ptr=attn, L=L_tokens, scale=sm_scale
            )

            # Matvec output = attn @ Kc_tmp -> [head_dim_ckv]
            out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            matvec_kernel[(head_dim_ckv,)](
                attn_ptr=attn, K_ptr=Kc_tmp, out_ptr=out_vec, head_dim=head_dim_ckv, L=L_tokens
            )
            output[b, h] = out_vec

            # Per-head lse: logsumexp of logits_scaled
            max_vals = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            sum_exps = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            lse_per_head_kernel[(L_tokens,)](
                x_ptr=logits_scaled, out_max_ptr=max_vals, out_sum_ptr=sum_exps, L=L_tokens, scale=sm_scale
            )
            # lse = max + log(sum_exp)
            lse_per_head[b, h] = max_vals + torch.log(sum_exps)

        # Reduce lse across heads for this batch, convert to base-2
        lse_reduce_kernel[(batch_size,)](
            lse_ptrs=lse_per_head, out_ptr=lse_base2, num_heads=num_qo_heads
        )

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)

    # Convert lse to base-2 (original divides by math.log(2.0))
    lse_base2 = lse_base2 / math.log(2.0)

    return output, lse_base2


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
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)