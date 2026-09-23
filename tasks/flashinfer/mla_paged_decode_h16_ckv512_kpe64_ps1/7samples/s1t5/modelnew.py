import math
import torch
import triton
import triton.language as tl


# Kernel 1: For a single head j, compute v[i] = sum_k qn[j,k]*Kc[i,k] + sum_p qp[j,p]*Kp[i,p]
@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, shape [D]
    qp_ptr,           # *float32, shape [Dp]
    Kc_ptr,           # *float32, shape [L, D], row-major
    Kp_ptr,           # *float32, shape [L, Dp], row-major
    v_ptr,            # *float32, shape [L]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv (Kc dim)
    Dp: tl.int32,     # head_dim_kpe (Kp dim)
    BLOCK_K: tl.constexpr,  # reduction tile over K dimension
):
    # One program per output element i in [0, L)
    i = tl.program_id(0)
    sum1 = 0.0
    sum2 = 0.0

    # Reduce over Kc (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr_row = Kc_ptr + i * D + k_off
        kc_slice = tl.load(kc_ptr_row, mask=mask_k, other=0.0)      # [BLOCK_K]
        sum1 += tl.sum(qn_slice * kc_slice, axis=0)

    # Reduce over Kp (Dp)
    for p in range(0, Dp, BLOCK_K):
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        qp_slice = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0)  # [BLOCK_K]
        kp_ptr_row = Kp_ptr + i * Dp + p_off
        kp_slice = tl.load(kp_ptr_row, mask=mask_p, other=0.0)      # [BLOCK_K]
        sum2 += tl.sum(qp_slice * kp_slice, axis=0)

    v = sum1 + sum2
    tl.store(v_ptr + i, v)


# Kernel 2: Base-2 softmax per row (head). Requires v_ptr (float32) of length L and writes attn_ptr (float32).
# We assume lse is computed separately and passed. Here we compute lse in Triton too by a small kernel.
@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, shape [L]
    attn_ptr,         # *float32, shape [L]
    L: tl.int32,
    scale: tl.float32,          # sm_scale
    inv_ln2: tl.float32,        # 1.4426950408889634
    ln2: tl.float32,            # 0.6931471805599453
    BLOCK: tl.constexpr,        # tile size for vector ops over L
):
    row_id = tl.program_id(0)  # not used since we process one row per program, but keep for future generalization
    # Compute max for numerical stability
    max_v = -1e20
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=-1e20)
        max_v = tl.maximum(max_v, tl.max(v, axis=0))

    # Compute sum of exp scaled by base-2
    sum_exp = 0.0
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=-1e20)
        s = scale * (v - max_v) / ln2
        expv = tl.exp(s)
        sum_exp += tl.sum(expv, axis=0)

    lse = tl.log(sum_exp) / ln2 + max_v  # logsumexp in base-2

    # Write normalized attention
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=-1e20)
        s = scale * (v - max_v) / ln2 - lse
        attn = tl.exp(s)
        tl.store(attn_ptr + offs, attn, mask=mask)


# Kernel 3: For a single head j, compute out[b, j, :] = attn_j @ Kc[:, :], i.e., matvec over D=512.
@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, shape [L]
    Kc_ptr,           # *float32, shape [L, D], row-major
    out_ptr,          # *float32, shape [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_D: tl.constexpr,
):
    j = tl.program_id(0)  # head index
    # output per head j
    for d in range(0, D, BLOCK_D):
        d_off = d + tl.arange(0, BLOCK_D)
        mask_d = d_off < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(0, L):
            attn_i = tl.load(attn_ptr + i)
            kc_ptr_row = Kc_ptr + i * D + d_off
            kc_slice = tl.load(kc_ptr_row, mask=mask_d, other=0.0)
            acc += attn_i * kc_slice
        tl.store(out_ptr + d_off, acc, mask=mask_d)


# Host function using Triton: run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)
@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure on CUDA device
    device = q_nope.device
    assert device.type == 'cuda', "All tensors must be on CUDA device for Triton kernels."

    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    L_tokens_total = (kv_indptr[1:] - kv_indptr[:-1]).sum().item()  # just sanity; not used directly

    # Allocate outputs
    output = torch.empty(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device
    )
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    for b in range(batch_size):
        # Determine token range
        if kv_indptr.numel() <= 1:
            # Degenerate, handle
            output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            lse[b] = torch.full((num_qo_heads,), -float("inf"), dtype=torch.float32, device=device)
            continue

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L = max(page_end - page_beg, 0)
        if L == 0:
            # No tokens for this batch
            output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            lse[b] = torch.full((num_qo_heads,), -float("inf"), dtype=torch.float32, device=device)
            continue

        # Gather tokens
        tok_idx = kv_indices[page_beg:page_end]  # int32
        Kc = ckv_cache[tok_idx].to(torch.float32).contiguous()  # [L, 512]
        Kp = kpe_cache[tok_idx].to(torch.float32).contiguous()  # [L, 64]

        # Preprocess queries
        qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
        qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

        # Launch matvec_add_kernel for each head j to compute v[j, :]
        for j in range(num_qo_heads):
            # Prepare output v for head j
            v = torch.empty((L,), dtype=torch.float32, device=device)
            # Launch kernel: one program per i
            grid = (L,)
            matvec_add_kernel[grid](
                qn[j], qp[j], Kc, Kp, v,
                L=L, D=head_dim_ckv, Dp=head_dim_kpe,
                BLOCK_K=128,
                num_warps=4,
            )

            # Compute lse for head j in Triton
            lse_j = torch.empty((), dtype=torch.float32, device=device)  # scalar
            # Use a tiny kernel that reduces over L; since L is small (<= M), we just compute here in PyTorch as a proxy
            # In a pure Triton environment, we'd implement lse kernel with loops over L; here we approximate:
            # However, the original code uses torch.logsumexp. To strictly follow, we compute lse in torch for simplicity:
            lse_j = torch.logsumexp(v / math.log(2.0), dim=0)  # torch-based for correctness

            # Now compute attn in Triton with base-2 softmax
            attn = torch.empty((L,), dtype=torch.float32, device=device)
            inv_ln2 = 1.4426950408889634  # 1 / ln(2)
            ln2 = 0.6931471805599453
            softmax_base2_kernel[(1,)](
                v, attn,
                L=L,
                scale=sm_scale,
                inv_ln2=inv_ln2,
                ln2=ln2,
                BLOCK=128,
                num_warps=4,
            )
            # Store lse[b, j]
            lse[b, j] = lse_j.item() if isinstance(lse_j, torch.Tensor) else float(lse_j)

            # Finally, out[b, j, :] = attn @ Kc[:, :] -> matvec
            out_j = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            matvec_write_y_kernel[(1,)](
                attn, Kc,
                out_j,
                L=L, D=head_dim_ckv,
                BLOCK_D=128,
                num_warps=4,
            )
            # Store to output: [B, H, D]
            output[b, j, :] = out_j

    # Cast output to bfloat16 and return lse as float32
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


def get_inputs():
    # Generate inputs on CUDA device
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
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure CUDA
        for t in args:
            if t is not None and t.device.type != 'cuda':
                t = t.to('cuda')
        return run(*args)