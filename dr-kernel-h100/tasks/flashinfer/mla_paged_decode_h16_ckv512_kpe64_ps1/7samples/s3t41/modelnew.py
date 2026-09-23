import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits_scaled vector for one (b, h) pair.
# Inputs:
#   qn_ptr: pointer to q_nope[b, h] -> shape [Dc], dtype float32
#   qp_ptr: pointer to q_pe[b, h] -> shape [Dp], dtype float32
#   Kc_ptr: pointer to Kc gathered for this batch -> shape [L_b, Dc], dtype float32
#   Kp_ptr: pointer to Kp gathered for this batch -> shape [L_b, Dp], dtype float32
#   scale_ptr: pointer to output logits_scaled vector [L_b], dtype float32
# Launch grid: (1,), using b and h as scalars inside the kernel
@triton.jit
def compute_logits_kernel(qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, scale_ptr,
                          L_b: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr, sm_scale: tl.float32,
                          BLOCK: tl.constexpr):
    # Each program handles one (b, h). We decode program_id(0) into b and h by passing them as launch parameters,
    # but since grid is (1,), we keep it simple: load qn, qp, and compute logits for all L_b.
    qn = tl.load(qn_ptr)  # [Dc]
    qp = tl.load(qp_ptr)  # [Dp]

    logits = tl.zeros((L_b,), dtype=tl.float32)
    # Accumulate over tokens
    for l in tl.static_range(0, L_b):
        # Load Kc[l, :] and Kp[l, :]
        # Note: Kc_ptr is laid out as [L_b, Dc], so offset = l * Dc + d
        acc1 = tl.zeros((), dtype=tl.float32)
        for d in tl.static_range(0, Dc, BLOCK):
            offs_d = d + tl.arange(0, BLOCK)
            mask_d = offs_d < Dc
            kc = tl.load(Kc_ptr + l * Dc + offs_d, mask=mask_d, other=0.0)
            acc1 += tl.sum(qn[offs_d] * kc, axis=0)
        acc2 = tl.zeros((), dtype=tl.float32)
        for d in tl.static_range(0, Dp, BLOCK):
            offs_d = d + tl.arange(0, BLOCK)
            mask_d = offs_d < Dp
            kp = tl.load(Kp_ptr + l * Dp + offs_d, mask=mask_d, other=0.0)
            acc2 += tl.sum(qp[offs_d] * kp, axis=0)
        logits[l] = (acc1 + acc2) * sm_scale

    tl.store(scale_ptr + tl.arange(0, L_b), logits)  # store vector


# Triton kernel: compute base-2 logsumexp for a vector of length L_b (one per program)
@triton.jit
def compute_lse_kernel(x_ptr, lse_ptr, L_b: tl.constexpr, BLOCK: tl.constexpr):
    # We have one program per (b, h); grid can be (1,). Compute stable lse
    max_val = tl.full((), -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for l in tl.static_range(0, L_b, BLOCK):
        offs = l + tl.arange(0, BLOCK)
        mask = offs < L_b
        vals = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)
    # Second pass: sum exp
    for l in tl.static_range(0, L_b, BLOCK):
        offs = l + tl.arange(0, BLOCK)
        mask = offs < L_b
        vals = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)
    lse_val = tl.log(sum_exp) + max_val  # ln sumexp at max-shift
    # Convert to base-2 logsumexp: log2(sumexp) = ln(sumexp)/ln(2)
    lse_val_base2 = lse_val / 0.6931471805599453  # 1/ln(2)
    tl.store(lse_ptr, lse_val_base2)


# Triton kernel: compute softmax of a vector (stable) and write attn
@triton.jit
def compute_softmax_kernel(x_ptr, attn_ptr, L_b: tl.constexpr, BLOCK: tl.constexpr):
    max_val = tl.full((), -float('inf'), dtype=tl.float32)
    # First pass: max
    for l in tl.static_range(0, L_b, BLOCK):
        offs = l + tl.arange(0, BLOCK)
        mask = offs < L_b
        vals = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)
    # Second pass: sum of exp
    sum_exp = tl.zeros((), dtype=tl.float32)
    for l in tl.static_range(0, L_b, BLOCK):
        offs = l + tl.arange(0, BLOCK)
        mask = offs < L_b
        vals = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)
    # Third pass: write normalized
    for l in tl.static_range(0, L_b, BLOCK):
        offs = l + tl.arange(0, BLOCK)
        mask = offs < L_b
        vals = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
        attn = tl.exp(vals - max_val) / sum_exp
        tl.store(attn_ptr + offs, attn, mask=mask)


# Triton kernel: compute out[h, :] = attn @ Kc_b using chunked reduction over L_b
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr,
                       L_b: tl.constexpr, Dc: tl.constexpr, BLOCK_L: tl.constexpr, sm_scale: tl.float32):
    # This kernel assumes Dc is small enough; we vectorize over Dc with a static loop
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    for d in tl.static_range(0, Dc, BLOCK_L):
        offs_d = d + tl.arange(0, BLOCK_L)
        mask_d = offs_d < Dc
        acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
        for l in tl.static_range(0, L_b, BLOCK_L):
            offs_l = l + tl.arange(0, BLOCK_L)
            mask_l = offs_l < L_b
            attn_chunk = tl.load(attn_ptr + offs_l, mask=mask_l, other=0.0)  # [BLOCK_L]
            kc_chunk = tl.load(Kc_ptr + offs_l[:, None] * Dc + offs_d[None, :], mask=mask_l[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_L, BLOCK_L]
            acc += tl.sum(attn_chunk[:, None] * kc_chunk, axis=0)
        out_vec[offs_d] = acc
    # Store out_vec
    tl.store(out_ptr + tl.arange(0, Dc), out_vec)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    B = q_nope.shape[0]
    H = q_nope.shape[1]
    Dc = q_nope.shape[2]
    Dp = q_pe.shape[2]

    # Output tensors
    output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    for b in range(B):
        # Gather token indices for this batch
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        L_b = end - start
        tok_idx = kv_indices[start:end]  # [L_b]
        # Gather Kc and Kp for this batch
        Kc_b = ckv_cache[tok_idx, 0].to(torch.float32)  # [L_b, Dc]
        Kp_b = kpe_cache[tok_idx, 0].to(torch.float32)  # [L_b, Dp]
        # Prepare pointers for Triton
        # Flatten qn, qp to 1-element tensors for simple loading
        qn = q_nope[b].to(torch.float32)  # [Dc]
        qp = q_pe[b].to(torch.float32)   # [Dp]
        # Buffer for logits_scaled
        logits_scaled = torch.empty((L_b,), dtype=torch.float32, device=device)

        # Kernel 1: compute logits_scaled
        # Launch grid: (1,)
        compute_logits_kernel[(1,)](
            qn, qp, Kc_b, Kp_b, logits_scaled,
            L_b=L_b, Dc=Dc, Dp=Dp, sm_scale=float(sm_scale),
            BLOCK=128,
        )
        # Kernel 2: compute base-2 logsumexp
        compute_lse_kernel[(1,)](
            logits_scaled, lse[b],
            L_b=L_b, BLOCK=256,
        )
        # Kernel 3: compute softmax (stable)
        attn = torch.empty((L_b,), dtype=torch.float32, device=device)
        compute_softmax_kernel[(1,)](
            logits_scaled, attn,
            L_b=L_b, BLOCK=256,
        )
        # Kernel 4: compute out[h, :] = attn @ Kc_b
        out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
        compute_out_kernel[(1,)](
            attn, Kc_b, out_vec,
            L_b=L_b, Dc=Dc, BLOCK_L=128, sm_scale=float(sm_scale),
        )
        output[b] = out_vec.to(torch.bfloat16)

    return output, lse


# Optional helpers matching the original
def get_inputs():
    # Ensure tensors are on CUDA
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# Optional: wrapper using ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)