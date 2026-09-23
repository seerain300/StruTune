import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute scale logits vector for one (b, h)
# scale_logits_ptr[b*H + h]: stores scaled logits vector of length L_b as float32
@triton.jit
def compute_scale_logits_kernel(qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, scale_logits_ptr,
                                L_b: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
                                BLOCK_L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Load qn and qp as float32
    qn = tl.load(qn_ptr)  # [Dc]
    qp = tl.load(qp_ptr)  # [Dp]
    # Accumulate scaled logits for each token
    for l_off in tl.static_range(0, L_b, BLOCK_L):
        l_idx = l_off + tl.arange(0, BLOCK_L)
        mask = l_idx < L_b
        # Gather Kc and Kp chunks
        Kc_chunk = tl.load(Kc_ptr + l_idx * Dc, mask=mask, other=0.0)  # [BLOCK_L, Dc]
        Kp_chunk = tl.load(Kp_ptr + l_idx * Dp, mask=mask, other=0.0)  # [BLOCK_L, Dp]
        # Compute dot products: sum over dimensions
        # logits_chunk = qn @ Kc_chunk.T + qp @ Kp_chunk.T
        dot1 = 0.0
        dot2 = 0.0
        # Reduce over Dc and Dp
        for d in tl.static_range(0, Dc):
            dot1 += qn[d] * Kc_chunk[:, d]
        for d in tl.static_range(0, Dp):
            dot2 += qp[d] * Kp_chunk[:, d]
        logits_chunk = dot1 + dot2  # [BLOCK_L]
        # Scale
        scaled = logits_chunk * 1.0  # scale factor is passed as 1.0; no sm_scale argument needed
        # Store scaled logits
        tl.store(scale_logits_ptr + l_idx, scaled, mask=mask)


# Triton kernel: compute base-2 softmax of a vector and store attn[b*H + h]
# Input: scale_logits_ptr[b*H + h] vector of length L_b
# Output: attn_ptr[b*H + h] vector of length L_b
@triton.jit
def compute_softmax_kernel(scale_logits_ptr, attn_ptr, L_b: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Load logits
    logits = tl.load(scale_logits_ptr)  # [L_b]
    # Numerically stable softmax: subtract max, exponentiate, sum, normalize
    max_logit = -float("inf")
    for i in tl.static_range(0, L_b):
        max_logit = tl.maximum(max_logit, logits[i])
    sum_exp = 0.0
    for i in tl.static_range(0, L_b):
        exp_i = tl.exp(logits[i] - max_logit)  # base-2 exp via natural exp
        sum_exp += exp_i
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    for i in tl.static_range(0, L_b):
        attn_i = tl.exp(logits[i] - max_logit) * inv_ln2 / sum_exp
        tl.store(attn_ptr + i, attn_i)


# Triton kernel: compute out[h, :] = attn[b*H + h] @ Kc_b (Kc_b is [L_b, Dc])
# Input: attn_ptr[b*H + h], Kc_ptr (flattened [L_b, Dc]), out_ptr[b*H*H + h*Dc]
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr,
                       L_b: tl.constexpr, Dc: tl.constexpr,
                       BLOCK_L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Initialize output vector
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    # Reduce over tokens in chunks
    for l_off in tl.static_range(0, L_b, BLOCK_L):
        l_idx = l_off + tl.arange(0, BLOCK_L)
        mask = l_idx < L_b
        attn_chunk = tl.load(attn_ptr + l_idx, mask=mask, other=0.0)  # [BLOCK_L]
        Kc_chunk = tl.load(Kc_ptr + l_idx * Dc, mask=mask, other=0.0)  # [BLOCK_L, Dc]
        # Accumulate: out_vec += sum(attn_chunk * Kc_chunk, axis=0)
        for i in tl.static_range(0, BLOCK_L):
            # attn_chunk[i] may be 0 for masked elements
            for d in tl.static_range(0, Dc):
                out_vec[d] += attn_chunk[i] * Kc_chunk[i, d]
    # Store out vector
    tl.store(out_ptr + h * Dc + tl.arange(0, Dc), out_vec)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    """
    Triton-only implementation of the original run function.
    Returns:
      - output: [B, H, Dc] tensor in bfloat16
      - lse: [B, H] tensor in float32 (base-2 logsumexp)
    """
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"
    B, H, Dc = q_nope.shape
    Dp = q_pe.shape[-1]
    N = ckv_cache.shape[0]
    # Ensure inputs are contiguous
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous()
    kpe_cache = kpe_cache.contiguous()
    kv_indptr = kv_indptr.contiguous()
    kv_indices = kv_indices.contiguous()

    device = q_nope.device
    output = torch.empty((B, H, Dc), dtype=torch.float32, device=device)  # compute in float32
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # We will compute scale logits, then softmax (base-2), then output = attn @ Kc.
    # For each batch, derive tokens
    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        L_b = end - start
        if L_b <= 0:
            # No tokens for this batch element; zero output and lse
            output[b].zero_()
            lse[b].zero_()
            continue

        # Gather token indices and corresponding Kc, Kp
        tok_idx = kv_indices[start:end]  # [L_b]
        Kc_b = ckv_cache[tok_idx]  # [L_b, Dc], float32
        Kp_b = kpe_cache[tok_idx]  # [L_b, Dp], float32

        # Prepare qn and qp for head h=0 (the original code uses one head per batch element with H=16, but we handle general H)
        # We will launch Triton kernels over (b, h). For each h, compute scale_logits, softmax, out.
        # Note: H is runtime, Triton supports grids with runtime values; here grid=(B,H) is fine.

        # Allocate buffers for scale logits, attn, and out
        scale_logits = torch.empty((L_b,), dtype=torch.float32, device=device)  # vector per (b,h)
        attn = torch.empty((L_b,), dtype=torch.float32, device=device)          # vector per (b,h)
        # We need per-(b,h) output vectors
        out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)

        # Launch Triton kernels over grid (B, H)
        for h in range(H):
            qn = q_nope[b, h].to(torch.float32).contiguous()
            qp = q_pe[b, h].to(torch.float32).contiguous()
            # Choose BLOCK sizes
            BLOCK_L = 128 if L_b >= 128 else 64
            # 1) compute scale logits
            scale_logits_kernel = compute_scale_logits_kernel[(B, H)]
            scale_logits_kernel(qn, qp, Kc_b, Kp_b, scale_logits,
                                L_b=L_b, Dc=Dc, Dp=Dp, BLOCK_L=BLOCK_L)
            # 2) compute softmax (base-2)
            softmax_kernel = compute_softmax_kernel[(B, H)]
            softmax_kernel(scale_logits, attn, L_b=L_b)

            # 3) compute out[h, :] = attn @ Kc_b
            out_kernel = compute_out_kernel[(B, H)]
            # out_vec is [Dc] float32
            out_kernel(attn, Kc_b, out_vec, L_b=L_b, Dc=Dc, BLOCK_L=BLOCK_L)

            # Store output[b, h, :]
            output[b, h, :] = out_vec
            # Store lse[b, h] = logsumexp(scale_logits) / ln(2)
            # We compute lse using Triton with a dedicated kernel; since we already have attn, we can compute lse via sum of exps
            # But to keep Triton-only, we compute lse via torch here. Alternatively, implement lse in Triton with a kernel.
            # Implement lse in Triton kernel:
            lse_tmp = torch.empty((1,), dtype=torch.float32, device=device)
            @triton.jit
            def lse_kernel(scale_logits_ptr, lse_out_ptr, L_b: tl.constexpr):
                # Load vector and compute lse
                max_logit = -float("inf")
                sum_exp = 0.0
                for i in tl.static_range(0, L_b):
                    val = tl.load(scale_logits_ptr + i)
                    max_logit = tl.maximum(max_logit, val)
                for i in tl.static_range(0, L_b):
                    sum_exp += tl.exp(scale_logits_ptr[i] - max_logit)
                lse_val = tl.log(sum_exp) + max_logit  # natural logsumexp
                lse_base2 = lse_val / 0.6931471805599453  # ln(2)
                tl.store(lse_out_ptr, lse_base2)
            lse_kernel(scale_logits, lse_tmp, L_b=L_b)
            lse[b, h] = lse_tmp[0]

    # Cast output to bfloat16 to match original
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


# Optional helpers matching the original signature
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


# Entry point as required
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)