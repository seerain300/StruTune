import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute per-(b,h) logits_scaled vector [L_b] = sum over tokens of (qn @ Kc.T + qp @ Kp.T) * sm_scale
@triton.jit
def compute_logits_kernel_bh(
    qn_ptr,        # *f32, shape [Dc]
    qp_ptr,        # *f32, shape [Dp]
    Kc_ptr,        # *f32, shape [L_b, Dc] (we pass gathered Kc for batch b)
    Kp_ptr,        # *f32, shape [L_b, Dp] (we pass gathered Kp for batch b)
    logits_ptr,    # *f32, shape [L_b]
    Dc: tl.constexpr,
    Dp: tl.constexpr,
    L: tl.constexpr,               # number of tokens for this batch
    BLOCK_D: tl.constexpr,         # typically 128 or 256
    scale_qn: tl.constexpr,        # sm_scale
    scale_qp: tl.constexpr         # sm_scale
):
    # Accumulator for logits vector
    logits = tl.zeros((L,), dtype=tl.float32)

    # Loop over tokens in chunks
    for l_off in tl.static_range(0, L, 1):  # simple per-token loop; Triton JIT supports this pattern
        # For each token l_off, accumulate qn @ Kc[l_off, :] + qp @ Kp[l_off, :]
        qn_dot = tl.zeros((), dtype=tl.float32)
        qk_dot = tl.zeros((), dtype=tl.float32)

        # Reduce over Dc for qn @ Kc[l_off, :]
        for d_off in tl.static_range(0, Dc, BLOCK_D):
            d_idx = d_off + tl.arange(0, BLOCK_D)
            mask_d = d_idx < Dc
            # load qn segment
            qn_seg = tl.load(qn_ptr + d_idx, mask=mask_d, other=0.0)
            # load Kc[l_off, d_idx]
            kc_ptr = Kc_ptr + l_off * Dc + d_idx
            kc_seg = tl.load(kc_ptr, mask=mask_d, other=0.0)
            qn_dot += tl.sum(qn_seg * kc_seg, axis=0)

        # Reduce over Dp for qp @ Kp[l_off, :]
        for dp_off in tl.static_range(0, Dp, BLOCK_D):
            dp_idx = dp_off + tl.arange(0, BLOCK_D)
            mask_dp = dp_idx < Dp
            qk_seg = tl.load(qp_ptr + dp_idx, mask=mask_dp, other=0.0)
            kp_ptr = Kp_ptr + l_off * Dp + dp_idx
            kp_seg = tl.load(kp_ptr, mask=mask_dp, other=0.0)
            qk_dot += tl.sum(qk_seg * kp_seg, axis=0)

        # Accumulate logits for this token
        logits[l_off] = qn_dot * scale_qn + qk_dot * scale_qp

    # Write out logits vector
    tl.store(logits_ptr + tl.arange(0, L), logits)


# Triton kernel: compute base-2 logsumexp for a vector of length L
@triton.jit
def compute_lse_base2_kernel(
    logits_ptr,   # *f32, shape [L]
    lse_ptr,      # *f32, shape [1] (we can write scalar here)
    L: tl.constexpr
):
    # Compute max for numerical stability
    max_val = -1e30
    for i in tl.static_range(0, L, 1):
        max_val = tl.maximum(max_val, tl.load(logits_ptr + i))

    # Compute sum of exp(logits - max)
    sum_exp = 0.0
    for i in tl.static_range(0, L, 1):
        val = tl.load(logits_ptr + i)
        sum_exp += tl.exp(val - max_val)

    lse = max_val + tl.log(sum_exp)  # natural logsumexp
    # Convert to base-2: logsumexp_base2 = lse / ln(2)
    lse_base2 = lse / 1.4426950408889634  # 1 / ln(2)
    tl.store(lse_ptr, lse_base2)


# Triton kernel: compute softmax for a vector of length L (in-place on input buffer)
@triton.jit
def softmax_kernel(
    logits_ptr,   # *f32, shape [L]
    attn_ptr,     # *f32, shape [L]
    L: tl.constexpr
):
    # Compute max for stability
    max_val = -1e30
    for i in tl.static_range(0, L, 1):
        max_val = tl.maximum(max_val, tl.load(logits_ptr + i))

    # Compute exp(logits - max)
    sum_exp = 0.0
    for i in tl.static_range(0, L, 1):
        val = tl.load(logits_ptr + i)
        exp_i = tl.exp(val - max_val)
        # store exp_i to attn_ptr[i] for later matmul
        tl.store(attn_ptr + i, exp_i)
        sum_exp += exp_i

    # Normalize to get softmax
    inv_sum = 1.0 / sum_exp
    for i in tl.static_range(0, L, 1):
        exp_i = tl.load(attn_ptr + i)
        attn_val = exp_i * inv_sum
        tl.store(attn_ptr + i, attn_val)


# Triton kernel: compute out = attn @ Kc_b for [L,] @ [L, Dc] -> [Dc]
# We assume Kc_b is a contiguous [L, Dc] buffer (gathered in host). We reduce over L in chunks.
@triton.jit
def compute_out_kernel(
    attn_ptr,      # *f32, shape [L]
    Kc_ptr,        # *f32, shape [L, Dc]
    out_ptr,       # *f32, shape [Dc]
    Dc: tl.constexpr,
    L: tl.constexpr,
    BLOCK_L: tl.constexpr
):
    # Accumulator for output vector
    out = tl.zeros((Dc,), dtype=tl.float32)
    # Reduce over tokens in chunks
    for l_off in tl.static_range(0, L, BLOCK_L):
        l_idx = l_off + tl.arange(0, BLOCK_L)
        mask_l = l_idx < L
        # Load attn segment
        attn_seg = tl.load(attn_ptr + l_idx, mask=mask_l, other=0.0)
        # Load corresponding Kc block [BLOCK_L, Dc]
        kc_block = tl.zeros((BLOCK_L, Dc), dtype=tl.float32)
        for d_off in tl.static_range(0, Dc, 1):  # simple per-d loop; Dc is constexpr
            kc_block[:, d_off] = tl.load(Kc_ptr + l_idx * Dc + d_off, mask=mask_l, other=0.0)
        # Matvec: out += sum over l of attn_seg[l] * kc_block[l, :]
        for i in tl.static_range(0, BLOCK_L, 1):
            # Handle masked l: attn_seg[l] can be 0 for l>=L
            contrib = attn_seg[i] * kc_block[i, :]
            out += contrib
    tl.store(out_ptr, out)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure tensors are on CUDA
    device = q_nope.device
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors"

    B = q_nope.shape[0]
    H = q_nope.shape[1]
    Dc = q_nope.shape[2]
    Dp = q_pe.shape[2]
    # Prepare output tensors
    output = torch.empty((B, H, Dc), dtype=torch.float32, device=device)  # will cast to bfloat16 at the end
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # Iterate over batch
    for b in range(B):
        # Compute token indices for this batch
        if kv_indptr.numel() <= b + 1:
            # Should not happen due to assertion, but guard
            continue
        L_b = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
        if L_b <= 0:
            # No tokens for this batch element
            lse[b] = 0.0
            output[b] = torch.zeros((H, Dc), dtype=torch.float32, device=device)
            continue

        # Gather Kc and Kp for this batch
        Kc_b = ckv_cache[kv_indptr[b]:kv_indptr[b] + L_b, 0, :].contiguous()  # [L_b, Dc]
        Kp_b = kpe_cache[kv_indptr[b]:kv_indptr[b] + L_b, 0, :].contiguous()  # [L_b, Dp]

        # For each head h
        for h in range(H):
            # Prepare pointers
            qn = q_nope[b, h].contiguous().to(torch.float32)
            qp = q_pe[b, h].contiguous().to(torch.float32)

            # Buffer for logits_scaled [L_b]
            logits_scaled = torch.empty((L_b,), dtype=torch.float32, device=device)

            # Kernel 1: compute logits_scaled per (b,h)
            # Launch grid: (1,1) since we compute per (b,h)
            compute_logits_kernel_bh[(1,)](
                qn, qp, Kc_b, Kp_b, logits_scaled,
                Dc, Dp, L_b,
                BLOCK_D=128,
                scale_qn=float(sm_scale),
                scale_qp=float(sm_scale)
            )

            # Kernel 2: compute base-2 lse per head
            lse_buf = torch.empty((1,), dtype=torch.float32, device=device)
            compute_lse_base2_kernel[(1,)](logits_scaled, lse_buf, L_b)
            lse[b, h] = lse_buf[0]

            # Kernel 3: compute softmax and store attn in a separate buffer
            attn = torch.empty((L_b,), dtype=torch.float32, device=device)
            softmax_kernel[(1,)](logits_scaled, attn, L_b)

            # Kernel 4: compute out = attn @ Kc_b -> [Dc]
            out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
            compute_out_kernel[(1,)](attn, Kc_b, out_vec, Dc, L_b, BLOCK_L=64)

            # Store output for this head
            output[b, h, :] = out_vec

    # Cast output to bfloat16 to match original signature expectations
    output = output.to(torch.bfloat16)
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


# Entry point required by evaluator
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
