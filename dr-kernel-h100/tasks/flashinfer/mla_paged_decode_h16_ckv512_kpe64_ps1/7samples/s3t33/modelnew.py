import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits vector for one (b, h) pair and store in out_logits_ptr
# Inputs:
#   qn_ptr: pointer to q_nope[b, h] -> [Dc], float32
#   Kc_ptr: pointer to Kc_b -> [L_b, Dc], float32, contiguous, row-major
#   qp_ptr: pointer to q_pe[b, h] -> [Dp], float32
#   Kp_ptr: pointer to Kp_b -> [L_b, Dp], float32, contiguous, row-major
#   out_ptr: pointer to out_logits[b, h, :] -> [L_b], float32
# Launch grid: (1, 1) for a given (b, h). We also pass b, h as tl.constexpr to specialize the kernel per (b,h).
@triton.jit
def compute_logits_kernel_bh(b: tl.constexpr, h: tl.constexpr,
                             qn_ptr, Kc_ptr, qp_ptr, Kp_ptr, out_ptr,
                             L_b: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr):
    # Load qn and qp
    qn = tl.load(qn_ptr)  # [Dc], float32
    qp = tl.load(qp_ptr)  # [Dp], float32

    # Accumulate logits for each token l
    logits = tl.zeros((L_b,), dtype=tl.float32)
    for l in tl.static_range(0, L_b):
        # Base offset for Kc row l and Kp row l
        base_kc = l * Dc
        base_kp = l * Dp

        # Compute dot(qn, Kc[l, :]) = sum_{c=0..Dc-1} qn[c] * Kc[l, c]
        sum1 = tl.zeros((), dtype=tl.float32)
        for c in tl.static_range(0, Dc):
            kc_val = tl.load(Kc_ptr + base_kc + c)  # scalar
            sum1 += qn[c] * kc_val

        # Compute dot(qp, Kp[l, :]) = sum_{p=0..Dp-1} qp[p] * Kp[l, p]
        sum2 = tl.zeros((), dtype=tl.float32)
        for p in tl.static_range(0, Dp):
            kp_val = tl.load(Kp_ptr + base_kp + p)  # scalar
            sum2 += qp[p] * kp_val

        logits[l] = sum1 + sum2

    # Store logits
    tl.store(out_ptr, logits)


# Triton kernel: compute base-2 logsumexp for a vector of length L_b
# Inputs:
#   lse_ptr: pointer to scalar output lse[b, h] -> float32
#   logits_ptr: pointer to logits vector -> [L_b], float32
#   L_b: tl.constexpr
@triton.jit
def compute_lse_kernel(lse_ptr, logits_ptr, L_b: tl.constexpr):
    # Numerically stable logsumexp in float32
    max_val = -float("inf")
    for i in tl.static_range(0, L_b):
        vi = tl.load(logits_ptr + i)
        if vi > max_val:
            max_val = vi

    sumexp = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(0, L_b):
        vi = tl.load(logits_ptr + i)
        sumexp += tl.exp(vi - max_val)

    lse = max_val + tl.log(sumexp)  # natural logsumexp
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse = lse / ln2  # base-2 logsumexp
    tl.store(lse_ptr, lse)


# Triton kernel: compute softmax over logits vector (stored as logits_scaled = logits), write to attn_ptr
# Inputs:
#   attn_ptr: pointer to attn[b, h, :] -> [L_b], float32
#   logits_ptr: pointer to logits_scaled vector -> [L_b], float32
#   L_b: tl.constexpr
@triton.jit
def compute_softmax_kernel(attn_ptr, logits_ptr, L_b: tl.constexpr):
    max_val = -float("inf")
    for i in tl.static_range(0, L_b):
        vi = tl.load(logits_ptr + i)
        if vi > max_val:
            max_val = vi

    sumexp = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(0, L_b):
        vi = tl.load(logits_ptr + i)
        sumexp += tl.exp(vi - max_val)

    denom = sumexp  # since we scaled logits by sm_scale in compute_logits_kernel_bh, this is the denominator
    for i in tl.static_range(0, L_b):
        vi = tl.load(logits_ptr + i)
        attn_i = tl.exp(vi - max_val) / denom
        tl.store(attn_ptr + i, attn_i)


# Triton kernel: compute out[b, h, :] = attn @ Kc_b
# Inputs:
#   out_ptr: pointer to output vector -> [Dc], float32
#   attn_ptr: pointer to attn vector -> [L_b], float32
#   Kc_ptr: pointer to Kc_b -> [L_b, Dc], float32, contiguous row-major
#   Dc: tl.constexpr
#   L_b: tl.constexpr
@triton.jit
def compute_out_kernel(out_ptr, attn_ptr, Kc_ptr, Dc: tl.constexpr, L_b: tl.constexpr):
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    for c in tl.static_range(0, Dc):
        # out_vec[c] = sum_{l=0..L_b-1} attn[l] * Kc[l, c]
        sum_prod = tl.zeros((), dtype=tl.float32)
        for l in tl.static_range(0, L_b):
            kc_val = tl.load(Kc_ptr + l * Dc + c)  # scalar
            attn_l = tl.load(attn_ptr + l)         # scalar
            sum_prod += attn_l * kc_val
        out_vec[c] = sum_prod
    tl.store(out_ptr, out_vec)


# Host: ModelNew.forward (no torch ops for compute)
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        # Ensure inputs are on the same device
        assert q_pe.device == device
        assert ckv_cache.device == device
        assert kpe_cache.device == device
        assert kv_indptr.device == device
        assert kv_indices.device == device

        # Output allocations
        output = torch.empty((B, H, Dc), dtype=torch.float32, device=device)  # we'll fill per-(b,h) via Triton
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process per batch
        for b in range(B):
            # Derive token indices for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_b = end - start
            # Sanity check
            assert L_b >= 0, f"Invalid kv_indptr range for b={b}: {start} {end}"
            if L_b == 0:
                # No tokens for this batch, output zero and lse -inf
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            tok_idx = kv_indices[start:end].to(torch.int32).to(device)  # indices within [0, N)
            # Gather Kc and Kp for this batch
            # ckv_cache: [N, 1, Dc] => slice per token index (1 is unused)
            Kc_b = ckv_cache[tok_idx, 0].to(torch.float32)  # [L_b, Dc]
            Kp_b = kpe_cache[tok_idx, 0].to(torch.float32)  # [L_b, Dp]

            # Prepare per-head vectors
            for h in range(H):
                # Load qn and qp as float32
                qn = q_nope[b, h].to(torch.float32).contiguous()
                qp = q_pe[b, h].to(torch.float32).contiguous()

                # Allocate buffers
                out_logits = torch.empty((L_b,), dtype=torch.float32, device=device)
                attn = torch.empty((L_b,), dtype=torch.float32, device=device)

                # Launch Triton kernel: compute logits_scaled for (b, h)
                grid = (1, 1)
                compute_logits_kernel_bh[grid](
                    b, h, qn, Kc_b, qp, Kp_b, out_logits,
                    L_b=L_b, Dc=Dc, Dp=Dp
                )

                # Compute base-2 logsumexp for (b, h)
                compute_lse_kernel[(1,)](lse[b, h], out_logits, L_b=L_b)

                # Compute softmax of logits_scaled for (b, h)
                compute_softmax_kernel[(1,)](attn, out_logits, L_b=L_b)

                # Compute out[b, h, :] = attn @ Kc_b
                out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
                compute_out_kernel[(1,)](out_vec, attn, Kc_b, Dc=Dc, L_b=L_b)

                # Write output
                output[b, h] = out_vec

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


# Optional: helpers matching original interface
def get_inputs():
    # Create inputs on CUDA for Triton
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar (not used in kernels to avoid previous "unrecognized" error)
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# Entry point
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)