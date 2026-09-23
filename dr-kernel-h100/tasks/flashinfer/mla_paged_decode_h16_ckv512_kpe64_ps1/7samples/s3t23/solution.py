import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits per (b, h) into a buffer [L] (float32)
# Args:
#   qn_ptr: pointer to q_nope[b, h] -> [Dc], float32
#   Kc_ptr: pointer to ckv_cache[token_indices] -> [L, Dc], float32
#   Kp_ptr: pointer to kpe_cache[token_indices] -> [L, Dp], float32
#   qp_ptr: pointer to q_pe[b, h] -> [Dp], float32
#   out_ptr: pointer to output logits_scaled buffer [L], float32
# Meta:
#   L: number of tokens in this batch, tl.constexpr
#   Dc: 512, tl.constexpr
#   Dp: 64, tl.constexpr
#   BLOCK_D: chunk size for reduction over Dc/Dp, tl.constexpr
@triton.jit
def compute_logits_kernel_bh(qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
                             L: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
                             BLOCK_D: tl.constexpr):
    # Load qn[h, :] and qp[h, :]
    qn = tl.load(qn_ptr)  # [Dc]
    qp = tl.load(qp_ptr)  # [Dp]

    # Accumulate logits for each token l
    for l in tl.static_range(0, L):
        acc = tl.zeros((), dtype=tl.float32)
        # Reduce over Dc in chunks
        for d_off in tl.static_range(0, Dc, BLOCK_D):
            d = d_off + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            mask = d < Dc
            kc = tl.load(Kc_ptr + l * Dc + d, mask=mask, other=0.0)  # [BLOCK_D]
            acc += tl.sum(qn[d] * kc)
        # Reduce over Dp in chunks for Kp
        for d_off in tl.static_range(0, Dp, BLOCK_D):
            d = d_off + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            mask = d < Dp
            kp = tl.load(Kp_ptr + l * Dp + d, mask=mask, other=0.0)  # [BLOCK_D]
            acc += tl.sum(qp[d] * kp)
        tl.store(out_ptr + l, acc * sm_scale)  # sm_scale is a global scalar, defined in host


# Triton kernel: base-2 logsumexp over a float32 vector x of length L
# Assumes x_ptr points to a contiguous vector of length L, all float32.
@triton.jit
def compute_lse_kernel(x_ptr, out_ptr, L: tl.constexpr, inv_ln2: tl.float32, BLOCK_L: tl.constexpr):
    # One program instance: two-pass reduction
    m = -float("inf")
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)  # [BLOCK_L]
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        # Reduce to max across the chunk
        chunk_max = -float("inf")
        for i in tl.static_range(0, BLOCK_L):
            vi = x[i]
            # masked positions are -inf
            chunk_max = tl.maximum(chunk_max, vi)
        m = tl.maximum(m, chunk_max)

    sumexp = 0.0
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        sumexp += tl.sum(tl.exp(x - m))

    lse = m + tl.log(sumexp) * inv_ln2
    tl.store(out_ptr, lse)


# Triton kernel: compute softmax of a float32 vector x (length L), write to out[0..L-1]
@triton.jit
def compute_softmax_kernel(x_ptr, out_ptr, L: tl.constexpr, BLOCK_L: tl.constexpr):
    # Pass 1: max for numerical stability
    m = -float("inf")
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        chunk_max = -float("inf")
        for i in tl.static_range(0, BLOCK_L):
            vi = x[i]
            chunk_max = tl.maximum(chunk_max, vi)
        m = tl.maximum(m, chunk_max)

    # Pass 2: compute softmax and store
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        expv = tl.exp(x - m)
        sumexp = tl.sum(expv)
        attn = expv / sumexp
        # store only valid positions
        for i in tl.static_range(0, BLOCK_L):
            if (l_off + i) < L:
                tl.store(out_ptr + (l_off + i), attn[i])


# Triton kernel: compute out vector for one (b, h): out[h, :] = attn @ Kc[:, :]
# attn_ptr: pointer to attn vector [L], float32
# Kc_ptr: pointer to Kc matrix [L, Dc], float32
# out_ptr: pointer to output vector [Dc], float32
# Meta:
#   L: number of tokens, tl.constexpr
#   Dc: 512, tl.constexpr
#   BLOCK_D: chunk size for reduction over Dc, tl.constexpr
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr, L: tl.constexpr, Dc: tl.constexpr, BLOCK_D: tl.constexpr):
    out = tl.zeros((Dc,), dtype=tl.float32)
    for d_off in tl.static_range(0, Dc, BLOCK_D):
        d = d_off + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        mask_d = d < Dc
        # accumulate over tokens in chunks
        for l_off in tl.static_range(0, L, 1):
            # scalar attn[l]
            attn_l = tl.load(attn_ptr + l_off)
            kc = tl.load(Kc_ptr + l_off * Dc + d, mask=mask_d, other=0.0)  # [BLOCK_D]
            out[d] += attn_l * kc
    tl.store(out_ptr, out)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    B, H, Dc = q_nope.shape
    Dp = q_pe.shape[-1]
    N = ckv_cache.shape[0]
    L_tot = kv_indices.numel()

    # Allocate outputs
    output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # Precompute inv_ln2 for base-2 logsumexp
    inv_ln2 = 1.0 / math.log(2.0)

    # Loop over batches
    for b in range(B):
        # Derive token indices for this batch: tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        L_b = end - start
        if L_b <= 0:
            lse[b, :] = torch.full((H,), -float("inf"), dtype=torch.float32, device=device)
            # output rows zero
            output[b] = torch.zeros((H, Dc), dtype=torch.bfloat16, device=device)
            continue

        tok_idx = kv_indices[start:end].to(torch.long).to(device)

        # Gather Kc and Kp for this batch
        Kc_b = ckv_cache[tok_idx].to(torch.float32)  # [L_b, Dc]
        Kp_b = kpe_cache[tok_indices[tok_idx]].to(torch.float32)  # [L_b, Dp]
        # Sanity: kpe_cache indexing should be kpe_cache[tok_idx], not with tok_indices again
        Kp_b = kpe_cache[tok_idx].to(torch.float32)  # [L_b, Dp]

        # Preallocate buffers for Triton
        logits_scaled = torch.empty((L_b,), dtype=torch.float32, device=device)

        # Compute logits_scaled for all heads via Triton kernel: we launch per (b, h)
        for h in range(H):
            qn = q_nope[b, h].to(torch.float32).contiguous()  # [Dc]
            qp = q_pe[b, h].to(torch.float32).contiguous()   # [Dp]

            # Launch Triton kernel to compute logits_scaled[l] for this (b, h)
            compute_logits_kernel_bh[(1,)](qn, qp, Kc_b, Kp_b, logits_scaled,
                                           L_b, Dc, Dp, BLOCK_D=128,
                                           sm_scale=sm_scale)

            # lse[h] = logsumexp(logits_scaled, base=2)
            lse_scalar = torch.empty((), dtype=torch.float32, device=device)
            compute_lse_kernel[(1,)](logits_scaled, lse_scalar, L_b, inv_ln2, BLOCK_L=1024)
            lse[b, h] = lse_scalar

            # Softmax of scaled logits: attn[l]
            attn = torch.empty((L_b,), dtype=torch.float32, device=device)
            compute_softmax_kernel[(1,)](logits_scaled, attn, L_b, BLOCK_L=1024)

            # Compute out[h, :] = attn @ Kc_b using Triton reduction kernel over L_b
            out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
            compute_out_kernel[(1,)](attn, Kc_b, out_vec, L_b, Dc, BLOCK_D=128)

            # Store to output
            output[b, h] = out_vec.to(torch.bfloat16)

    return output, lse


# Optional helpers to match the original signature
def get_inputs():
    # Place inputs on CUDA for Triton
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


# Entry point model class
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
