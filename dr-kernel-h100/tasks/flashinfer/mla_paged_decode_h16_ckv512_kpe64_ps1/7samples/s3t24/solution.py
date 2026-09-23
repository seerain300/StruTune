import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits_scaled per (b, h) into a buffer of length L_b
# Inputs:
#   qn_ptr: pointer to q_nope[b, h] -> [Dc], float32
#   qp_ptr: pointer to q_pe[b, h] -> [Dp], float32
#   Kc_ptr: pointer to ckv_cache[tok_idx] -> [L_b, Dc], float32
#   Kp_ptr: pointer to kpe_cache[tok_idx] -> [L_b, Dp], float32
#   out_ptr: pointer to output logits_scaled buffer [L_b], float32
# Meta:
#   L: number of tokens in this batch (L_b), tl.constexpr
#   Dc: 512, tl.constexpr
#   Dp: 64, tl.constexpr
#   BLOCK_D: chunk size for reduction over Dc/Dp, tl.constexpr
@triton.jit
def compute_logits_kernel_bh(qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
                             L: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
                             BLOCK_D: tl.constexpr):
    # Load qn[h, :] and qp[h, :]
    qn = tl.load(qn_ptr)  # [Dc], float32
    qp = tl.load(qp_ptr)  # [Dp], float32

    # Accumulate logits for each token l
    for l in tl.static_range(0, L):
        dot_qn = 0.0
        dot_qp = 0.0
        # Reduce over Dc in chunks
        for d_off in tl.static_range(0, Dc, BLOCK_D):
            d = d_off + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            mask = d < Dc
            kc = tl.load(Kc_ptr + l * Dc + d, mask=mask, other=0.0)  # [BLOCK_D]
            dot_qn += tl.sum(qn[d] * kc)
        # Reduce over Dp in chunks
        for p_off in tl.static_range(0, Dp, BLOCK_D):
            p = p_off + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            mask = p < Dp
            kp = tl.load(Kp_ptr + l * Dp + p, mask=mask, other=0.0)  # [BLOCK_D]
            dot_qp += tl.sum(qp[p] * kp)
        logits = dot_qn + dot_qp  # scalar float32
        tl.store(out_ptr + l, logits)


# Triton kernel: compute base-2 logsumexp of a float32 vector x (length L), write to out[0]
# Assumes x_ptr points to a contiguous vector of length L, all float32.
@triton.jit
def compute_lse_kernel(x_ptr, out_ptr, L: tl.constexpr, inv_ln2: tl.float32, BLOCK_L: tl.constexpr):
    # One program instance: reduce over tokens in chunks
    m = -float("inf")
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)  # [BLOCK_L]
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        # Reduce to max across the chunk
        chunk_max = -float("inf")
        for i in tl.static_range(0, BLOCK_L):
            vi = x[i]
            chunk_max = tl.maximum(chunk_max, vi)
        m = tl.maximum(m, chunk_max)

    # Compute sum(exp(x - m)) across all tokens
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

    # Pass 2: compute exp(x - m) / sum
    sumexp = 0.0
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        sumexp += tl.sum(tl.exp(x - m))

    # Write normalized softmax
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        expv = tl.exp(x - m) / sumexp
        # Store only valid positions
        # Create a tensor for output with mask
        out_chunk = tl.zeros((BLOCK_L,), dtype=tl.float32)
        for i in tl.static_range(0, BLOCK_L):
            if (idx[i] < L):
                out_chunk[i] = expv[i]
        tl.store(out_ptr + idx, out_chunk, mask=(idx < L))


# Triton kernel: compute out[h, :] = attn[h, :] @ Kc[:, :] where attn is [L], Kc is [L, Dc]
# We implement reduction over L in chunks.
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr, L: tl.constexpr, Dc: tl.constexpr, BLOCK_L: tl.constexpr):
    # Initialize output vector
    acc = tl.zeros((Dc,), dtype=tl.float32)
    for l_off in tl.static_range(0, L, BLOCK_L):
        l = l_off + tl.arange(0, BLOCK_L)  # [BLOCK_L]
        mask = l < L
        attn = tl.load(attn_ptr + l, mask=mask, other=0.0)  # [BLOCK_L]
        for d_off in tl.static_range(0, Dc, BLOCK_L):
            d = d_off + tl.arange(0, BLOCK_L)  # [BLOCK_L]
            mask_d = d < Dc
            K = tl.load(Kc_ptr + d, mask=mask_d, other=0.0)  # [BLOCK_L]
            # acc[d] += attn[l] * Kc[l, d] reduced over l
            acc += tl.sum(attn[:, None] * K[None, :], axis=0)
    # Store acc to out_ptr
    for d in tl.static_range(0, Dc):
        tl.store(out_ptr + d, acc[d])


@triton.jit
def compute_logsumexp_scaled_kernel(x_ptr, out_ptr, L: tl.constexpr, scale: tl.float32, inv_ln2: tl.float32, BLOCK_L: tl.constexpr):
    # Pass 1: max of scale * x
    m = -float("inf")
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        scaled = x * scale
        chunk_max = -float("inf")
        for i in tl.static_range(0, BLOCK_L):
            vi = scaled[i]
            chunk_max = tl.maximum(chunk_max, vi)
        m = tl.maximum(m, chunk_max)

    # Pass 2: sum exp(x - m / scale)
    sumexp = 0.0
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        scaled = x * scale
        sumexp += tl.sum(tl.exp(scaled - m))

    lse = m + tl.log(sumexp) * inv_ln2
    tl.store(out_ptr, lse)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Constants
    B, H, Dc = q_nope.shape
    _, _, Dp = q_pe.shape
    assert Dc == 512 and Dp == 64, "head dimensions must be 512 and 64 respectively"
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA"

    # Process per batch
    device = q_nope.device
    L_b_list = []
    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        L_b = end - start
        if L_b <= 0:
            # No tokens for this batch element: output zeros, lse = -inf
            output_b = torch.zeros((H, Dc), dtype=torch.bfloat16, device=device)
            lse_b = torch.full((H,), -float("inf"), dtype=torch.float32, device=device)
            for h in range(H):
                output_b[h] = torch.zeros((Dc,), dtype=torch.bfloat16, device=device)
            if b == 0:
                output = output_b
                lse = lse_b
            else:
                output = torch.cat([output, output_b], dim=0)
                lse = torch.cat([lse, lse_b], dim=0)
            continue
        L_b_list.append(L_b)
        tok_idx = kv_indices[start:end].to(torch.long).to(device)  # [L_b]
        # Gather Kc and Kp
        Kc_b = ckv_cache[tok_idx].to(torch.float32).contiguous()  # [L_b, Dc]
        Kp_b = kpe_cache[tok_idx].to(torch.float32).contiguous()  # [L_b, Dp]

        # Output and lse buffers
        output_b = torch.empty((H, Dc), dtype=torch.float32, device=device)
        lse_b = torch.empty((H,), dtype=torch.float32, device=device)

        # Precompute inv_ln2
        inv_ln2 = 1.0 / math.log(2.0)

        for h in range(H):
            # Compute qn and qp (float32), vectors
            qn = q_nope[b, h].to(torch.float32).contiguous()  # [Dc]
            qp = q_pe[b, h].to(torch.float32).contiguous()   # [Dp]

            # Allocate logits buffer [L_b] in float32
            logits_scaled = torch.empty((L_b,), dtype=torch.float32, device=device)

            # Triton: compute logits_scaled per (b, h)
            compute_logits_kernel_bh[(1,)](qn, qp, Kc_b, Kp_b, logits_scaled, L_b, Dc, Dp, BLOCK_D=128)

            # Triton: compute base-2 logsumexp per head
            compute_lse_kernel[(1,)](logits_scaled, lse_b[h], L_b, inv_ln2, BLOCK_L=1024)

            # Triton: compute softmax of scaled logits into attn
            attn = torch.empty((L_b,), dtype=torch.float32, device=device)
            compute_softmax_kernel[(1,)](logits_scaled, attn, L_b, BLOCK_L=1024)

            # Triton: compute out[h, :] = attn @ Kc_b
            out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
            compute_out_kernel[(1,)](attn, Kc_b, out_vec, L_b, Dc, BLOCK_L=128)

            output_b[h] = out_vec  # store as float32, cast later

        if b == 0:
            output = output_b
            lse = lse_b
        else:
            output = torch.cat([output, output_b], dim=0)
            lse = torch.cat([lse, lse_b], dim=0)

    # Cast output to bfloat16 as per original
    output = output.to(torch.bfloat16)
    return output, lse


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
