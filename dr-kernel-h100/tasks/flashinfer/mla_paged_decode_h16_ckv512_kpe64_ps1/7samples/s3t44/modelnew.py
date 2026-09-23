import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute single scaled logits for token l and (b, h).
# Launch grid: (b * H, L_b). Each program handles one token l for a specific (b,h).
@triton.jit
def compute_logits_single_kernel(qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, scale_ptr,
                                 L_b: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
                                 sm_scale: tl.float32,
                                 BLOCK_D: tl.constexpr, BLOCK_P: tl.constexpr):
    pid_bh = tl.program_id(0)
    l = tl.program_id(1)

    # Load qn and qp vectors for this (b,h)
    # qn_ptr and qp_ptr point to [Dc] and [Dp] respectively
    qn = tl.zeros((Dc,), dtype=tl.float32)
    qp = tl.zeros((Dp,), dtype=tl.float32)
    # Here qn_ptr and qp_ptr are pointers to base vectors; they are not indexed by l.
    # In the host code, we pass separate pointers for each (b,h) vector to this kernel.
    # The kernel expects qn_ptr and qp_ptr to be pointers to the current head's vectors.
    # To simplify, we directly load qn and qp via pointer arithmetic using the vector length.
    # Since qn_ptr and qp_ptr are [Dc] and [Dp], we just load them as flat vectors.
    # Triton does not support indexing by arbitrary strides; we rely on host to pass correct pointers.
    # Load qn and qp vectors
    qn = tl.load(qn_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0)
    qp = tl.load(qp_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)

    # For this token l, compute dot(qn, Kc[l]) and dot(qp, Kp[l])
    dot_qn = 0.0
    dot_qp = 0.0
    # Loop over Dc in chunks
    for d_off in tl.static_range(0, Dc, BLOCK_D):
        d_idx = d_off + tl.arange(0, BLOCK_D)
        mask_d = d_idx < Dc
        # Kc_ptr is a flat pointer to [L_b, Dc]; to get Kc[l, d:d+BLOCK_D], we compute addresses:
        # offset = l * Dc + d_idx
        kc_vals = tl.load(Kc_ptr + l * Dc + d_idx, mask=mask_d, other=0.0)
        dot_qn += tl.sum(qn[d_idx] * kc_vals, axis=0)

    for p_off in tl.static_range(0, Dp, BLOCK_P):
        p_idx = p_off + tl.arange(0, BLOCK_P)
        mask_p = p_idx < Dp
        kp_vals = tl.load(Kp_ptr + l * Dp + p_idx, mask=mask_p, other=0.0)
        dot_qp += tl.sum(qp[p_idx] * kp_vals, axis=0)

    logits = dot_qn + dot_qp
    scale = logits * sm_scale
    tl.store(scale_ptr + l, scale)


# Triton kernel: compute base-2 logsumexp for (b, h) over L_b.
# Launch grid: (B*H,)
@triton.jit
def compute_lse_kernel(logits_ptr, lse_ptr,
                       L_b: tl.constexpr, sm_scale: tl.float32):
    pid = tl.program_id(0)
    # Reductions using static loop over L_b
    max_val = -float('inf')
    for l in tl.static_range(0, L_b):
        val = tl.load(logits_ptr + l)
        max_val = tl.maximum(max_val, val)

    sumexp = 0.0
    for l in tl.static_range(0, L_b):
        val = tl.load(logits_ptr + l)
        sumexp += tl.exp(val - max_val)

    lse = tl.log(sumexp) + max_val  # logsumexp in natural log
    # Convert to base-2: divide by ln(2)
    lse = lse / 0.6931471805599453
    tl.store(lse_ptr + pid, lse)


# Triton kernel: compute softmax of scaled logits for (b, h) over L_b, store in attn_ptr.
# Launch grid: (B*H,)
@triton.jit
def compute_softmax_kernel(logits_ptr, attn_ptr,
                           L_b: tl.constexpr, sm_scale: tl.float32):
    pid = tl.program_id(0)
    max_val = -float('inf')
    for l in tl.static_range(0, L_b):
        val = tl.load(logits_ptr + l)  # already scaled
        max_val = tl.maximum(max_val, val)

    sumexp = 0.0
    for l in tl.static_range(0, L_b):
        val = tl.load(logits_ptr + l)
        sumexp += tl.exp(val - max_val)

    for l in tl.static_range(0, L_b):
        val = tl.load(logits_ptr + l)
        attn = tl.exp(val - max_val) / sumexp
        tl.store(attn_ptr + l, attn)


# Triton kernel: compute out[h, :] = attn @ Kc_b, output_vec_ptr[Dc].
# Launch grid: (B*H,)
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, output_vec_ptr,
                       L_b: tl.constexpr, Dc: tl.constexpr,
                       BLOCK_L: tl.constexpr):
    pid = tl.program_id(0)
    out = tl.zeros((Dc,), dtype=tl.float32)
    # Reduce over tokens in chunks
    for l_off in tl.static_range(0, L_b, BLOCK_L):
        l_idx = l_off + tl.arange(0, BLOCK_L)
        mask_l = l_idx < L_b
        attn_chunk = tl.load(attn_ptr + l_idx, mask=mask_l, other=0.0)
        # For each chunk, accumulate out += sum(attn_chunk[l] * Kc[l, :]) over l in the chunk
        for i in tl.static_range(0, BLOCK_L):
            li = l_off + i
            if li < L_b:
                alpha = attn_chunk[i]
                kc_vals = tl.load(Kc_ptr + li * Dc + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0)
                out += alpha * kc_vals
    tl.store(output_vec_ptr, out)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device

    B = q_nope.shape[0]
    H = q_nope.shape[1]
    Dc = q_nope.shape[2]
    Dp = q_pe.shape[2]

    # Prepare outputs
    output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # Loop over batch
    for b in range(B):
        # Determine L_b
        L_b = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())

        # Gather token indices for this batch
        tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.long).to(device)

        # Gather Kc and Kp for this batch
        # Note: caches are [N, 1, D] -> index 0
        Kc_b = ckv_cache[tok_idx, 0].to(torch.float32)  # [L_b, Dc]
        Kp_b = kpe_cache[tok_idx, 0].to(torch.float32)  # [L_b, Dp]

        # Prepare buffers
        logits_scaled = torch.empty((L_b,), dtype=torch.float32, device=device)
        attn = torch.empty((L_b,), dtype=torch.float32, device=device)

        # Compute qn and qp per head: we need to loop over H, but Triton grid is (B*H,) inside kernels.
        # We'll compute per head by launching with pid_bh = b*H + h.
        # First, compute logits_scaled for all l
        # We need to pass qn and qp vectors for each head; construct them per head inside the loop below.

        # Compute lse per head
        for h in range(H):
            pid_bh = b * H + h

            # qn = q_nope[b, h], qp = q_pe[b, h] (already bfloat16 on device, convert to float32)
            qn_vec = q_nope[b, h].to(torch.float32).contiguous()  # [Dc]
            qp_vec = q_pe[b, h].to(torch.float32).contiguous()   # [Dp]

            # Launch compute_logits_single_kernel: grid (1, L_b) with program_id(1)=l per program
            # We need a grid over (B*H, L_b); but since we loop h, we can set grid=(L_b,)
            # Each program will receive qn_ptr and qp_ptr via pointer arithmetic — here we pass them as separate tensors.
            # Triton doesn't support dynamic indexing into tensors; so we launch compute per head by calling kernel again for each h.
            # Simpler approach: create separate kernels per (b,h). The following launches with appropriate pointers.

            # For Triton, we can't easily pass qn_ptr and qp_ptr to the kernel; so we instead compute qn @ Kc and qp @ Kp using
            # separate kernels. To satisfy TRITON-ONLY, we implement compute per (b,h) using a grid over L_b and vector pointers
            # for qn and qp — but Triton requires static loops; thus we compute qn and qp as vectors per kernel call.
            # Implement compute_logits_single_kernel launch:
            # We need to pass qn_ptr and qp_ptr; Triton cannot index by h inside the kernel automatically, so we call this kernel
            # H times, each time passing q_nope[b, h] and q_pe[b, h] as pointers.
            # To do that, we reconstruct pointers: Triton expects pointers, but we can pass q_nope[b, h].contiguous() and q_pe[b, h].
            # Launch:
            compute_logits_single_kernel[(L_b,)](
                qn_vec, qp_vec, Kc_b, Kp_b, logits_scaled,
                L_b=L_b, Dc=Dc, Dp=Dp,
                sm_scale=float(sm_scale),
                BLOCK_D=64, BLOCK_P=32,
            )

            # Now compute base-2 logsumexp for this (b,h)
            compute_lse_kernel[(1,)](
                logits_scaled, lse[b, h],
                L_b=L_b, sm_scale=float(sm_scale),
            )

            # Compute softmax of scaled logits into attn
            compute_softmax_kernel[(1,)](
                logits_scaled, attn,
                L_b=L_b, sm_scale=float(sm_scale),
            )

            # Compute output vector out[h, :]
            out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
            compute_out_kernel[(1,)](
                attn, Kc_b, out_vec,
                L_b=L_b, Dc=Dc, BLOCK_L=64,
            )
            output[b, h, :] = out_vec.to(torch.bfloat16)

    return output, lse


# Optional helpers matching the original
def get_inputs():
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


# Optional wrapper using ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)