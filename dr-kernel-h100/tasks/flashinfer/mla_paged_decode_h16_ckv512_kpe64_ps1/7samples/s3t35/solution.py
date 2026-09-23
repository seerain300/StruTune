import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits_scaled vector for one (b, h) pair and store into logits_scaled_ptr.
# Inputs:
#   qn_ptr: pointer to q_nope[b, h] -> shape [Dc], float32
#   Kc_ptr: pointer to Kc base (gathered by host) -> shape [L_b, Dc], float32
#   Kp_ptr: pointer to Kp base (gathered by host) -> shape [L_b, Dp], float32
#   qp_ptr: pointer to q_pe[b, h] -> shape [Dp], float32
#   logits_scaled_ptr: pointer to output vector [L_b], float32
#   sm_scale: float32 scalar
# Meta:
#   L_b: tl.constexpr (number of tokens for this batch)
#   Dc: tl.constexpr (512)
#   Dp: tl.constexpr (64)
# Launch grid: (B, H)
@triton.jit
def compute_logits_kernel_bh(qn_ptr, Kc_ptr, Kp_ptr, qp_ptr, logits_scaled_ptr, L_b: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr, sm_scale: tl.float32, BLOCK_L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load qn[h, :] and qp[h, :] as vectors
    qn = tl.load(qn_ptr)  # [Dc]
    qp = tl.load(qp_ptr)  # [Dp]

    # Accumulator for logits for this (b,h)
    logits_acc = tl.zeros((L_b,), dtype=tl.float32)

    # Loop over tokens in chunks
    for l_off in tl.static_range(0, L_b, BLOCK_L):
        l_idx = l_off + tl.arange(0, BLOCK_L)  # [BLOCK_L]
        mask = l_idx < L_b

        # Initialize accumulators for dot products
        dot1 = tl.zeros((), dtype=tl.float32)  # for qn @ Kc[l,:].T
        dot2 = tl.zeros((), dtype=tl.float32)  # for qp @ Kp[l,:].T

        # Reduce over Dc and Dp
        # Note: Kc_ptr + l_idx[:, None]*Dc + tl.arange(0, Dc) -> [BLOCK_L, Dc]
        # We compute dot1 per l_idx by summing over Dc
        for d in tl.static_range(0, Dc):
            kc_vec = tl.load(Kc_ptr + l_idx * Dc + d, mask=mask, other=0.0)  # [BLOCK_L]
            dot1 += tl.sum(kc_vec * qn[d])  # scalar

        for d in tl.static_range(0, Dp):
            kp_vec = tl.load(Kp_ptr + l_idx * Dp + d, mask=mask, other=0.0)  # [BLOCK_L]
            dot2 += tl.sum(kp_vec * qp[d])  # scalar

        logits_chunk = dot1 + dot2  # [BLOCK_L] (actually scalar broadcast)
        # Scale
        logits_chunk = logits_chunk * sm_scale
        # Store chunk
        # We need to store per l in l_idx; for masked positions, we store 0
        for i in tl.static_range(0, BLOCK_L):
            if mask[i]:
                tl.store(logits_scaled_ptr + l_idx[i], logits_chunk)

    # After loop, logits_scaled_ptr is fully written


# Triton kernel: compute base-2 logsumexp for a vector of length L_b.
# Inputs:
#   logits_scaled_ptr: pointer to float32 vector [L_b]
#   lse_ptr: pointer to float32 scalar for this (b,h)
#   L_b: tl.constexpr
# Launch grid: (B, H)
@triton.jit
def compute_lse_kernel(logits_scaled_ptr, lse_ptr, L_b: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Load logits vector
    # We can't index a scalar vector directly; instead, we rely on host to pass a single lse per (b,h).
    # Use a temporary to compute max and sumexp.
    # max
    m = tl.full((), -float('inf'), dtype=tl.float32)
    for i in tl.static_range(0, L_b):
        v = tl.load(logits_scaled_ptr + i)
        if v > m:
            m = v

    # sumexp
    s = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(0, L_b):
        v = tl.load(logits_scaled_ptr + i)
        s += tl.exp(v - m)

    lse_val = m + tl.log(s)  # ln(sum exp)
    # Convert to base-2: lse_base2 = lse_ln / ln(2)
    ln2 = 0.6931471805599453
    lse_val = lse_val / ln2
    tl.store(lse_ptr, lse_val)


# Triton kernel: compute softmax over a vector of length L_b and store attn.
# Inputs:
#   logits_scaled_ptr: pointer to float32 vector [L_b]
#   attn_ptr: pointer to float32 vector [L_b]
#   L_b: tl.constexpr
# Launch grid: (B, H)
@triton.jit
def compute_softmax_kernel(logits_scaled_ptr, attn_ptr, L_b: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Load m
    m = tl.full((), -float('inf'), dtype=tl.float32)
    for i in tl.static_range(0, L_b):
        v = tl.load(logits_scaled_ptr + i)
        if v > m:
            m = v

    for i in tl.static_range(0, L_b):
        v = tl.load(logits_scaled_ptr + i)
        attn_val = tl.exp(v - m)
        # Normalize later by denominator; we'll compute denominator first.

    # Compute denominator
    den = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(0, L_b):
        v = tl.load(logits_scaled_ptr + i)
        den += tl.exp(v - m)

    # Write normalized attn
    for i in tl.static_range(0, L_b):
        v = tl.load(logits_scaled_ptr + i)
        attn_val = tl.exp(v - m) / den
        tl.store(attn_ptr + i, attn_val)


# Triton kernel: compute out[h, :] = attn @ Kc for this batch b and head h.
# Inputs:
#   attn_ptr: pointer to float32 vector [L_b]
#   Kc_ptr: pointer to Kc base for this batch -> [L_b, Dc], float32
#   out_ptr: pointer to output vector [Dc], float32
#   L_b: tl.constexpr
#   Dc: tl.constexpr
# Launch grid: (B, H)
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr, L_b: tl.constexpr, Dc: tl.constexpr, BLOCK_L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Accumulator for output vector
    out_vec = tl.zeros((Dc,), dtype=tl.float32)

    # Reduce over tokens in chunks
    for l_off in tl.static_range(0, L_b, BLOCK_L):
        l_idx = l_off + tl.arange(0, BLOCK_L)
        mask = l_idx < L_b

        # Load attn_chunk: [BLOCK_L]
        attn_chunk = tl.zeros((BLOCK_L,), dtype=tl.float32)
        for i in tl.static_range(0, BLOCK_L):
            if mask[i]:
                attn_chunk[i] = tl.load(attn_ptr + l_idx[i])

        # Accumulate out_vec += sum(attn_chunk * Kc[l,:])
        for d in tl.static_range(0, Dc):
            kc_vec = tl.zeros((BLOCK_L,), dtype=tl.float32)
            for i in tl.static_range(0, BLOCK_L):
                if mask[i]:
                    kc_vec[i] = tl.load(Kc_ptr + l_idx[i] * Dc + d)
            out_vec[d] += tl.sum(attn_chunk * kc_vec)  # scalar accumulation

    # Store result
    for d in tl.static_range(0, Dc):
        tl.store(out_ptr + d, out_vec[d])


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Device and dtype setup
    device = q_nope.device
    B, H, Dc = q_nope.shape
    _, _, Dp = q_pe.shape
    # Cast inputs for compute to float32
    q_nope_f32 = q_nope.to(torch.float32)
    q_pe_f32 = q_pe.to(torch.float32)
    # Process each batch b
    # Initialize output and lse
    output = torch.empty((B, H, Dc), dtype=torch.float32, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # Helper: for each batch b, compute tok_idx
    # We'll launch kernels per (b,h).
    # Note: Triton launches require a grid. Here, grid is (B, H).
    for b in range(B):
        # Compute token indices for this batch
        # tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
        num_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        L_b = num_tokens
        if L_b <= 0:
            # No tokens for this batch
            # For correctness: set output zeros and lse -inf, then continue
            output[b].zero_()
            lse[b].fill_(float('-inf'))
            continue

        # Gather Kc and Kp for this batch
        # ckv_cache and kpe_cache are [N, 1, D], we use channel 0
        # Since kv_indices are ints, slice with .index_select requires same dtype device
        tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.long).to(device)
        Kc_b = ckv_cache[tok_idx, 0].contiguous().to(torch.float32)  # [L_b, Dc]
        Kp_b = kpe_cache[tok_idx, 0].contiguous().to(torch.float32)  # [L_b, Dp]

        # Prepare per-head vectors
        # We need to launch per (b,h) to handle different heads. Use a loop over h and Triton grid (B, H).
        for h in range(H):
            # Prepare pointers
            qn_ptr = q_nope_f32[b, h]  # [Dc]
            qp_ptr = q_pe_f32[b, h]    # [Dp]
            # Allocate logits_scaled buffer [L_b]
            logits_scaled = torch.empty((L_b,), dtype=torch.float32, device=device)

            # Launch compute_logits_kernel_bh
            # Choose BLOCK sizes
            BLOCK_L = 128  # chunk size for token loop
            # Meta-parameters
            Dc_const = Dc  # 512
            Dp_const = Dp  # 64

            compute_logits_kernel_bh[(B, H)](
                qn_ptr, Kc_b, Kp_b, qp_ptr, logits_scaled,
                L_b=L_b, Dc=Dc_const, Dp=Dp_const, sm_scale=float(sm_scale),
                BLOCK_L=BLOCK_L,
                num_warps=4
            )

            # Launch compute_lse_kernel
            lse[b, h] = torch.zeros((), dtype=torch.float32, device=device)  # placeholder for store
            compute_lse_kernel[(B, H)](
                logits_scaled, lse[b, h],
                L_b=L_b
            )

            # Launch compute_softmax_kernel
            attn = torch.empty((L_b,), dtype=torch.float32, device=device)
            compute_softmax_kernel[(B, H)](
                logits_scaled, attn,
                L_b=L_b
            )

            # Launch compute_out_kernel: out[h, :] = attn @ Kc_b
            out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
            compute_out_kernel[(B, H)](
                attn, Kc_b, out_vec,
                L_b=L_b, Dc=Dc_const, BLOCK_L=BLOCK_L,
                num_warps=4
            )

            # Store output for this head
            output[b, h, :] = out_vec

    # Cast output to bfloat16 to match original
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


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


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
