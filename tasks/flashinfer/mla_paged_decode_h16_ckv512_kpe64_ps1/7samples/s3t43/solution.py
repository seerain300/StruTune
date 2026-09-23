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
    pid_bh = tl.program_id(0)  # linear over B*H
    l = tl.program_id(1)       # token index in [0..L_b-1]
    # Compute b and h from pid_bh
    H = 16  # constant in the problem setup
    b = pid_bh // H
    h = pid_bh % H

    # Load qn[h, :] and qp[h, :] as float32
    qn = tl.load(qn_ptr)      # [Dc], float32
    qp = tl.load(qp_ptr)      # [Dp], float32

    # Accumulator for logits_scaled[l]
    acc = 0.0

    # Loop over Dc in chunks
    for d0 in tl.static_range(0, Dc, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        # For each d, load Kc[l, d] and accumulate sum(qn[d] * Kc[l, d])
        for dd in tl.static_range(0, BLOCK_D):
            d_idx = d0 + dd
            mask = d_idx < Dc
            Kc_val = tl.load(Kc_ptr + l * Dc + d_idx, mask=mask, other=0.0)
            acc += qn[d_idx] * Kc_val

    # Loop over Dp in chunks and add qp @ Kp[l, :]
    for p0 in tl.static_range(0, Dp, BLOCK_P):
        p = p0 + tl.arange(0, BLOCK_P)
        for pp in tl.static_range(0, BLOCK_P):
            p_idx = p0 + pp
            mask = p_idx < Dp
            Kp_val = tl.load(Kp_ptr + l * Dp + p_idx, mask=mask, other=0.0)
            acc += qp[p_idx] * Kp_val

    # Apply scaling
    acc *= sm_scale
    # Store scaled logits for this token
    tl.store(scale_ptr + l, acc)


# Triton kernel: compute base-2 logsumexp for one (b, h).
# Inputs: scale_ptr -> [L_b] logits_scaled vector
@triton.jit
def compute_lse_kernel(scale_ptr, lse_ptr,
                       L_b: tl.constexpr, sm_scale: tl.float32):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Pass 1: max
    max_val = -float("inf")
    for l in tl.static_range(0, L_b, 1):
        val = tl.load(scale_ptr + l)
        if val > max_val:
            max_val = val

    # Pass 2: sum of exp(val - max)
    sum_exp = 0.0
    for l in tl.static_range(0, L_b, 1):
        val = tl.load(scale_ptr + l)
        sum_exp += tl.exp(val - max_val)

    lse = tl.log(sum_exp) + max_val  # ln-sumexp
    # base-2: divide by ln(2)
    lse = lse / 0.6931471805599453
    tl.store(lse_ptr + b * H + h, lse)


# Triton kernel: compute softmax over logits_scaled for one (b, h).
# Writes attn vector into attn_ptr[l] for l in [0..L_b-1]
@triton.jit
def compute_softmax_kernel(scale_ptr, attn_ptr,
                           L_b: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Pass 1: max
    max_val = -float("inf")
    for l in tl.static_range(0, L_b, 1):
        val = tl.load(scale_ptr + l)
        if val > max_val:
            max_val = val

    # Pass 2: sum of exp
    sum_exp = 0.0
    for l in tl.static_range(0, L_b, 1):
        val = tl.load(scale_ptr + l)
        sum_exp += tl.exp(val - max_val)

    # Pass 3: write normalized probabilities
    for l in tl.static_range(0, L_b, 1):
        val = tl.load(scale_ptr + l)
        attn = tl.exp(val - max_val) / sum_exp
        tl.store(attn_ptr + l, attn)


# Triton kernel: compute out[h, :] = attn @ Kc_b (chunked reduction over L_b)
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr,
                       L_b: tl.constexpr, Dc: tl.constexpr, BLOCK_L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Accumulator for out[h, :]
    out_vec = tl.zeros((Dc,), dtype=tl.float32)

    for l0 in tl.static_range(0, L_b, BLOCK_L):
        l_idx = l0 + tl.arange(0, BLOCK_L)
        mask = l_idx < L_b
        # Load attn[l_idx] vector
        attn_vec = tl.load(attn_ptr + l_idx, mask=mask, other=0.0)
        # For each dim d in Dc, accumulate sum_l attn[l] * Kc[l, d]
        for d in tl.static_range(0, Dc):
            # Accumulate over this chunk
            acc = 0.0
            for i in tl.static_range(0, BLOCK_L):
                li = l0 + i
                mi = li < L_b
                Kc_val = tl.load(Kc_ptr + li * Dc + d, mask=mi, other=0.0)
                acc += attn_vec[i] * Kc_val
            out_vec[d] += acc

    # Store out[h, :]
    tl.store(out_ptr + h * Dc + tl.arange(0, Dc), out_vec)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    B, H = q_nope.shape[0], q_nope.shape[1]
    Dc = q_nope.shape[2]
    Dp = q_pe.shape[2]

    # Derived from kv_indptr
    assert kv_indptr.dim() == 1
    assert kv_indptr[0].item() == 0
    assert kv_indptr[-1].item() == torch.sum(qv_indptr[-1]).item(), "kv_indptr not valid"

    # Prepare output
    output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # Constants for kernels
    BLOCK_D = 128  # for Dc=512
    BLOCK_P = 16   # for Dp=64
    BLOCK_L = 64   # token chunk for reduction

    # Iterate per batch b
    for b in range(B):
        # Compute number of tokens for this batch element
        if b < kv_indptr.numel() - 1:
            L_b = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
        else:
            L_b = 0
        if L_b <= 0:
            # No tokens: output zero and lse -inf
            output[b].zero_()
            lse[b * H: (b + 1) * H].fill_(-float("inf"))
            continue

        # Gather token indices for this batch
        tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.int32).to(device)  # [L_b]
        # Gather Kc_b and Kp_b: [L_b, Dc] and [L_b, Dp]
        Kc_b = ckv_cache[tok_idx, 0].to(torch.float32)  # [L_b, Dc]
        Kp_b = kpe_cache[tok_idx, 0].to(torch.float32)  # [L_b, Dp]

        # Per-head vectors qn and qp
        for h in range(H):
            # Pointers to qn[h, :] and qp[h, :]
            qn = q_nope[b, h].to(torch.float32)  # [Dc]
            qp = q_pe[b, h].to(torch.float32)   # [Dp]

            # 1) Compute logits_scaled[l] for all l in Triton
            scale = torch.empty(L_b, dtype=torch.float32, device=device)
            grid_logit = (B * H, L_b)
            compute_logits_single_kernel[grid_logit](
                qn, qp, Kc_b, Kp_b, scale,
                L_b=L_b, Dc=Dc, Dp=Dp,
                sm_scale=float(sm_scale),
                BLOCK_D=BLOCK_D, BLOCK_P=BLOCK_P,
            )

            # 2) Compute base-2 logsumexp lse[h]
            grid_lse = (B, H)
            compute_lse_kernel[grid_lse](
                scale, lse,
                L_b=L_b, sm_scale=float(sm_scale),
            )

            # 3) Compute softmax attn[h, :]
            attn = torch.empty(L_b, dtype=torch.float32, device=device)
            grid_softmax = (B, H)
            compute_softmax_kernel[grid_softmax](
                scale, attn,
                L_b=L_b,
            )

            # 4) Compute out[h, :] = attn @ Kc_b via Triton
            out_vec = torch.empty(Dc, dtype=torch.float32, device=device)
            grid_out = (B, H)
            compute_out_kernel[grid_out](
                attn, Kc_b, out_vec,
                L_b=L_b, Dc=Dc, BLOCK_L=BLOCK_L,
            )

            # Store output
            output[b, h, :] = out_vec.to(torch.bfloat16)

    return output, lse


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


# Optional wrapper using ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
