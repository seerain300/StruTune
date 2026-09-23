import math
import torch
import triton
import triton.language as tl


# Triton kernels
@triton.jit
def compute_logits_kernel(
    qn_ptr,       # [H*Dc] flattened, q_nope[b, :, :] as float32
    qp_ptr,       # [H*Dp] flattened, q_pe[b, :, :] as float32
    Kc_ptr,       # [L*Dc] flattened
    Kp_ptr,       # [L*Dp] flattened
    logits_ptr,   # [H*L] flattened, output logits_scaled
    H: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr, L: tl.constexpr, sm_scale: tl.float32
):
    i = tl.program_id(0)  # head index
    t = tl.program_id(1)  # token index
    if i >= H or t >= L:
        return

    base_qn = i * Dc
    base_qp = i * Dp

    acc_qn = 0.0
    for k in range(0, Dc):
        q = tl.load(qn_ptr + base_qn + k)
        K = tl.load(Kc_ptr + t * Dc + k)
        acc_qn += q * K

    acc_qp = 0.0
    for k in range(0, Dp):
        q = tl.load(qp_ptr + base_qp + k)
        K = tl.load(Kp_ptr + t * Dp + k)
        acc_qp += q * K

    logit = acc_qn + acc_qp
    logit_scaled = logit * sm_scale
    tl.store(logits_ptr + i * L + t, logit_scaled)


@triton.jit
def row_max_kernel(
    logits_ptr,  # [H*L] flattened
    m_ptr,       # [H] float32
    H: tl.constexpr, L: tl.constexpr, BLOCK_L: tl.constexpr
):
    i = tl.program_id(0)
    if i >= H:
        return
    m = -float("inf")
    for offset in range(0, L, BLOCK_L):
        offs = offset + tl.arange(0, BLOCK_L)
        mask = offs < L
        vals = tl.load(logits_ptr + i * L + offs, mask=mask, other=-float("inf"))
        m_chunk = tl.max(vals, axis=0)
        m = tl.maximum(m, m_chunk)
    tl.store(m_ptr + i, m)


@triton.jit
def row_sumexp_kernel(
    logits_ptr,   # [H*L] flattened
    sum_ptr,      # [H] float32
    m_ptr,        # [H] float32
    H: tl.constexpr, L: tl.constexpr, BLOCK_L: tl.constexpr
):
    i = tl.program_id(0)
    if i >= H:
        return
    m = tl.load(m_ptr + i)
    sum_exp = 0.0
    for offset in range(0, L, BLOCK_L):
        offs = offset + tl.arange(0, BLOCK_L)
        mask = offs < L
        vals = tl.load(logits_ptr + i * L + offs, mask=mask, other=-float("inf"))
        vals = vals - m
        exp_vals = tl.exp(vals)
        exp_vals = tl.where(mask, exp_vals, 0.0)
        sum_exp += tl.sum(exp_vals, axis=0)
    tl.store(sum_ptr + i, sum_exp)


@triton.jit
def softmax_row_kernel(
    logits_ptr,    # [H*L] flattened
    m_ptr,         # [H] float32
    sum_ptr,       # [H] float32
    attn_ptr,      # [H*L] flattened, output softmax
    H: tl.constexpr, L: tl.constexpr, BLOCK_L: tl.constexpr
):
    i = tl.program_id(0)
    if i >= H:
        return
    m_i = tl.load(m_ptr + i)
    sum_exp_i = tl.load(sum_ptr + i)
    inv_ln2 = 1.0 / math.log(2.0)
    for offset in range(0, L, BLOCK_L):
        offs = offset + tl.arange(0, BLOCK_L)
        mask = offs < L
        vals = tl.load(logits_ptr + i * L + offs, mask=mask, other=-float("inf"))
        probs = tl.exp(vals - m_i) / sum_exp_i
        probs = probs * inv_ln2
        probs = tl.where(mask, probs, 0.0)
        tl.store(attn_ptr + i * L + offs, probs, mask=mask)


@triton.jit
def matvec_kernel_2d(
    attn_ptr,      # [H*L] flattened
    Kc_ptr,        # [L*Dc] flattened
    out_ptr,       # [H*Dc] flattened
    H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr, BLOCK_D: tl.constexpr
):
    i = tl.program_id(0)  # head index
    db = tl.program_id(1) # d-block index
    if i >= H:
        return
    d_offsets = db * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < Dc

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t in range(0, L):
        attn_t = tl.load(attn_ptr + i * L + t)  # scalar
        K_vec = tl.load(Kc_ptr + t * Dc + d_offsets, mask=d_mask, other=0.0)
        acc += attn_t * K_vec

    out_base = i * Dc
    tl.store(out_ptr + out_base + d_offsets, acc, mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device

        # Extract shapes
        B = q_nope.shape[0]
        H = q_nope.shape[1]  # num_qo_heads
        Dc = q_nope.shape[2]  # head_dim_ckv
        Dp = q_pe.shape[2]    # head_dim_kpe

        # Squeeze caches (size-1 dim) and cast to float32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Dp]

        # Output tensors
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                lse[b].zero_()
                for i in range(H):
                    output[b, i] = torch.zeros((Dc,), dtype=torch.bfloat16, device=device)
                continue

            # Gather token indices
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32).contiguous()  # [L_tokens]

            # Flatten caches and gather Kc_flat and Kp_flat
            Kc_flat = torch.empty((L_tokens * Dc,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * Dp,), dtype=torch.float32, device=device)

            grid_gather = (L_tokens,)
            gather_rows_c_kernel[grid_gather](Kc_all, tok_idx, Kc_flat, L_tokens, Dc)
            Kc = Kc_flat.view(L_tokens, Dc)  # [L_tokens, Dc]

            gather_rows_p_kernel[grid_gather](Kp_all, tok_idx, Kp_flat, L_tokens, Dp)
            Kp = Kp_flat.view(L_tokens, Dp)  # [L_tokens, Dp]

            # Prepare q_nope[b] and q_pe[b] as flattened float32
            qn_flat = q_nope[b].to(torch.float32).contiguous().view(-1)  # [H*Dc]
            qp_flat = q_pe[b].to(torch.float32).contiguous().view(-1)   # [H*Dp]

            # Allocate flattened logits_scaled [H*L_tokens]
            logits_scaled = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)

            # Launch compute_logits_kernel: grid=(H, L_tokens)
            compute_logits_kernel[(H, L_tokens)](
                qn_flat, qp_flat, Kc.contiguous().view(-1), Kp.contiguous().view(-1),
                logits_scaled, H=H, Dc=Dc, Dp=Dp, L=L_tokens, sm_scale=float(sm_scale)
            )

            # Reductions in Triton
            m = torch.empty((H,), dtype=torch.float32, device=device)  # per head max
            sum_exp = torch.empty((H,), dtype=torch.float32, device=device)  # per head sum exp

            row_max_kernel[(H,)](
                logits_scaled, m, H=H, L=L_tokens, BLOCK_L=128
            )

            row_sumexp_kernel[(H,)](
                logits_scaled, sum_exp, m, H=H, L=L_tokens, BLOCK_L=128
            )

            # Compute lse per head in base-2: lse[b, i] = m[i] + log(sum_exp[i]) / ln(2)
            inv_ln2 = 1.0 / math.log(2.0)
            lse[b] = m + torch.log(sum_exp) * inv_ln2

            # Compute attention weights using softmax_row_kernel: grid=(H, L_tokens


def run(*args):
    return ModelNew()(*args)
