import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.constexpr, Dc: tl.constexpr):
    # Each program handles one token row; copy a single row [Dc] from cache into out
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


@triton.jit
def gather_rows_p_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.constexpr, Dp: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


@triton.jit
def lse_row_kernel(logits_ptr, lse_ptr,
                   L: tl.constexpr):
    # One program per row: compute logsumexp in base-2 for that row and write per-token lse
    row_id = tl.program_id(0)
    m = -float("inf")
    sum_exp = 0.0
    # First pass: max
    for t in range(0, L):
        val = tl.load(logits_ptr + row_id * L + t)
        m = tl.maximum(m, val)
    # Second pass: sum of exp(x - m)
    for t in range(0, L):
        val = tl.load(logits_ptr + row_id * L + t)
        sum_exp += tl.exp(val - m)
    lse_val = tl.log(sum_exp) / math.log(2.0)  # logsumexp / ln(2)
    tl.store(lse_ptr + row_id, lse_val)  # write per-token lse


@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr,
                        L: tl.constexpr):
    # One program per row: compute softmax for that row and write to attn_ptr
    row_id = tl.program_id(0)
    m = -float("inf")
    # First pass: max
    for t in range(0, L):
        val = tl.load(logits_ptr + row_id * L + t)
        m = tl.maximum(m, val)
    # Second pass: sum of exp(x - m)
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logits_ptr + row_id * L + t)
        sum_exp += tl.exp(val - m)
    # Third pass: write normalized attn
    for t in range(0, L):
        val = tl.load(logits_ptr + row_id * L + t)
        attn_val = tl.exp(val - m) / sum_exp
        tl.store(attn_ptr + row_id * L + t, attn_val)


@triton.jit
def matvec_kernel(attn_flat_ptr, K_flat_ptr, out_flat_ptr,
                   H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr,
                   BLOCK_D: tl.constexpr):
    # Grid: (H, ceil(Dc / BLOCK_D))
    pid_h = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_start = pid_d * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    mask = d_offsets < Dc

    # Accumulator for this head
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Sum over tokens: attn[j] * K[j, d]
    for j in range(0, L):
        attn_j = tl.load(attn_flat_ptr + pid_h * L + j)
        # K[j, d_offsets]
        Kj = tl.load(K_flat_ptr + j * Dc + d_offsets, mask=mask, other=0.0)
        acc += attn_j * Kj

    # Store results
    tl.store(out_flat_ptr + pid_h * Dc + d_offsets, acc, mask=mask)


@triton.jit
def sum_tokens_kernel(lse_vec_ptr, per_head_lse_ptr,
                       L: tl.constexpr):
    # One program per row: read per-token lse and atomically add to per_head_lse_ptr[row_id]
    row_id = tl.program_id(0)
    sum_val = 0.0
    for t in range(0, L):
        val = tl.load(lse_vec_ptr + row_id * L + t)
        sum_val += val
    tl.atomic_add(per_head_lse_ptr + row_id, sum_val)


# Constants
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        assert H == NUM_QO_HEADS
        assert Dc == HEAD_DIM_CKV
        assert Dp == HEAD_DIM_KPE

        # Squeeze caches
        Kc_all = ckv_cache.squeeze(1).contiguous()  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous()  # [P, Dp]

        # Output and lse
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Process each batch
        for b in range(B):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                for i in range(H):
                    output[b, i] = torch.zeros((Dc,), dtype=torch.bfloat16, device=q_nope.device)
                lse[b] = torch.zeros((H,), dtype=torch.float32, device=q_nope.device)
                continue

            # Token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # Gather Kc_flat and Kp_flat (float32)
            Kc_flat = torch.empty((L_tokens * Dc,), dtype=torch.float32, device=q_nope.device)
            Kp_flat = torch.empty((L_tokens * Dp,), dtype=torch.float32, device=q_nope.device)

            grid_g = (L_tokens,)
            gather_rows_c_kernel[grid_g](Kc_all, tok_idx, Kc_flat, L_tokens, Dc)
            Kc = Kc_flat.view(L_tokens, Dc)

            grid_g2 = (L_tokens,)
            gather_rows_p_kernel[grid_g2](Kp_all, tok_idx, Kp_flat, L_tokens, Dp)
            Kp = Kp_flat.view(L_tokens, Dp)

            # For each head: compute logits_scaled, lse, attn, and final output
            for i in range(H):
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]

                # Compute logits: qn @ Kc.T + qp @ Kp.T
                logits_qn = qn @ Kc.T                          # [1, L_tokens]
                logits_qp = qp @ Kp.T                          # [1, L_tokens]
                logits = (logits_qn + logits_qp).squeeze(0)    # [L_tokens]
                logits_scaled = logits * sm_scale              # [L_tokens], float32

                # 1) per-token lse (base-2): Triton kernel writes per-token lse to lse_vec[i, :]
                lse_vec = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                grid_lse = (L_tokens,)
                lse_row_kernel[grid_lse](logits_scaled, lse_vec)  # [L_tokens]

                # 2) reduce per head via Triton sum
                per_head_lse = torch.zeros((1,), dtype=torch.float32, device=q_nope.device)  # dummy 1-element buffer
                sum_tokens_kernel[grid_lse](lse_vec, per_head_lse, L=L_tokens)  # grid = (L_tokens


def run(*args):
    return ModelNew()(*args)
