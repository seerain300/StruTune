import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_per_batch_kernel(
    qn_ptr,         # *fp32, [H, D], contiguous
    qp_ptr,         # *fp32, [H, Dp], contiguous
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, [L_tokens, Dp], contiguous
    tok_idx_ptr,    # *int32, [L_tokens]
    logits_ptr,     # *fp32, flattened [B*H*L_tokens], contiguous
    H: tl.int32,
    D: tl.int32,
    Dp: tl.int32,
    L_tokens: tl.int32,
    b: tl.int32,
    sm_scale: tl.float32,
):
    # 2D grid over heads and tokens
    h = tl.program_id(0)
    t = tl.program_id(1)

    if (h >= H) or (t >= L_tokens):
        return

    # Compute flat index for logits[b, h, t]
    idx = (b * H + h) * L_tokens + t

    # Load qn row for head h: [D]
    offs = tl.arange(0, D)
    qn_row = tl.load(qn_ptr + h * D + offs, mask=offs < D, other=0.0)

    # Load qp row for head h: [Dp]
    koffs = tl.arange(0, Dp)
    qp_row = tl.load(qp_ptr + h * Dp + koffs, mask=koffs < Dp, other=0.0)

    # Gather token index for this token
    tok_idx_t = tl.load(tok_idx_ptr + t)  # int32 scalar

    # Load Kc row for token t: [D]
    Kc_row = tl.load(Kc_ptr + tok_idx_t * D + offs, mask=offs < D, other=0.0)
    # Load Kp row for token t: [Dp]
    Kp_row = tl.load(Kp_ptr + tok_idx_t * Dp + koffs, mask=koffs < Dp, other=0.0)

    # Compute dot products
    acc1 = 0.0
    for kk in range(0, D):
        acc1 += qn_row[kk] * Kc_row[kk]
    acc2 = 0.0
    for kk in range(0, Dp):
        acc2 += qp_row[kk] * Kp_row[kk]

    logits_val = acc1 + acc2
    # Scale logits
    logits_val = logits_val * sm_scale
    # Store
    tl.store(logits_ptr + idx, logits_val)


@triton.jit
def lse_per_head_kernel(
    logits_ptr,     # *fp32, flattened [B*H*L_tokens], contiguous
    lse_ptr,        # *fp32, flattened [B*H], contiguous
    H: tl.int32,
    D: tl.int32,    # unused
    Dp: tl.int32,   # unused
    L_tokens: tl.int32,
    b: tl.int32,
    sm_scale: tl.float32,
):
    h = tl.program_id(0)
    if h >= H:
        return
    base = (b * H + h) * L_tokens

    # Compute max for numerical stability
    max_val = -1e30
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + base + t) * sm_scale
        if val > max_val:
            max_val = val

    # Compute sum of exp(logits - max)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + base + t) * sm_scale
        sum_exp += tl.exp(val - max_val)

    # lse = log(sum_exp) / log(2)
    logsum = tl.log(sum_exp)
    lse_val = logsum / 0.6931471805599453  # 1 / log(2)
    # Store to lse[b, h]
    tl.store(lse_ptr + b * H + h, lse_val)


@triton.jit
def compute_output_per_batch_kernel(
    logits_ptr,     # *fp32, flattened [B*H*L_tokens], contiguous
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, [L_tokens, Dp], contiguous
    tok_idx_ptr,    # *int32, [L_tokens]
    out_ptr,        # *fp32, flattened [B*H*D], contiguous
    H: tl.int32,
    D: tl.int32,
    Dp: tl.int32,
    L_tokens: tl.int32,
    b: tl.int32,
    sm_scale: tl.float32,
):
    h = tl.program_id(0)
    if h >= H:
        return

    base_logits = (b * H + h) * L_tokens

    # First pass: compute max for numerical stability
    max_val = -1e30
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + base_logits + t) * sm_scale
        if val > max_val:
            max_val = val

    # Second pass: compute sum of exp and attn
    sum_exp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + base_logits + t) * sm_scale
        sum_exp += tl.exp(val - max_val)

    # Initialize output vector for head h
    # We will update each element kk in [0, D)
    for kk in range(0, D):
        tl.store(out_ptr + (b * H + h) * D + kk, 0.0)

    # Compute and accumulate attn contributions
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + base_logits + t) * sm_scale
        attn_t = tl.exp(val - max_val) / sum_exp  # scalar

        # Gather token index for this token
        tok_idx_t = tl.load(tok_idx_ptr + t)  # int32 scalar

        # Load Kc row for token t: [D]
        offs = tl.arange(0, D)
        Kc_row = tl.load(Kc_ptr + tok_idx_t * D + offs, mask=offs < D, other=0.0)

        # out[h, :] += attn_t * Kc_row
        for kk in range(0, D):
            out_k = tl.load(out_ptr + (b * H + h) * D + kk)
            out_k += attn_t * Kc_row[kk]
            tl.store(out_ptr + (b * H + h) * D + kk, out_k)

    # No vectorized store at the end; loop accumulates per element.


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA and dtype/device compatibility
        device = q_nope.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors."

        # Cast to float32 for computation and make contiguous
        q_nope_f32 = q_nope.to(torch.float32).contiguous()   # [B, H, D]
        q_pe_f32 = q_pe.to(torch.float32).contiguous()       # [B, H, Dp]

        # Kc/Kp caches: [N_total, D] and [N_total, Dp]
        Kc_all = ckv_cache.to(torch.float32).squeeze(1).contiguous()  # [N_total, D]
        Kp_all = kpe_cache.to(torch.float32).squeeze(1).contiguous()  # [N_total, Dp]

        B = q_nope_f32.shape[0]
        H = q_nope_f32.shape[1]
        D = q_nope_f32.shape[2]
        Dp = q_pe_f32.shape[2]

        # Compute per-batch L_tokens using kv_indptr
        L_tokens = [0] * B
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            N_total = end - start
            # Cap by available kv_indices length
            L = min(N_total, len(kv_indices))
            L_tokens[b] = L

        # Prepare tok_idx per batch element: slice kv_indices[start:start+L_tokens[b]]
        tok_idx_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            L = int(L_tokens[b])
            idx_slice = kv_indices[start:start + L].to(torch.int32).contiguous()
            tok_idx_list.append(idx_slice)

        # Allocate outputs (flat buffers for Triton kernels)
        logits_flat = torch.empty(B * H * max(L_tokens), dtype=torch.float32, device=device)
        output_flat = torch.empty(B * H * D, dtype=torch.float32, device=device)
        lse_flat = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch kernels per batch element
        for b in range(B):
            Hb = H
            Db = D
            Dpb = Dp
            Lb = int(L_tokens[b])

            # First kernel: compute all logits[h, t]
            # Grid over (H, Lb)
            grid = (Hb, Lb)
            compute_logits_per_batch_kernel[grid](
                q_nope_f32[b], q_pe_f32[b], Kc_all, Kp_all, tok_idx_list[b],
                logits_flat, H=Hb, D=Db, Dp=Dpb, L_tokens=Lb, b=b, sm_scale=float(sm_scale)
            )

            # Second kernel: lse for each head
            lse_per_head_kernel[(Hb,)](
                logits_flat, lse_flat, H=Hb, D=Db, Dp=Dpb, L_tokens=Lb, b=b, sm_scale=float(sm_scale)
            )

            # Third kernel: compute output per head (initialize output to zeros)
            output_flat.zero_()
            compute_output_per_batch_kernel[(Hb,)](
                logits_flat, Kc_all, Kp_all, tok_idx_list[b], output_flat,
                H=Hb, D=Db, Dp=Dpb, L_tokens=Lb, b=b, sm_scale=float(sm_scale)
            )

        # Reshape outputs
        output = output_flat.view(B, H, D).to(torch.bfloat16)
        lse = lse_flat.view(B, H)

        return output, lse


def run(*args):
    return ModelNew()(*args)
