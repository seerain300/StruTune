import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute output and lse for a single (batch b, head h).
# Assumes:
# - Kc_all: [num_pages, Dc]
# - Kp_all: [num_pages, Dp]
# - L_tokens: number of tokens for this batch element (computed on host).
@triton.jit
def _compute_single_head_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    out_ptr, lse_ptr,
    B: tl.constexpr, H: tl.constexpr,
    Dc: tl.constexpr, Dp: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.float32
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load qn and qp vectors for head h (fp32)
    # q_nope shape [B, H, Dc], contiguous, offset for (b,h) is b*H*Dc + h*Dc
    qn = tl.load(
        q_nope_ptr + b * H * Dc + h * Dc + tl.arange(0, Dc),
        mask=tl.arange(0, Dc) < Dc,
        other=0.0
    ).to(tl.float32)

    # q_pe shape [B, H, Dp], contiguous, offset for (b,h) is b*H*Dp + h*Dp
    qp = tl.load(
        q_pe_ptr + b * H * Dp + h * Dp + tl.arange(0, Dp),
        mask=tl.arange(0, Dp) < Dp,
        other=0.0
    ).to(tl.float32)

    # First pass: compute max of logits_scaled for numerical stability
    max_val = -float("inf")
    for t in tl.static_range(0, L_tokens):
        # sum_i qn[i] * Kc[t, i]
        sum_qn_Kc = 0.0
        for i in tl.static_range(0, Dc):
            sum_qn_Kc += qn[i] * tl.load(Kc_all_ptr + t * Dc + i).to(tl.float32)
        # sum_j qp[j] * Kp[t, j]
        sum_qp_Kp = 0.0
        for j in tl.static_range(0, Dp):
            sum_qp_Kp += qp[j] * tl.load(Kp_all_ptr + t * Dp + j).to(tl.float32)
        logits = sum_qn_Kc + sum_qp_Kp
        logits_scaled = logits * sm_scale
        # track max
        if logits_scaled > max_val:
            max_val = logits_scaled

    # Second pass: sum of exp(logits_scaled - max_val)
    sum_exp = 0.0
    for t in tl.static_range(0, L_tokens):
        sum_qn_Kc = 0.0
        for i in tl.static_range(0, Dc):
            sum_qn_Kc += qn[i] * tl.load(Kc_all_ptr + t * Dc + i).to(tl.float32)
        sum_qp_Kp = 0.0
        for j in tl.static_range(0, Dp):
            sum_qp_Kp += qp[j] * tl.load(Kp_all_ptr + t * Dp + j).to(tl.float32)
        logits = sum_qn_Kc + sum_qp_Kp
        logits_scaled = logits * sm_scale
        sum_exp += tl.exp(logits_scaled - max_val)

    # lse in natural log domain; final result should be logsumexp / ln(2)
    lse = tl.log(sum_exp) + max_val  # natural log
    lse = lse / math.log(2.0)        # base-2 normalization

    # Third pass: compute attn per token and accumulate output vector
    out_vec = tl.zeros([Dc], dtype=tl.float32)
    for t in tl.static_range(0, L_tokens):
        sum_qn_Kc = 0.0
        for i in tl.static_range(0, Dc):
            sum_qn_Kc += qn[i] * tl.load(Kc_all_ptr + t * Dc + i).to(tl.float32)
        sum_qp_Kp = 0.0
        for j in tl.static_range(0, Dp):
            sum_qp_Kp += qp[j] * tl.load(Kp_all_ptr + t * Dp + j).to(tl.float32)
        logits = sum_qn_Kc + sum_qp_Kp
        logits_scaled = logits * sm_scale
        attn_t = tl.exp(logits_scaled - lse) / math.log(2.0)  # softmax in base-2
        Kc_t = tl.load(Kc_all_ptr + t * Dc + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        out_vec += attn_t * Kc_t

    # Store results
    # lse: write float32 to [B, H] at (b, h)
    tl.store(lse_ptr + b * H + h, lse)
    # out: write bfloat16 to [B, H, Dc] at (b, h, :)
    out_offset = b * H * Dc + h * Dc
    for i in tl.static_range(0, Dc):
        tl.store(out_ptr + out_offset + i, out_vec[i].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device

        # Convert caches to fp32 contiguous for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Compute L_tokens per batch from kv_indptr (assumes len_indptr == B+1)
        # L_tokens[b] = int(kv_indptr[b+1]) - int(kv_indptr[b])
        # Note: torch operations here are necessary for host-side computation of L_tokens
        L_tokens_list = (kv_indptr[1:] - kv_indptr[:kv_indptr.shape[0]-1]).to(torch.int32).tolist()
        # Ensure we have exactly B values
        if len(L_tokens_list) < B:
            raise RuntimeError("kv_indptr length must be B+1")
        L_tokens_list = L_tokens_list[:B]
        L_tokens = L_tokens_list  # list of ints per batch

        # Output tensor [B, H, Dc] bfloat16
        out = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        # lse tensor [B, H] float32
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _compute_single_head_kernel[grid](
            q_nope, q_pe,
            Kc_all, Kp_all,
            out, lse,
            B=B, H=H,
            Dc=Dc, Dp=Dp,
            L_tokens=L_tokens,  # list of ints per batch element
            sm_scale=float(sm_scale),
            num_warps=4,
            num_stages=2
        )

        return out, lse


def run(*args):
    return ModelNew()(*args)
