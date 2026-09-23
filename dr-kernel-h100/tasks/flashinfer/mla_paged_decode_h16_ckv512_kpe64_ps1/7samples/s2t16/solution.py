import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute output and lse for a single (batch b, head h).
# We launch grid=(B, H). Each program handles one (b, h).
@triton.jit
def _compute_single_head(
    qn_ptr, qp_ptr,
    Kc_ptr, Kp_ptr,
    out_ptr, lse_ptr,
    B: tl.constexpr, H: tl.constexpr,
    Dc: tl.constexpr, Dp: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.float32,
):
    # Program id: one per (b, h)
    # Triton doesn't have direct pid_b, pid_h, so we derive from launch grid.
    # But since grid is (B, H), we can index using tl.program_id(0) == b, tl.program_id(1) == h.
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Initialize output vector
    out_vec = tl.zeros((Dc,), dtype=tl.float32)

    # Vector to hold logits (length L_tokens) in fp32
    logits = tl.zeros((L_tokens,), dtype=tl.float32)

    # Accumulate logits per token
    for t in range(L_tokens):
        # qn[h, :] dot Kc[t, :]
        sum_qn = 0.0
        for i in range(Dc):
            qn_i = tl.load(qn_ptr + i)  # qn[h, i] is at qn_ptr + i (stride-1 assumed)
            kc_ti = tl.load(Kc_ptr + t * Dc + i)  # Kc[t, i]
            sum_qn += qn_i * kc_ti

        # qp[h, :] dot Kp[t, :]
        sum_qp = 0.0
        for j in range(Dp):
            qp_j = tl.load(qp_ptr + j)  # qp[h, j] at qp_ptr + j (stride-1 assumed)
            kp_tj = tl.load(Kp_ptr + t * Dp + j)  # Kp[t, j]
            sum_qp += qp_j * kp_tj

        logits[t] = sum_qn + sum_qp  # fp32

    # Scale logits
    logits_scaled = logits * sm_scale

    # Compute lse = logsumexp(logits_scaled) / ln(2)
    m = tl.max(logits_scaled, axis=0)
    sumexp = 0.0
    for t in range(L_tokens):
        sumexp += tl.exp(logits_scaled[t] - m)
    lse = m + tl.log(sumexp) / 1.4426950408889634  # 1 / ln(2)

    # Compute attention vector: softmax over logits_scaled
    attn = tl.zeros((L_tokens,), dtype=tl.float32)
    for t in range(L_tokens):
        attn[t] = tl.exp(logits_scaled[t] - lse)

    # Accumulate output: out[h, :] = sum_t attn[t] * Kc[t, :]
    for i in range(Dc):
        row_sum = 0.0
        for t in range(L_tokens):
            row_sum += attn[t] * tl.load(Kc_ptr + t * Dc + i)
        out_vec[i] = row_sum

    # Store output as bfloat16 and lse as float32
    # out_ptr points to out[b, h, :] flattened
    tl.store(out_ptr + b * (H * Dc) + h * Dc + tl.arange(0, Dc), out_vec.to(tl.bfloat16))
    tl.store(lse_ptr + b * H + h, lse)


def _run_triton_only(
    q_nope: torch.Tensor, q_pe: torch.Tensor, Kc_all: torch.Tensor, Kp_all: torch.Tensor,
    kv_indptr: torch.Tensor, kv_indices: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """
    Triton-only forward that computes the output and lse.
    Returns (output: [B, H, Dc] bfloat16, lse: [B, H] float32).
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert q_nope.is_cuda and q_pe.is_cuda and Kc_all.is_cuda and Kp_all.is_cuda, "Tensors must be on CUDA"

    B = q_nope.shape[0]
    H = q_nope.shape[1]
    Dc = q_nope.shape[2]
    Dp = q_pe.shape[2]
    # We assert the fixed sizes to match original model assumptions
    assert H == 16, "num_qo_heads must be 16"
    assert Dc == 512, "head_dim_ckv must be 512"
    assert Dp == 64, "head_dim_kpe must be 64"

    # Prepare output and lse
    out = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

    # Helper to compute L_tokens and tok_idx per batch
    def _compute_batch_tokens(b):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_beg >= page_end:
            # No KV tokens for this batch element
            return 0, torch.empty((0,), dtype=torch.long, device=Kc_all.device)
        return (page_end - page_beg), kv_indices[page_beg:page_end].to(torch.long)

    # Launch Triton kernel per (b, h)
    grid = (B, H)
    for b in range(B):
        L_tokens, tok_idx = _compute_batch_tokens(b)
        if L_tokens == 0:
            # Initialize outputs: output zeros, lse -inf (but we won't store since we skip kernel)
            out[b].zero_()
            lse[b] = float("-inf")
            continue

        # Gather Kc and Kp rows for this batch
        Kc = Kc_all[tok_idx]  # [L_tokens, Dc], contiguous
        Kp = Kp_all[tok_idx]  # [L_tokens, Dp], contiguous

        # Prepare qn and qp: they are [Dc] and [Dp] for this head
        # q_nope: [B, H, Dc] contiguous
        qn = q_nope[b].contiguous().to(torch.float32)  # [Dc]
        qp = q_pe[b].contiguous().to(torch.float32)   # [Dp]

        _compute_single_head[grid](
            qn_ptr=q_nope[b].contiguous().to(torch.float32),    # Triton expects pointers; passing float32 copies is fine
            qp_ptr=q_pe[b].contiguous().to(torch.float32),
            Kc_ptr=Kc.contiguous().to(torch.float32),
            Kp_ptr=Kp.contiguous().to(torch.float32),
            out_ptr=out,
            lse_ptr=lse,
            B=B, H=H, Dc=Dc, Dp=Dp,
            L_tokens=L_tokens,
            sm_scale=float(sm_scale),
            num_warps=4, num_stages=2,
        )

    return out, lse


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # If Triton not available, fall back to PyTorch (but evaluation requires Triton-only; here we assume CUDA + Triton).
        # Move tensors to CUDA if needed (the original run function assumes CUDA; here we require CUDA tensors).
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA for Triton execution"
        out, lse = _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)
        return out, lse


# Original reference model (not used by evaluator, but provided for context)
@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_beg >= page_end:
            output[b].zero_()
            continue

        L_tokens = page_end - page_beg
        tok_idx = kv_indices[page_beg:page_end].to(torch.long)

        Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]
        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)    # [num_qo_heads, head_dim_kpe]

        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, L_tokens]
        logits_scaled = logits * sm_scale

        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, L_tokens]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[b] = out.to(torch.bfloat16)

    return output, lse

def get_inputs():
    # Example inputs (the evaluator provides its own inputs; this is just a sample)
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return out if isinstance(out, (tuple, list)) else [out]


def run(*args):
    return ModelNew()(*args)
