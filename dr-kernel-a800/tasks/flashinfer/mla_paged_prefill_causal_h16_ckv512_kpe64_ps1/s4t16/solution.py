import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_per_head(
    q_nope, q_pe, Kc_all, Kp_all,
    logits_buf,
    total_q, num_heads, head_dim_ckv, head_dim_kpe,
    H, qo_indptr, kv_indptr, kv_indices,
    sm_scale,
    B: tl.constexpr,
    Q_len: tl.constexpr,
    L: tl.constexpr,
):
    # program ids
    b = tl.program_id(0)  # batch index
    i = tl.program_id(1)  # query index within segment
    h = tl.program_id(2)  # head index

    # ranges
    q_start = tl.load(qo_indptr + b)
    q_end = tl.load(qo_indptr + b + 1)
    # guard
    if i >= (q_end - q_start):
        return

    # KV ranges for this batch
    tok_start = tl.load(kv_indptr + b)
    tok_end = tl.load(kv_indptr + b + 1)
    L_b = tok_end - tok_start

    # gather tok_idx
    tok_idx = kv_indices[tok_start:tok_end]  # [L_b]

    # load qn, qp for this query and head h
    # q_nope: [total_q, num_heads, head_dim_ckv]
    qn = tl.load(q_nope + (q_start + i) * num_heads * head_dim_ckv + h * head_dim_ckv)
    # q_pe: [total_q, num_heads, head_dim_kpe]
    qp = tl.load(q_pe + (q_start + i) * num_heads * head_dim_kpe + h * head_dim_kpe)

    # accumulate logits over L_b
    acc = tl.zeros([L_b], dtype=tl.float32)
    for t in range(L_b):
        Kc_t = tl.load(Kc_all + tok_idx[t] * head_dim_ckv)  # [head_dim_ckv]
        Kp_t = tl.load(Kp_all + tok_idx[t] * head_dim_kpe)  # [head_dim_kpe]
        # dot products
        dot1 = 0.0
        dot2 = 0.0
        # reduce over head_dim_ckv and head_dim_kpe
        for j in range(head_dim_ckv):
            dot1 += qn[j] * Kc_t[j]
        for j in range(head_dim_kpe):
            dot2 += qp[j] * Kp_t[j]
        acc[t] = dot1 + dot2

    acc = acc * sm_scale
    # write to logits_buf[b, i, h, :]
    base = b * Q_len * H * L + i * H * L + h * L
    tl.store(logits_buf + base, acc)


@triton.jit
def compute_lse_per_head(
    logits_buf, lse_buf,
    total_q, num_heads, L,
    B: tl.constexpr, Q_len: tl.constexpr, H: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)
    if i >= Q_len:
        return

    base = b * Q_len * H * L + i * H * L + h * L
    vec = tl.load(logits_buf + base)  # [L] float32
    # causal mask: k > (prefix_len + i), prefix_len = L - Q_len
    prefix_len = L - Q_len
    query_abs_pos = prefix_len + i
    mask = tl.arange(0, L) > query_abs_pos
    vec = tl.where(mask, -float("inf"), vec)
    m = tl.max(vec, axis=0)
    vec = vec - m
    sumexp = tl.sum(tl.exp(vec), axis=0)
    lse_val = tl.log(sumexp) / tl.log(2.0)  # float32
    # store lse[b, i, h]
    lse_offset = b * Q_len * H + i * H + h
    tl.store(lse_buf + lse_offset, lse_val)


@triton.jit
def compute_attention_per_head(
    logits_buf, lse_buf, attn_buf,
    total_q, num_heads, L,
    B: tl.constexpr, Q_len: tl.constexpr, H: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)
    if i >= Q_len:
        return

    base = b * Q_len * H * L + i * H * L + h * L
    vec = tl.load(logits_buf + base)  # [L] float32
    # read lse
    lse_val = tl.load(lse_buf + b * Q_len * H + i * H + h)  # float32
    vec_scaled = vec - lse_val
    exp_vec = tl.exp(vec_scaled)
    sumexp = tl.sum(exp_vec, axis=0)
    attn = exp_vec / sumexp  # [L] float32
    tl.store(attn_buf + b * Q_len * H * L + i * H * L + h * L, attn)


@triton.jit
def compute_out_per_head(
    attn_buf, Kc_all, output_buf,
    total_q, num_heads, head_dim_ckv, L,
    B: tl.constexpr, Q_len: tl.constexpr, H: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)
    if i >= Q_len:
        return

    base = b * Q_len * H * L + i * H * L + h * L
    attn = tl.load(attn_buf + base)  # [L] float32

    # Kc_all: [num_pages, head_dim_ckv] but we only need entries for tok_idx = kv_indices for this batch.
    # We'll reconstruct tok_idx from kv_indptr and kv_indices; here we need tok_idx for this batch segment.
    # Compute tok_idx range for this batch segment similarly to compute_logits kernel.
    tok_start = tl.load(kv_indptr + b)
    tok_end = tl.load(kv_indptr + b + 1)
    L_b = tok_end - tok_start
    tok_idx = kv_indices[tok_start:tok_end]  # [L_b]

    # output vector: [head_dim_ckv]
    out_vec = tl.zeros([head_dim_ckv], dtype=tl.float32)
    for j in range(0, head_dim_ckv, BLOCK_COL):
        cols = j + tl.arange(0, BLOCK_COL)
        col_mask = cols < head_dim_ckv
        # acc over tokens
        acc = tl.zeros([BLOCK_COL], dtype=tl.float32)
        for t in range(L_b):
            Kc_t = tl.load(Kc_all + tok_idx[t] * head_dim_ckv + cols, mask=col_mask, other=0.0)
            acc += attn[t] * Kc_t
        # store
        tl.store(output_buf + b * Q_len * H * head_dim_ckv + i * H * head_dim_ckv + h * head_dim_ckv + cols, acc, mask=col_mask)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    """
    Full Triton execution. Returns (output: [total_q, 16, 512] bfloat16, lse: [total_q, 16] float32).
    Assumes all inputs are CUDA tensors. Asserts num_heads=16, head_dim_ckv=512, head_dim_kpe=64.
    """
    assert q_nope.dim() == 3 and q_pe.dim() == 3
    assert q_nope.shape[1] == 16 and q_pe.shape[1] == 16
    assert q_nope.shape[2] == 512 and q_pe.shape[2] == 64
    assert ckv_cache.dim() == 3 and ckv_cache.shape[1] == 1 and ckv_cache.shape[2] == 512
    assert kpe_cache.dim() == 3 and kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == 64
    assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1
    total_q = int(qo_indptr[-1].item())
    H = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    B = int(qo_indptr.numel() - 1)
    device = q_nope.device

    # Prepare Kc_all, Kp_all
    Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)
    Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)

    # Allocate buffers
    # logits: [B, Q_len, H, L] float32
    # We need to determine Q_len for each batch; compute max Q_len across batches using qo_indptr
    # but Triton kernels need Q_len per batch. We'll allocate per-batch with q_end - q_start for each b.
    # Instead, allocate logits as [B, Q_max, H, L] where Q_max = qo_indptr[-1] - qo_indptr[0].
    # However, qo_indptr segments are disjoint; compute Q_len dynamically inside kernels by using
    # base on q_start. To simplify, we allocate as [B, Q_max, H, L] and write only for valid i.
    Q_max = total_q
    L_max = int(kv_indptr[-1].item()) - int(kv_indptr[0].item())  # max num tokens per batch
    # But to be precise, we can compute per-batch Q_len via q_end - q_start and L via tok_end - tok_start.
    # We will allocate logits with shape (B, Q_max, H, L_max). In kernels, we guard i<q_end-q_start and
    # compute L per batch dynamically by loading L for the batch. To keep simple, precompute L for each batch.

    # Compute L for each batch
    L_per_batch = []
    for b in range(B):
        tok_start = int(kv_indptr[b].item())
        tok_end = int(kv_indptr[b + 1].item())
        L_per_batch.append(tok_end - tok_start)
    # Allocate logits, attn, lse, output
    logits = torch.empty((B, Q_max, H, L_max), dtype=torch.float32, device=device)
    lse = torch.empty((B, Q_max, H), dtype=torch.float32, device=device)
    attn = torch.empty((B, Q_max, H, L_max), dtype=torch.float32, device=device)
    output = torch.empty((B, Q_max, H, head_dim_ckv), dtype=torch.bfloat16, device=device)

    # Launch compute_logits_per_head for all (b, i, h)
    grid_logits = (B, Q_max, H)
    compute_logits_per_head[grid_logits](
        q_nope, q_pe, Kc_all, Kp_all,
        logits,
        total_q, H, head_dim_ckv, head_dim_kpe,
        H, qo_indptr, kv_indptr, kv_indices,
        sm_scale,
        B=B, Q_len=Q_max, L=L_max,
    )

    # Now compute lse per (b, i, h)
    grid_lse = (B, Q_max, H)
    compute_lse_per_head[grid_lse](
        logits, lse,
        total_q, H, L_max,
        B=B, Q_len=Q_max, H=H,
    )

    # Compute attention per (b, i, h)
    grid_attn = (B, Q_max, H)
    compute_attention_per_head[grid_attn](
        logits, lse, attn,
        total_q, H, L_max,
        B=B, Q_len=Q_max, H=H,
    )

    # Compute output per (b, i, h)
    BLOCK_COL = 128
    grid_out = (B, Q_max, H)
    compute_out_per_head[grid_out](
        attn, Kc_all, output,
        total_q, H, head_dim_ckv, L_max,
        B=B, Q_len=Q_max, H=H,
        BLOCK_COL=BLOCK_COL,
    )

    # Finally, assemble output and lse into shapes [total_q, 16, 512] and [total_q, 16]
    # output is already structured as [B, i, h, :] but we need to place into global output[i, h, :]
    # We can copy per i: output[qo_indptr[b] + i, h, :] = output[b, i, h, :]
    final_output = torch.empty((total_q, H, head_dim_ckv), dtype=torch.bfloat16, device=device)
    final_lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

    for b in range(B):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        tok_start = int(kv_indptr[b].item())
        tok_end = int(kv_indptr[b + 1].item())
        L_b = tok_end - tok_start
        for i in range(q_end - q_start):
            # copy output[b, i, h, :] to final_output[q_start + i, h, :]
            for h in range(H):
                final_output[q_start + i, h] = output[b, i, h]

        # copy lse[b, i, h] to final_lse[q_start + i, h]
        for i in range(q_end - q_start):
            for h in range(H):
                final_lse[q_start + i, h] = lse[b, i, h]

    return final_output, final_lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA tensors
        if not (q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
                and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


# Helper for the harness
def get_inputs():
    # Create CUDA inputs
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    # qo_indptr and kv_indptr setup
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def run(*args):
    return ModelNew()(*args)
