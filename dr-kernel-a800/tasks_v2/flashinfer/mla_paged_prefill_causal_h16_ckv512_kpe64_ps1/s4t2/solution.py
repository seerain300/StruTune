import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_heads_3d(
    q_nope_ptr, q_pe_ptr, Kc_ptr, Kp_ptr,
    qn_out_ptr,              # [q_len, L] float32
    q_len: tl.constexpr,     # number of queries in this batch segment
    L: tl.constexpr,         # number of KV tokens (length of tokens)
    H: tl.constexpr,         # number of heads (16)
    head_dim_ckv: tl.constexpr,  # 512
    head_dim_kpe: tl.constexpr,  # 64
):
    # Grid: (h in [0..H-1], l in [0..L-1], i in [0..q_len-1])
    h = tl.program_id(0)
    l = tl.program_id(1)
    i = tl.program_id(2)

    acc = 0.0

    # Iterate over t in query positions
    for t in range(q_len):
        # Load qn[h, t] and qp[h, t]
        qn_ptr = q_nope_ptr + i * H * head_dim_ckv + h * head_dim_ckv + t
        qp_ptr = q_pe_ptr + i * H * head_dim_kpe + h * head_dim_kpe + t
        qn_val = tl.load(qn_ptr, mask=True, other=0.0)  # float32
        qp_val = tl.load(qp_ptr, mask=True, other=0.0)  # float32

        # Load Kc[t, l] and Kp[t, l]
        Kc_ptr_t = Kc_ptr + t * head_dim_ckv + l
        Kp_ptr_t = Kp_ptr + t * head_dim_kpe + l
        Kc_val = tl.load(Kc_ptr_t, mask=True, other=0.0)  # float32
        Kp_val = tl.load(Kp_ptr_t, mask=True, other=0.0)  # float32

        # Accumulate contributions
        acc += qn_val * Kc_val + qp_val * Kp_val

    # Store the accumulated logits for this (h, l, i)
    out_ptr = qn_out_ptr + i * L * H + h * L + l
    tl.store(out_ptr, acc)


@triton.jit
def lse_and_causal_mask_1d(
    logits_ptr,           # [q_len, L] float32
    lse_out_ptr,          # [q_len, 16] float32
    attn_ptr,             # [q_len, L, 16] float32 (we'll allocate and write here)
    q_len: tl.constexpr,  # number of queries in segment
    L: tl.constexpr,      # number of KV tokens
    H: tl.constexpr,      # number of heads (16)
    query_abs_pos: tl.constexpr,  # scalar int
    SM_SCALE: tl.constexpr,       # float
):
    # One program per (i, h) to compute lse and attn for that head and query
    i = tl.program_id(0)
    h = tl.program_id(1)

    # Load logits for this head h across L tokens: logits[i, :, h]
    logits_vec = tl.zeros((L,), dtype=tl.float32)
    for l in range(L):
        ptr = logits_ptr + i * L * H + h * L + l
        logits_vec[l] = tl.load(ptr)

    # Scale logits
    logits_scaled = logits_vec * SM_SCALE

    # Apply causal mask: positions k > query_abs_pos are -inf
    for l in range(L):
        if l > query_abs_pos:
            logits_scaled[l] = -float('inf')

    # Compute lse = logsumexp(logits_scaled) / log(2)
    max_val = tl.max(logits_scaled, axis=0)
    sum_exp = 0.0
    for l in range(L):
        sum_exp += tl.exp(logits_scaled[l] - max_val)
    lse = tl.log(sum_exp) / tl.log(2.0)

    # Store lse to lse_out[i, h]
    lse_out_ptr_ih = lse_out_ptr + i * H + h
    tl.store(lse_out_ptr_ih, lse)

    # Compute attn = softmax(logits_scaled)
    inv_sum = 1.0 / (sum_exp)
    for l in range(L):
        attn_val = tl.exp(logits_scaled[l] - max_val) * inv_sum
        attn_ptr_ilh = attn_ptr + i * L * H + h * L + l
        tl.store(attn_ptr_ilh, attn_val)


@triton.jit
def matmul_vec_by_mat_T_block(
    attn_ptr,        # [q_len, L] float32 (we store attn per head here as 2D)
    Kc_ptr,          # [L, head_dim_ckv] float32
    out_ptr,         # [q_len, head_dim_ckv] float32
    q_len: tl.constexpr,   # number of queries in segment
    L: tl.constexpr,       # number of KV tokens
    head_dim_ckv: tl.constexpr,  # 512
    BLOCK_COL: tl.constexpr,     # e.g., 128
):
    # One program per i, producing out[i, :] = attn[i, :] @ Kc.T
    i = tl.program_id(0)
    for col_start in range(0, head_dim_ckv, BLOCK_COL):
        acc = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for l in range(L):
            attn_val = tl.load(attn_ptr + i * L + l)  # scalar
            Kc_sub = tl.load(
                Kc_ptr + l * head_dim_ckv + col_start + tl.arange(0, BLOCK_COL),
                mask=tl.arange(0, BLOCK_COL) < (head_dim_ckv - col_start),
                other=0.0
            )
            acc[col_start:col_start + BLOCK_COL] += attn_val * Kc_sub
        tl.store(
            out_ptr + i * head_dim_ckv + col_start + tl.arange(0, BLOCK_COL),
            acc[col_start:col_start + BLOCK_COL],
            mask=tl.arange(0, BLOCK_COL) < (head_dim_ckv - col_start)
        )


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    # Ensure inputs are contiguous and in float32 for compute
    q_nope = q_nope.contiguous().to(torch.float32)
    q_pe = q_pe.contiguous().to(torch.float32)
    Kc_all = ckv_cache[:, 0, :].contiguous().to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache[:, 0, :].contiguous().to(torch.float32)  # [num_pages, 64]

    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16 and head_dim_ckv == 512 and head_dim_kpe == 64

    batch_size = qo_indptr.shape[0] - 1
    assert kv_indptr.shape[0] - 1 == batch_size, "kv_indptr batch size must match qo_indptr"

    output = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        if q_start >= q_end:
            continue

        # Read KV range
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_beg >= page_end:
            continue

        # Token indices for this batch
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()  # [L]
        L = tok_idx.shape[0]
        q_len = q_end - q_start

        # Load Kc and Kp for these token indices
        Kc = Kc_all[tok_idx].contiguous()  # [L, 512]
        Kp = Kp_all[tok_idx].contiguous()  # [L, 64]

        # Prepare q_nope and q_pe for this batch segment: shape [q_len, 16, 512/64]
        qn_segment = q_nope[q_start:q_end].contiguous()  # [q_len, 16, 512]
        qp_segment = q_pe[q_start:q_end].contiguous()   # [q_len, 16, 64]

        # Allocate buffers for logits, lse, attn
        logits = torch.empty((q_len, L), dtype=torch.float32, device=device)    # [q_len, L]
        attn = torch.empty((q_len, L, 16), dtype=torch.float32, device=device) # [q_len, L, 16]
        lse_buf = torch.empty((q_len, 16), dtype=torch.float32, device=device) # [q_len, 16]

        # Kernel 1: compute logits qn @ Kc.T + qp @ Kp.T -> [q_len, L]
        grid = (16, L, q_len)  # H, L, q_len
        compute_logits_heads_3d[grid](
            qn_segment, qp_segment, Kc, Kp,
            logits,
            q_len=q_len, L=L, H=16, head_dim_ckv=512, head_dim_kpe=64
        )

        # Kernel 2: compute lse and attn with causal mask per (i, h)
        grid_lse = (q_len, 16)
        query_abs_pos = (L - q_len)  # absolute position within cache sequence
        lse_and_causal_mask_1d[grid_lse](
            logits, lse_buf, attn, q_len=q_len, L=L, H=16, query_abs_pos=query_abs_pos, SM_SCALE=1.0
        )

        # Copy lse_buf into lse for this batch segment
        lse[q_start:q_end] = lse_buf

        # Kernel 3: out[i, :] = attn[i, :] @ Kc.T -> [q_len, 512]
        out_float = torch.empty((q_len, 512), dtype=torch.float32, device=device)
        grid_out = (q_len,)
        BLOCK_COL = 128
        matmul_vec_by_mat_T_block[grid_out](
            attn.view(q_len, L), Kc, out_float, q_len=q_len, L=L, head_dim_ckv=512, BLOCK_COL=BLOCK_COL
        )

        # Store output to [total_q, 16, 512] in bfloat16
        for i in range(q_len):
            output[q_start + i] = out_float[i].to(torch.bfloat16)

    return output, lse

class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA for Triton
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda \
               and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Inputs must be on CUDA device"
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)

# Helper for the harness (optional)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
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

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = _run_triton_only(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
