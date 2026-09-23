import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_heads(q_nope, q_pe, Kc, Kp, logits, q_len, L, H, SM_SCALE):
    # Grid: (q_len, H) programs, each computes logits for one (i, h)
    i = tl.program_id(0)
    h = tl.program_id(1)
    # q_nope: [q_len, H, 512], q_pe: [q_len, H, 64], Kc: [L, 512], Kp: [L, 64], logits: [q_len, H, L]
    # We assume q_nope, q_pe are contiguous with shape [q_len, H, dim], and Kc/Kp are [L, dim]
    # For each token position l in 0..L-1:
    for l in range(0, L):
        # Extract qn[h, :] and qp[h, :]
        # For contiguous [q_len, H, dim], offset = i*H*dim + h*dim
        dim_ckv = 512
        offset_qn = i * H * dim_ckv + h * dim_ckv
        qn = tl.load(q_nope + offset_qn)
        # q_pe offset: same dim as q_nope for H and i
        offset_qp = i * H * 64 + h * 64
        qp = tl.load(q_pe + offset_qp)
        # Load Kc[l, :] and Kp[l, :]
        kc = tl.load(Kc + l * dim_ckv + tl.arange(0, dim_ckv))
        kp = tl.load(Kp + l * 64 + tl.arange(0, 64))
        # Compute dot products
        dot1 = tl.sum(qn * kc, axis=0)  # reduce over 512
        dot2 = tl.sum(qp * kp, axis=0)  # reduce over 64
        logit = dot1 + dot2
        # Store logits[i, h, l]
        tl.store(logits + i * H * L + h * L + l, logit * SM_SCALE)


@triton.jit
def lse_and_attn(logits, lse, q_len, L, H, SM_SCALE):
    # Grid: (q_len, H) programs, each computes lse and attn for one (i, h)
    i = tl.program_id(0)
    h = tl.program_id(1)
    # logits: [q_len, H, L]
    base = i * H * L + h * L
    # Compute max for numerical stability
    max_logit = -float("inf")
    for l in range(0, L):
        val = tl.load(logits + base + l) / SM_SCALE
        max_logit = tl.maximum(max_logit, val)
    # Apply causal mask: allow positions l > (L - q_len + i)
    prefix_len = L - q_len
    query_abs_pos = prefix_len + i
    sumexp = 0.0
    for l in range(0, L):
        val = tl.load(logits + base + l) / SM_SCALE
        if l > query_abs_pos:
            sumexp += tl.exp(val - max_logit)
    lse[i, h] = tl.log(sumexp) / 0.6931471805599453  # log(2)

    # Compute attn = softmax(logits_scaled)
    for l in range(0, L):
        val = tl.load(logits + base + l)
        # Skip masked positions by setting to 0 (they won't be stored since attn zeros them)
        attn_l = tl.exp((val * SM_SCALE) - lse[i, h])
        tl.store(attn + i * H * L + h * L + l, attn_l)


@triton.jit
def matmul_vec_by_mat(attn, Kc, out, q_len, H, L, head_dim_ckv, BLOCK_COL):
    # Grid: (q_len, H, ceil_div(head_dim_ckv, BLOCK_COL))
    i = tl.program_id(0)
    h = tl.program_id(1)
    col_block = tl.program_id(2)
    # attn: [q_len, H, L], Kc: [L, head_dim_ckv], out: [q_len, H, head_dim_ckv]
    out_row = tl.zeros((head_dim_ckv,), dtype=tl.float32)
    for k in range(0, L):
        attn_k = tl.load(attn + i * H * L + h * L + k)  # scalar
        kc_tile = tl.load(Kc + k * head_dim_ckv + tl.arange(0, head_dim_ckv))  # [head_dim_ckv]
        out_row += attn_k * kc_tile
    # Store out[i, h, :]
    out_cols = col_block * BLOCK_COL + tl.arange(0, BLOCK_COL)
    mask = out_cols < head_dim_ckv
    tl.store(out + (i * H + h) * head_dim_ckv + out_cols, out_row[out_cols], mask=mask)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    """
    Triton-only execution of the original logic. Assumes tensors are CUDA.
    Returns output [total_q, H, 512] bfloat16 and lse [total_q, H] float32.
    """
    # Ensure device and contiguity
    device = q_nope.device
    total_q = q_nope.shape[0]
    H = q_nope.shape[1]
    assert H == 16
    head_dim_ckv = q_nope.shape[2]
    assert head_dim_ckv == 512
    head_dim_kpe = q_pe.shape[2]
    assert head_dim_kpe == 64

    # Compute L and batch size from index tensors
    batch_size = qo_indptr.numel() - 1

    # Output buffers
    output = torch.empty((total_q, H, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

    # Precompute Kc and Kp for each batch b using indices
    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        q_len = q_end - q_start
        if q_len <= 0:
            continue

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L = page_end - page_beg
        if L <= 0:
            continue

        # Gather tok_idx and corresponding rows from cache (squeezed 1st dim)
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L]
        Kc = ckv_cache[tok_idx, 0, :].contiguous().to(torch.float32)  # [L, 512]
        Kp = kpe_cache[tok_idx, 0, :].contiguous().to(torch.float32)  # [L, 64]

        # Allocate logits buffer [q_len, H, L] float32
        logits = torch.empty((q_len, H, L), dtype=torch.float32, device=device)

        # Launch compute_logits_heads: grid (q_len, H)
        grid_log = (q_len, H)
        compute_logits_heads[grid_log](
            q_nope[q_start:q_end], q_pe[q_start:q_end], Kc, Kp, logits,
            q_len, L, H, sm_scale,
            num_warps=4,
        )

        # Launch lse_and_attn: grid (q_len, H), compute lse and attn
        grid_lse = (q_len, H)
        attn = torch.empty((q_len, H, L), dtype=torch.float32, device=device)
        lse_and_attn[grid_lse](
            logits, lse[q_start:q_start + q_len], q_len, L, H, sm_scale,
            num_warps=4,
        )

        # Merge attn into global attn for final matmul (use upper triangular positions only)
        # Here we just process this batch's attn and output per i, h.

        # Launch matmul_vec_by_mat: grid (q_len, H, ceil_div(head_dim_ckv, BLOCK_COL))
        BLOCK_COL = 128
        grid_out = (q_len, H, triton.cdiv(head_dim_ckv, BLOCK_COL))
        out_b = torch.empty((q_len, H, head_dim_ckv), dtype=torch.float32, device=device)
        matmul_vec_by_mat[grid_out](
            attn, Kc, out_b,
            q_len, H, L, head_dim_ckv, BLOCK_COL,
            num_warps=4,
        )

        # Copy per-batch output slice to global output at [q_start:q_end]
        for i in range(q_len):
            out_row = output[q_start + i]  # view [H, head_dim_ckv]
            out_b_i = out_b[i]  # [H, head_dim_ckv]
            out_row.copy_(out_b_i.to(torch.bfloat16))

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        if not (q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
                and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


# Helper function to generate inputs (CUDA)
def get_inputs():
    device = 'cuda'
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device=device)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device=device)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device=device)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32, device=device)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
