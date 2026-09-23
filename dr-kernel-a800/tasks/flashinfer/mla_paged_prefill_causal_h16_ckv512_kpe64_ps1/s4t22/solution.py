import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_heads(q_nope, q_pe, Kc, Kp, logits_2d, q_len, L, H, SM_SCALE):
    # Grid: (q_len, H) programs; each program computes logits for one (i, h) over all L tokens
    i = tl.program_id(0)  # query position in this batch segment
    h = tl.program_id(1)  # head index
    # q_nope: [q_len, H, 512], q_pe: [q_len, H, 64]
    # logits_2d: [q_len, H, L], we index as row = i * H + h, column = l
    for l in range(0, L):
        # Load qn[h, :] and qp[h, :]
        # For contiguous [q_len, H, dim], offset = i * (H*dim) + h * dim
        dim_ckv = 512
        dim_kpe = 64
        offset_qn = i * (H * dim_ckv) + h * dim_ckv
        qn = tl.load(q_nope + offset_qn)  # [512]
        offset_qp = i * (H * dim_kpe) + h * dim_kpe
        qp = tl.load(q_pe + offset_qp)    # [64]

        # Load Kc[l, :] and Kp[l, :]
        kc = tl.load(Kc + l * dim_ckv + tl.arange(0, dim_ckv))  # [512]
        kp = tl.load(Kp + l * dim_kpe + tl.arange(0, dim_kpe))  # [64]

        # Compute dot products
        dot1 = tl.sum(qn * kc, axis=0)  # scalar
        dot2 = tl.sum(qp * kp, axis=0)  # scalar
        logit = dot1 + dot2  # logits[i, h, l]
        # Store into logits_2d at row (i*H + h), column l
        row = i * H + h
        tl.store(logits_2d + row * L + l, logit * SM_SCALE)


@triton.jit
def lse_and_attn_1d(logits_2d, attn_2d, lse, q_len, L, H, SM_SCALE):
    # Grid: (q_len, H) programs; each program handles one (i, h) and computes lse[i, h] and attn[i, h, :]
    i = tl.program_id(0)
    h = tl.program_id(1)

    # Compute query_abs_pos for causal mask
    prefix_len = L - q_len
    query_abs_pos = prefix_len + i

    # Load logits_scaled for this (i, h): vector of length L
    row = i * H + h
    vec = tl.load(logits_2d + row * L + tl.arange(0, L))  # [L]
    vec = vec * SM_SCALE

    # Apply causal mask: positions k <= query_abs_pos are -inf, others remain
    for k in range(0, L):
        if k <= query_abs_pos:
            vec[k] = -float("inf")

    # Numerical stability: logsumexp
    max_val = tl.max(vec, axis=0)
    vec_shift = vec - max_val
    exp_vec = tl.exp(vec_shift)
    sumexp = tl.sum(exp_vec, axis=0)
    lse_ih = tl.log(sumexp)  # natural log, then we convert to base-2 by dividing
    # Store lse[i, h]
    tl.store(lse + i * H + h, lse_ih)

    # Softmax: attn = exp(vec_shift) / sumexp
    inv_sum = 1.0 / sumexp
    for k in range(0, L):
        attn_val = exp_vec[k] * inv_sum
        tl.store(attn_2d + i * H * L + h * L + k, attn_val)


@triton.jit
def matmul_vec_by_mat(attn_2d, Kc, out_vec, q_len, H, L, head_dim_ckv, BLOCK_COL):
    # Grid: (q_len, H, ceil_div(head_dim_ckv, BLOCK_COL))
    i = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    col_block = tile * BLOCK_COL
    out_cols = col_block + tl.arange(0, BLOCK_COL)
    mask_out = out_cols < head_dim_ckv

    # Load attn vector for (i, h): length L
    row = i * H * L + h * L
    attn_vec = tl.load(attn_2d + row + tl.arange(0, L))
    # Compute out_vec[h, out_cols] = attn_vec @ Kc[:, out_cols]
    # We reduce over K_len = L in chunks of 1 (L is not tl.constexpr, so we loop)
    out_row = tl.zeros([BLOCK_COL], dtype=tl.float32)
    for k in range(0, L):
        kc = tl.load(Kc + k * head_dim_ckv + out_cols, mask=mask_out, other=0.0)  # [BLOCK_COL]
        out_row += attn_vec[k] * kc
    tl.store(out_vec + (i * H + h) * head_dim_ckv + out_cols, out_row, mask=mask_out)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    """
    Triton-only execution of the original logic. Assumes tensors are CUDA.
    Returns output [total_q, H, head_dim_ckv] bfloat16 and lse [total_q, H] float32.
    """
    device = q_nope.device
    total_q = q_nope.shape[0]
    H = q_nope.shape[1]
    assert H == 16, "num_qo_heads must be 16"
    head_dim_ckv = q_nope.shape[2]
    assert head_dim_ckv == 512, "head_dim_ckv must be 512"
    head_dim_kpe = q_pe.shape[2]
    assert head_dim_kpe == 64, "head_dim_kpe must be 64"

    # Cast inputs to float32 for compute
    q_nope = q_nope.contiguous().to(torch.float32)  # [total_q, H, 512]
    q_pe = q_pe.contiguous().to(torch.float32)      # [total_q, H, 64]
    ckv_cache = ckv_cache.contiguous().to(torch.float32)  # [num_pages, 1, 512] -> [num_pages, 512]
    kpe_cache = kpe_cache.contiguous().to(torch.float32)  # [num_pages, 1, 64]  -> [num_pages, 64]

    qo_indptr = qo_indptr.to(torch.int32).contiguous()
    kv_indptr = kv_indptr.to(torch.int32).contiguous()
    kv_indices = kv_indices.to(torch.int32).contiguous()

    batch_size = qo_indptr.numel() - 1
    output = torch.empty((total_q, H, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, H), dtype=torch.float32, device=device)
    attn_2d = torch.empty((total_q, H, 9999), dtype=torch.float32, device=device)  # upper bound for L, will slice

    for b in range(batch_size):
        # Compute query range
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        q_len = max(q_end - q_start, 0)

        # Compute KV token range
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L = max(page_end - page_beg, 0)
        if q_len == 0 or L == 0:
            continue

        # Gather tok_idx and form Kc/Kp
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L]
        Kc = ckv_cache[tok_idx]  # [L, 512]
        Kp = kpe_cache[tok_idx]  # [L, 64]

        # Buffer for logits_2d: [q_len, H, L]
        logits_2d = torch.empty((q_len, H * L), dtype=torch.float32, device=device)

        # Launch compute_logits_heads: grid over (q_len, H)
        grid_log = (q_len, H)
        compute_logits_heads[grid_log](
            q_nope[q_start:q_end], q_pe[q_start:q_end], Kc, Kp, logits_2d,
            q_len, L, H, sm_scale,
            num_warps=4, num_stages=2
        )

        # Launch lse_and_attn_1d: grid over (q_len, H); compute lse[i, h] and attn[i, h, :]
        grid_lse = (q_len, H)
        attn_2d_tmp = attn_2d  # already allocated large, we'll write up to L columns
        lse_batch = torch.empty((q_len, H), dtype=torch.float32, device=device)
        lse_and_attn_1d[grid_lse](
            logits_2d, attn_2d_tmp, lse_batch,
            q_len, L, H, sm_scale,
            num_warps=4, num_stages=2
        )
        # Merge per-batch lse into global lse: index offset q_start
        lse[q_start:q_start + q_len, :] = lse_batch

        # Launch matmul_vec_by_mat for each (i, h), tiled over head_dim_ckv
        BLOCK_COL = 128
        grid_mm = (q_len, H, triton.cdiv(head_dim_ckv, BLOCK_COL))
        out_vec = torch.empty((q_len, H, head_dim_ckv), dtype=torch.float32, device=device)
        for i in range(q_len):
            for h in range(H):
                matmul_vec_by_mat[grid_mm](
                    attn_2d_tmp[i * H + h * L:i * H + (h + 1) * L, :],  # slice [L]
                    Kc, out_vec[i, h],
                    q_len, H, L, head_dim_ckv, BLOCK_COL,
                    num_warps=4, num_stages=2
                )

        # Copy out_vec[i, h, :] into output[q_start+i, h, :]
        for i in range(q_len):
            for h in range(H):
                output[q_start + i, h] = out_vec[i, h].to(torch.bfloat16)

    return output, lse


# Entry point
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        if not (q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
                and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


# Helper for the harness (return CUDA tensors)
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
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
