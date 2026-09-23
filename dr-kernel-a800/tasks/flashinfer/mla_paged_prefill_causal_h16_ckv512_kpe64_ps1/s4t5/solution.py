import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_heads_3d(
    qn_ptr,       # *float32, shape [H, D]
    qp_ptr,       # *float32, shape [H, Kp_dim]
    Kc_ptr,       # *float32, shape [L, D]
    Kp_ptr,       # *float32, shape [L, Kp_dim]
    out_ptr,      # *float32, shape [H, L]
    H: tl.constexpr,           # number of heads (16)
    D: tl.constexpr,           # head_dim_ckv (512)
    Kp_dim: tl.constexpr,      # head_dim_kpe (64)
    L,                        # number of KV tokens (runtime)
    BLOCK_K: tl.constexpr,     # tile over L
):
    # 2D grid: (H, ceil_div(L, BLOCK_K))
    h = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < L

    # Accumulator for logits for this head h
    logits = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Load qn[h, :] and qp[h, :]
    qn = tl.load(qn_ptr + h * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # [D]
    qp = tl.load(qp_ptr + h * Kp_dim + tl.arange(0, Kp_dim), mask=tl.arange(0, Kp_dim) < Kp_dim, other=0.0)  # [Kp_dim]

    # Accumulate dot-products over D and Kp_dim for each l in tile
    # We use a simple loop over k (L) and d (D), with tiling over l
    # For each l in offs:
    #   logits[l] += sum_d qn[d] * Kc[l, d] + sum_k qp[k] * Kp[l, k]
    # Note: Kc_ptr and Kp_ptr are row-major [L, D] and [L, Kp_dim]
    # We index by l (offs) and d or k.
    for l_idx in range(0, L):
        # Skip if l_idx not in offs (mask), but we can still load; however, better to mask loads by tile membership.
        # Instead, we loop over l from 0 to L-1 and use mask to avoid out-of-range issues by just loading when l in tile.
        # But since we iterate l_idx, we need to ensure we only accumulate when l_idx is in offs. We use masked load here.
        # However Triton for-loops expect static bounds; better to unroll small D via static range and dynamic l via masks.
        # To keep it simple and correct, we compute pointer for each l and mask with offs. Triton allows scalar indexing per iteration.
        pass
    # The above placeholder loop structure is not ideal. Triton prefers explicit tiling. We will compute with pointer arithmetic:
    # We need to implement two reductions: over D and over Kp_dim. We'll do outer-product accumulation.
    # We'll vectorize over l within a tile and reduce over D and Kp_dim using nested loops.
    # For vectorized l, we load Kc rows and Kp rows for each l in tile and reduce with qn and qp.

    # Prepare to accumulate per l in tile
    # We can't vectorize reductions across K easily without block pointers, so we do per-l accumulation with scalar l loop.
    for l_idx in range(0, L):
        # For each l, compute contributions
        # Kc[l, :] and Kp[l, :]
        Kc_row = tl.load(Kc_ptr + l_idx * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # [D]
        Kp_row = tl.load(Kp_ptr + l_idx * Kp_dim + tl.arange(0, Kp_dim), mask=tl.arange(0, Kp_dim) < Kp_dim, other=0.0)  # [Kp_dim]

        # Dot with qn over D
        # Cast qn to float32 for reduction if needed
        # Use tl.sum of elementwise product over D
        # We can't directly tl.sum over axis, so we loop over d
        acc1 = 0.0
        for d in range(0, D):
            acc1 += qn[d] * Kc_row[d]
        acc2 = 0.0
        for k in range(0, Kp_dim):
            acc2 += qp[k] * Kp_row[k]
        logits[l_idx] = acc1 + acc2

    # Store results for this tile
    out_row_ptr = out_ptr + h * L + offs
    tl.store(out_row_ptr, logits, mask=mask)


@triton.jit
def matmul_vec_by_mat_vec_row(
    attn_ptr,   # *float32, shape [H, L] (row-major, since H small here)
    K_ptr,      # *float32, shape [L, D]
    out_ptr,    # *float32, shape [H, D]
    H,          # number of heads
    L,          # number of tokens
    D: tl.constexpr,        # 512
    BLOCK_COL: tl.constexpr # tile size for output columns
):
    # Grid: (H, ceil_div(D, BLOCK_COL))
    h = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_COL + tl.arange(0, BLOCK_COL)
    mask = offs < D

    # Load attn row h
    attn_row = tl.load(attn_ptr + h * L + tl.arange(0, L), mask=tl.arange(0, L) < L, other=0.0)  # [L]

    # Accumulator for out[h, offs]
    acc = tl.zeros([BLOCK_COL], dtype=tl.float32)

    # Compute out[h, offs] = sum_{k=0..L-1} attn_row[k] * K[k, offs]
    # K is [L, D], row-major, so K[k, offs] = K_ptr + k*D + offs
    for k in range(0, L):
        Kk = tl.load(K_ptr + k * D + offs, mask=mask, other=0.0)  # [BLOCK_COL]
        acc += attn_row[k] * Kk

    tl.store(out_ptr + h * D + offs, acc, mask=mask)


@triton.jit
def matmul_mat_vec_col(
    K_ptr,      # *float32, shape [L, D]
    out_ptr,    # *float32, shape [L]
    L,          # number of tokens
    D: tl.constexpr,        # 512
    SM_SCALE: tl.constexpr  # float scale
):
    # Grid: 1 program (we loop over D inside)
    # Compute out[l] = SM_SCALE * sum_d K[l, d] for l in 0..L-1
    # This is a simple reduction per l. We'll compute per-l output.
    for l in range(0, L):
        acc = 0.0
        for d in range(0, D):
            acc += tl.load(K_ptr + l * D + d)
        tl.store(out_ptr + l, acc * SM_SCALE)


@triton.jit
def matmul_vec_by_mat_full(
    attn_ptr,   # *float32, shape [H, L]
    K_ptr,      # *float32, shape [L, D]
    out_ptr,    # *float32, shape [H, D]
    H,          # number of heads
    L,          # number of tokens
    D: tl.constexpr,        # 512
    BLOCK_COL: tl.constexpr # tile size for output columns
):
    # Grid: (H, ceil_div(D, BLOCK_COL))
    h = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_COL + tl.arange(0, BLOCK_COL)
    mask = offs < D

    # Load attn row h: attn[h, :] = attn_ptr + h*L + tl.arange(0, L)
    attn_row = tl.load(attn_ptr + h * L + tl.arange(0, L), mask=tl.arange(0, L) < L, other=0.0)  # [L]

    # Accumulator for out[h, offs]
    acc = tl.zeros([BLOCK_COL], dtype=tl.float32)

    # Compute out[h, offs] = sum_{k=0..L-1} attn_row[k] * K[k, offs]
    # K is [L, D], row-major
    for k in range(0, L):
        Kk = tl.load(K_ptr + k * D + offs, mask=mask, other=0.0)  # [BLOCK_COL]
        acc += attn_row[k] * Kk

    tl.store(out_ptr + h * D + offs, acc, mask=mask)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    total_q, H, D = q_nope.shape
    assert H == 16, "num_qo_heads must be 16"
    assert D == 512, "head_dim_ckv must be 512"
    Kp_dim = q_pe.shape[-1]
    assert Kp_dim == 64, "head_dim_kpe must be 64"

    # Prepare Kc_all and Kp_all
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

    # Output and lse
    output = torch.zeros((total_q, H, D), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)

    # Number of batches inferred from qo_indptr
    batch_count = qo_indptr.numel() - 1

    for b in range(batch_count):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        # If no queries for this batch, skip
        if q_start >= q_end:
            continue

        # KV ranges
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_beg >= page_end:
            continue

        # Tokens used for this batch
        tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [L]
        L = tok_idx.numel()

        # Extract K matrices for this batch
        Kc = Kc_all[tok_idx]  # [L, D]
        Kp = Kp_all[tok_idx]  # [L, Kp_dim]

        # Prepare q_nope and q_pe for this batch
        q_nope_b = q_nope[q_start:q_end].to(torch.float32)  # [q_len, H, D]
        q_pe_b = q_pe[q_start:q_end].to(torch.float32)      # [q_len, H, Kp_dim]
        q_len = q_end - q_start

        # For each query i in this batch
        for i in range(q_len):
            # Current query vectors per head
            qn = q_nope_b[i]  # [H, D]
            qp = q_pe_b[i]    # [H, Kp_dim]

            # Allocate logits for this batch
            logits = torch.empty((H, L), dtype=torch.float32, device=device)

            # Launch Triton kernel to compute logits[h, :] = qn[h] @ Kc.T + qp[h] @ Kp.T
            BLOCK_L = 64
            grid = (H, triton.cdiv(L, BLOCK_L))
            compute_logits_heads_3d[grid](
                qn, qp, Kc, Kp, logits,
                H=H, D=D, Kp_dim=Kp_dim, L=L, BLOCK_K=BLOCK_L,
                num_warps=4, num_stages=2
            )

            # Now compute lse and attn using torch (to preserve correctness and avoid Triton mask complexity)
            # Scale and causal mask: absolute position
            prefix_len = L - q_len
            query_abs_pos = prefix_len + i
            logits_scaled = logits * sm_scale
            # Causal mask: allow only positions k > query_abs_pos
            # mask = (torch.arange(L, device=device) > query_abs_pos)
            mask = torch.arange(L, device=device, dtype=torch.int32) > query_abs_pos
            logits_scaled = logits_scaled.masked_fill(~mask, float("-inf"))
            # Compute lse per head
            # lse[h] = logsumexp(logits_scaled[h, :]) / log(2)
            # Use torch for reductions
            m = logits_scaled.max(dim=1, keepdim=True).values
            sumexp = logits_scaled - m
            sumexp = sumexp.exp().sum(dim=1, keepdim=True)
            lse[q_start + i] = torch.log(sumexp) / math.log(2.0)
            # attention weights
            attn = logits_scaled - lse[q_start + i]  # broadcast subtract
            attn = attn.exp() / attn.sum(dim=1, keepdim=True)

            # Compute output vector per head: out[h, :] = attn[h, :] @ Kc.T
            # Use Triton matmul_vec_by_mat to write to output
            BLOCK_COL = 128
            grid_out = (H, triton.cdiv(D, BLOCK_COL))
            # out[h, :] already zeroed; we will compute into out[q_start+i, h, :]
            # For simplicity, compute attn[h, :] @ Kc.T using Triton by tiling columns.
            # We need to pass attn for each h. Build per-h attn vector and invoke kernel.
            for h in range(H):
                attn_h = attn[h]  # [L]
                out_row = torch.empty((D,), dtype=torch.float32, device=device)
                matmul_vec_by_mat_vec_row[grid_out](
                    attn_h, Kc, out_row,
                    H=1, L=L, D=D, BLOCK_COL=BLOCK_COL,
                    num_warps=4, num_stages=2
                )
                output[q_start + i, h] = out_row.to(torch.bfloat16)

    return output, lse


# Entry point module
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        if not (q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
                and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


# Example helper (for local testing)
def get_inputs():
    # Create CUDA tensors
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    # Simulate qo_indptr and kv_indptr (as in original)
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


# Optional: fused operator interface
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
