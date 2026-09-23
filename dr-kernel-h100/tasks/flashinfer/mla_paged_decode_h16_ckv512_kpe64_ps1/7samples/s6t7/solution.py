import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Compute logits vector for a single head h: logits = qn[h] @ Kc.T + qp[h] @ Kp.T
# Input:
#   qn_ptr: [Hc] float32 (per-head slice of q_nope)
#   qp_ptr: [Hp] float32 (per-head slice of q_pe)
#   Kc_ptr: [L, Hc] float32
#   Kp_ptr: [L, Hp] float32
#   out_ptr: [L] float32 (logits scaled by sm_scale)
#   sm_scale: float32
#   L: number of tokens (runtime)
#   Hc: head_dim_ckv (runtime)
#   Hp: head_dim_kpe (runtime)
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
    sm_scale, L: tl.int32, Hc: tl.int32, Hp: tl.int32,
    Kc_stride0: tl.int32, Kc_stride1: tl.int32,
    Kp_stride0: tl.int32, Kp_stride1: tl.int32,
    out_stride: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # Single program computes the entire logits vector for this row.
    acc = 0.0
    # Loop over tokens in chunks
    for k_start in range(0, L, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        mask = k_idx < L
        # Load qn[0:Hc] and qp[0:Hp] scalars
        qn_vals = tl.load(qn_ptr + tl.arange(0, Hc), mask=tl.arange(0, Hc) < Hc, other=0.0)
        qp_vals = tl.load(qp_ptr + tl.arange(0, Hp), mask=tl.arange(0, Hp) < Hp, other=0.0)
        # Compute partial sums: sum(Kc_chunk * qn_vals) and sum(Kp_chunk * qp_vals)
        # We do elementwise multiply across BLOCK_K and reduce across the vector.
        # Kc_ptr[k_idx, 0:Hc]: pointer grid for rows k_idx and columns 0..Hc-1
        # Load vectors of length Hc for each k in the chunk
        partial_qn = tl.zeros((), dtype=tl.float32)
        partial_qp = tl.zeros((), dtype=tl.float32)
        # Loop over Hc and Hp to build partial sums (simple accumulation without forming full matrices)
        for i in range(Hc):
            kc_vec = tl.load(Kc_ptr + k_idx * Kc_stride0 + i * Kc_stride1, mask=mask, other=0.0)
            partial_qn += kc_vec * qn_vals[i]
        for j in range(Hp):
            kp_vec = tl.load(Kp_ptr + k_idx * Kp_stride0 + j * Kp_stride1, mask=mask, other=0.0)
            partial_qp += kp_vec * qp_vals[j]
        acc += partial_qn + partial_qp
    # Scale and store
    acc = acc * sm_scale
    # out_ptr is [L]; we can store directly; Triton allows scalar store per loop iteration, but we
    # store acc into the corresponding position by using a loop to assign to each k in chunk.
    for k_start in range(0, L, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        mask = k_idx < L
        # Store acc for each k in chunk; acc is scalar, so we write it across mask
        tl.store(out_ptr + k_idx * out_stride, acc, mask=mask)


# Kernel 2: Row-wise softmax_logsumexp: compute lse for a single row (batch, head) over tokens
# Input:
#   logits_ptr: [L] float32
#   lse_ptr: [1] float32
#   L: number of tokens (runtime)
# Output:
#   lse_ptr[0] = logsumexp(logits) / log(2)
@triton.jit
def softmax_logsumexp_row_kernel(
    logits_ptr, lse_ptr, L: tl.int32, out_stride: tl.int32,
):
    # Compute max for numerical stability
    max_val = -1.0e30
    for i in range(0, L):
        val = tl.load(logits_ptr + i * out_stride)
        if val > max_val:
            max_val = val
    # Compute sum exp(x - max)
    sum_exp = 0.0
    for i in range(0, L):
        val = tl.load(logits_ptr + i * out_stride)
        sum_exp += tl.exp(val - max_val)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # log(2) inverse
    tl.store(lse_ptr, lse_val)


# Kernel 3: Compute out_row = attn_row @ Kc for a single head, output chunks over columns
# Input:
#   attn_ptr: [L] float32 (softmax of scaled logits for that head)
#   Kc_ptr: [L, Hc] float32
#   out_ptr: [Hc] float32
#   L: int, Hc: int
#   Kc_stride0: int, Kc_stride1: int
#   out_stride: int
# Launch grid: (ceil_div(Hc, BLOCK_N),)
@triton.jit
def matvec_row_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    L: tl.int32, Hc: tl.int32,
    Kc_stride0: tl.int32, Kc_stride1: tl.int32,
    out_stride: tl.int32,
    BLOCK_N: tl.constexpr,
):
    col_start = tl.program_id(0) * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    mask_cols = cols < Hc
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(0, L):
        attn_val = tl.load(attn_ptr + k)  # scalar softmax value for token k
        Kc_row = tl.load(Kc_ptr + k * Kc_stride0 + cols * Kc_stride1, mask=mask_cols, other=0.0)
        acc += attn_val * Kc_row
    tl.store(out_ptr + cols * out_stride, acc, mask=mask_cols)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Prepare Kc_all and Kp_all: squeeze segment dim and make contiguous
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, head_dim_kpe]

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Compute token indices range for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                # No KV cache for this batch element: output zeros and lse -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            L_tokens = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)

            # Gather Kc and Kp for these tokens: shapes [L_tokens, Hc] and [L_tokens, Hp]
            Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv], float32
            Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe], float32
            Kc = Kc.contiguous()
            Kp = Kp.contiguous()

            # Per-head slices of q_nope and q_pe (float32)
            for h in range(num_qo_heads):
                # 1) Compute logits_scaled for this head using Triton kernel
                qn = q_nope[b, h].contiguous().to(torch.float32)  # [Hc]
                qp = q_pe[b, h].contiguous().to(torch.float32)   # [Hp]
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                # Launch matmul_add_row_kernel: one program computes entire logits vector
                matmul_add_row_kernel[(1,)](
                    qn, qp, Kc, Kp, logits,
                    sm_scale, L_tokens, head_dim_ckv, head_dim_kpe,
                    Kc.stride(0), Kc.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    1,  # out_stride
                    BLOCK_K=64,  # chunk size for tokens
                )

                # 2) Compute lse for this head using Triton kernel
                lse[b, h] = torch.full((), -float("inf"), dtype=torch.float32, device=device)
                softmax_logsumexp_row_kernel[(1,)](
                    logits, lse[b], L_tokens, 1,
                )

                # 3) Compute output[b, h, :] = softmax(logits_scaled) @ Kc using Triton kernel
                # Compute attn = softmax(logits * sm_scale)
                attn = torch.softmax(logits * sm_scale, dim=0)  # [L_tokens], torch op (small vector)
                out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                matvec_row_kernel[(triton.cdiv(head_dim_ckv, 128),)](
                    attn, Kc, out_row,
                    L_tokens, head_dim_ckv,
                    Kc.stride(0), Kc.stride(1),
                    1,
                    BLOCK_N=128,
                )
                # Store to output[b, h, :]
                output[b, h] = out_row.to(torch.bfloat16)

        return output, lse


# Optional helpers for testing (not used by evaluator)
def get_inputs():
    device = 'cuda'
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device=device)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to(device)
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    return ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)


def run(*args):
    return ModelNew()(*args)
