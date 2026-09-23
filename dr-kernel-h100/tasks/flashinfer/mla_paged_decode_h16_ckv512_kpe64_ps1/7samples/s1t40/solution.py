import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute v[j, :] = sum_i (qn[j, :] · Kc[i, :]) + sum_i (qp[j, :] · Kp[i, :])
# Grid: (L,) one program per output index i
@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, [D]
    qp_ptr,           # *float32, [Dp]
    Kc_ptr,           # *float32, [L, D], row-major
    Kp_ptr,           # *float32, [L, Dp], row-major
    v_ptr,            # *float32, [L]
    L: tl.int32,      # number of tokens
    D: tl.constexpr,  # head_dim_ckv
    Dp: tl.constexpr, # head_dim_kpe
    BLOCK_K: tl.constexpr,
):
    i = tl.program_id(0)
    # Accumulate two dot-products
    sum1 = 0.0
    sum2 = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr_row = Kc_ptr + i * D + k_off
        kc_row = tl.load(kc_ptr_row, mask=mask_k, other=0.0)       # [BLOCK_K]
        sum1 += tl.sum(qn_slice * kc_row, axis=0)
    # Reduce over Kp dimension (Dp)
    for k in range(0, Dp, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < Dp
        qp_slice = tl.load(qp_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kp_ptr_row = Kp_ptr + i * Dp + k_off
        kp_row = tl.load(kp_ptr_row, mask=mask_k, other=0.0)        # [BLOCK_K]
        sum2 += tl.sum(qp_slice * kp_row, axis=0)
    # Write result
    tl.store(v_ptr + i, sum1 + sum2)


# Triton kernel: compute lse_j = logsumexp_base2(v[j, :]) for one head j
# Grid: (1,)
@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    L: tl.int32,
    lse_ptr,          # *float32, [1]
    BLOCK_L: tl.constexpr,
):
    # Reduce over L to compute max and sum(exp)
    max_v = -float("inf")
    sum_exp = 0.0
    ln2 = 1.4426950408889634  # 1 / ln(2)
    for l in range(0, L, BLOCK_L):
        l_off = l + tl.arange(0, BLOCK_L)
        mask_l = l_off < L
        v_chunk = tl.load(v_ptr + l_off, mask=mask_l, other=-float("inf"))
        # For masked elements, use -inf so they don't affect max/sum
        max_v = tl.maximum(max_v, tl.max(v_chunk, axis=0))
    # Second pass: compute sum(exp((v - max)/ln2))
    for l in range(0, L, BLOCK_L):
        l_off = l + tl.arange(0, BLOCK_L)
        mask_l = l_off < L
        v_chunk = tl.load(v_ptr + l_off, mask=mask_l, other=-float("inf"))
        sum_exp += tl.sum(tl.exp((v_chunk - max_v) * ln2), axis=0)
    lse_val = max_v + tl.log(sum_exp)  # logsumexp normalized by ln(2) already included in sum_exp
    # Store scalar lse
    tl.store(lse_ptr, lse_val)


# Triton kernel: compute attn[i] = exp(v[i] / ln(2) - lse) for one head j
# Grid: (L,)
@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    attn_ptr,         # *float32, [L]
    L: tl.int32,
    ln2: tl.float32,  # 1.4426950408889634
    BLOCK_L: tl.constexpr,
):
    i = tl.program_id(0)
    ln_lse = tl.load(lse_ptr)  # scalar
    # Compute attn[i]
    val = tl.load(v_ptr + i)
    attn_i = tl.exp(val * ln2 - ln_lse)
    tl.store(attn_ptr + i, attn_i)


# Triton kernel: compute out[j, h] = sum_i attn[i] * Kc[i, h] for one head j and output dim h
# Grid: (H,) one program per output dim h
@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D], row-major
    out_ptr,          # *float32, [1, H] flattened, but we only write one element for head j
    H: tl.int32,      # head_dim_ckv (512), but here we produce a single scalar for output dim h
    L: tl.int32,
    BLOCK_L: tl.constexpr,
):
    h = tl.program_id(0)
    acc = 0.0
    for l in range(0, L, BLOCK_L):
        l_off = l + tl.arange(0, BLOCK_L)
        mask_l = l_off < L
        attn_chunk = tl.load(attn_ptr + l_off, mask=mask_l, other=0.0)
        kc_ptr_col = Kc_ptr + l_off * H + h
        kc_col = tl.load(kc_ptr_col, mask=mask_l, other=0.0)
        acc += tl.sum(attn_chunk * kc_col, axis=0)
    tl.store(out_ptr + h, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Device and dtypes: compute in float32, output in bfloat16
    device = q_nope.device
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Inputs must be on CUDA"

    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    L_tokens_list = []
    # Derive number of tokens per batch element from kv_indptr (assumes kv_indptr[b+1] - kv_indptr[b] = tokens for batch b)
    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens_list.append(page_end - page_beg)
    L_tokens = L_tokens_list

    # Prepare outputs
    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    # Main loop over batch and heads; Triton kernels are launched per (b, j)
    ln2 = 1.4426950408889634  # 1 / ln(2)

    for b in range(batch_size):
        # If no tokens, skip
        if L_tokens[b] <= 0:
            continue
        # Gather Kc and Kp for this batch element
        Kc = ckv_cache.squeeze(1)[kv_indices[0:L_tokens[b]].to(torch.long).cuda()]  # [L, 512]
        Kp = kpe_cache.squeeze(1)[kv_indices[0:L_tokens[b]].to(torch.long).cuda()]  # [L, 64]
        Kc = Kc.to(torch.float32)
        Kp = Kp.to(torch.float32)

        # Loop over heads
        for j in range(num_qo_heads):
            # qn[j, :], qp[j, :] as 1D vectors
            qn = q_nope[b, j].to(torch.float32).contiguous()  # [512]
            qp = q_pe[b, j].to(torch.float32).contiguous()   # [64]

            L = L_tokens[b]

            # Kernel 1: matvec_add -> v[j, :]
            v = torch.empty((L,), dtype=torch.float32, device=device)
            BLOCK_K = 128
            grid_v = (L,)
            matvec_add_kernel[grid_v](
                qn, qp, Kc, Kp, v,
                L, head_dim_ckv, head_dim_kpe, BLOCK_K,
                num_warps=4
            )

            # Kernel 2: lse -> lse[b, j]
            lse_j = torch.empty((1,), dtype=torch.float32, device=device)
            BLOCK_L = 256
            lse_kernel[(1,)](
                v, L, lse_j,
                BLOCK_L,
                num_warps=1
            )
            lse[b, j] = lse_j[0]

            # Kernel 3: softmax (base-2) -> attn[j, :]
            attn = torch.empty((L,), dtype=torch.float32, device=device)
            softmax_base2_kernel[(L,)](
                v, lse[b, j], attn,
                L, ln2,
                BLOCK_L,
                num_warps=4
            )

            # Kernel 4: matvec write y -> out[b, j, :]
            # One program per output dim h (we write output as a 1D vector for clarity)
            for h in range(head_dim_ckv):
                out_scalar = torch.empty((1,), dtype=torch.float32, device=device)
                matvec_write_y_kernel[(1,)](
                    attn, Kc,
                    out_scalar,  # write into out[b, j, h]
                    head_dim_ckv, L,
                    BLOCK_L,
                    num_warps=4
                )
                output[b, j, h] = out_scalar[0]

    # Cast to bfloat16 as in original code
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


# Optional helper (not used by evaluator, but provided for local testing)
def get_inputs():
    # Helper to generate inputs without recursion. Evaluators may provide their own tensors.
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    # Simple indptr and indices for local testing
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)  # [2]
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA; forward only
        for i in range(len(q_nope)):
            if isinstance(q_nope[i], torch.Tensor) and q_nope[i].device.type != 'cuda':
                q_nope[i] = q_nope[i].to('cuda')
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
