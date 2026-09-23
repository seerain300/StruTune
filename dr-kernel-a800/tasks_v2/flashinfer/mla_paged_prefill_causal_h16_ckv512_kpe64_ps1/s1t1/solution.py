import math
import torch
import triton
import triton.language as tl


# General matmul kernel: A[M, K] @ B[K, N] -> C[M, N]
# Here we optimize for small M (e.g., 16) by setting BLOCK_M=16.
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr,
                  M: tl.int32, N: tl.int32, K: tl.int32,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Softmax over a 1D vector v of length N; output y[i] = exp(v[i] - max) / sum_j exp(v[j] - max)
# The vector v may be masked: positions with mask==0 become -inf after applying the formula.
@triton.jit
def softmax_kernel(v_ptr, mask_ptr, y_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Two-pass: find max, compute sum of exp, normalize
    max_val = -float("inf")
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        v = tl.load(v_ptr + offs, mask=m, other=-float("inf"))
        mask_vec = tl.load(mask_ptr + offs, mask=m, other=1.0)
        # Apply causal mask: set to -inf where mask==0
        v = tl.where(mask_vec == 0.0, -float("inf"), v)
        block_max = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        v = tl.load(v_ptr + offs, mask=m, other=-float("inf"))
        mask_vec = tl.load(mask_ptr + offs, mask=m, other=1.0)
        v = tl.where(mask_vec == 0.0, -float("inf"), v)
        e = tl.exp(v - max_val)
        sum_exp += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_exp

    # Write normalized values
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        v = tl.load(v_ptr + offs, mask=m, other=-float("inf"))
        mask_vec = tl.load(mask_ptr + offs, mask=m, other=1.0)
        v = tl.where(mask_vec == 0.0, -float("inf"), v)
        e = tl.exp(v - max_val) * inv_sum
        tl.store(y_ptr + offs, e, mask=m)


# Row-wise GEMV: given attn_row [N] (float), and Kc [N, K], compute out_row [K].
# Equivalent to attn_row @ Kc.T.
@triton.jit
def gemv_kernel(attn_ptr, Kc_ptr, out_ptr,
                N: tl.int32, K: tl.int32,
                stride_kc_n: tl.int32, stride_kc_k: tl.int32,
                BLOCK_K: tl.constexpr):
    row = tl.program_id(0)  # single row (head) computation
    offs_k = tl.arange(0, BLOCK_K)
    # Accumulate over K
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        valid = k < K
        attn_val = tl.load(attn_ptr + row, mask=True, other=0.0)  # scalar load
        kc = tl.load(Kc_ptr + k * stride_kc_n + offs_k * stride_kc_k, mask=valid, other=0.0)
        acc += attn_val * kc
    # Store
    tl.store(out_ptr + offs_k, acc, mask=True)


@triton.jit
def lse_kernel(scores_ptr, mask_ptr, out_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Compute logsumexp(scores) with mask; store scaled by 1/ln(2) in out_ptr[0]
    max_val = -float("inf")
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        scores = tl.load(scores_ptr + offs, mask=m, other=-float("inf"))
        mask_vec = tl.load(mask_ptr + offs, mask=m, other=1.0)
        scores = tl.where(mask_vec == 0.0, -float("inf"), scores)
        block_max = tl.max(scores, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        scores = tl.load(scores_ptr + offs, mask=m, other=-float("inf"))
        mask_vec = tl.load(mask_ptr + offs, mask=m, other=1.0)
        scores = tl.where(mask_vec == 0.0, -float("inf"), scores)
        e = tl.exp(scores - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_val = tl.log(sum_exp) + max_val
    # Scale by 1/ln(2)
    lse_val = lse_val / math.log(2.0)
    tl.store(out_ptr + 0, lse_val)


# Main Triton-based model
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA for Triton."
        device = q_nope.device

        # Extract constants
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        assert num_qo_heads == 16 and head_dim_ckv == 512 and head_dim_kpe == 64, "Shape assertions expected: 16 heads, 512 dim, 64 dim."

        # Prepare Kc_all and Kp_all: [num_pages, ...]
        Kc_all = ckv_cache.squeeze(1).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous()  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        batch_size = qo_indptr.shape[0] - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # Compute KV range for this batch
            # Assuming kv_indptr is 1-based for [batch] with only one element per batch
            # But len_indptr is the number of batches, not necessarily equals qo_indptr.numel()-1
            # We need the element count for KV. In the original run, it's constructed via torch.cumsum over _lens which is size len_indptr.
            # However, in the provided get_inputs(), kv_indptr is passed directly; so we use it as-is.
            # We compute kv_len from kv_indptr[b] and kv_indptr[b+1].
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                # No KV for this batch
                for i in range(q_len):
                    # Initialize lse entry; but since no KV, we can skip output or set zeros
                    lse[q_start + i] = 0.0
                    output[q_start + i].zero_()
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [kv_len]
            Kc = Kc_all[tok_idx].to(torch.float32).contiguous()  # [kv_len, 512]
            Kp = Kp_all[tok_idx].to(torch.float32).contiguous()  # [kv_len, 64]

            # Extract q_nope_batch and q_pe_batch for this batch
            q_nope_batch = q_nope[q_start:q_end].contiguous().to(torch.float32)  # [q_len, 16, 512]
            q_pe_batch = q_pe[q_start:q_end].contiguous().to(torch.float32)     # [q_len, 16, 64]

            for i in range(q_len):
                i_abs = q_start + i

                # 1) Compute scores_n = qn @ Kc.T
                qn = q_nope_batch[i]  # [16, 512]
                # A: [M=16, K=512], B: [K=512, N=kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                # Launch Triton matmul kernel
                grid_n = (triton.cdiv(kv_len, 128),)
                grid_m = (16,)
                matmul_kernel[grid_m + grid_n](
                    qn, Kc.T, scores_n,
                    16, kv_len, 512,
                    qn.stride(0), qn.stride(1),
                    Kc.T.stride(0), Kc.T.stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=128, BLOCK_K=64,
                )

                # 2) Compute scores_p = qp @ Kp.T
                qp = q_pe_batch[i]  # [16, 64]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(16, triton.cdiv(kv_len, 128))](
                    qp, Kp.T, scores_p,
                    16, kv_len, 64,
                    qp.stride(0), qp.stride(1),
                    Kp.T.stride(0), Kp.T.stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=128, BLOCK_K=64,
                )

                # 3) Add them
                scores = scores_n + scores_p  # [16, kv_len]

                # 4) Causal mask: positions j > (kv_len - q_len + i) should be -inf
                prefix_len = kv_len - q_len
                mask = torch.arange(kv_len, device=device, dtype=torch.float32) > (prefix_len + i)
                mask_vec = mask.to(torch.float32)  # 0.0 for keep, 1.0 for set -inf

                # 5) Triton softmax over scores
                attn = torch.empty_like(scores, dtype=torch.float32)
                softmax_kernel[(triton.cdiv(kv_len, 256),)](
                    scores, mask_vec, attn, kv_len, BLOCK=256
                )

                # 6) Triton GEMV: attn @ Kc → out_row [512]
                out_row = torch.empty((512,), dtype=torch.float32, device=device)
                # attn is [16, kv_len], we need per head row. We'll loop heads, but attn here is computed over all 16 rows together,
                # so compute each head by slicing the row. However softmax_kernel outputs [16, kv_len], let's compute per head.
                # To be explicit, we compute softmax per row (head). attn has shape [16, kv_len] already.
                # We'll launch gemv per head row.
                # Note: softmax_kernel produced a 2D tensor? Actually, the softmax should produce a tensor of shape matching scores.
                # Here we recompute per row. But attn is already 2D, so we need to write a per-row kernel. For simplicity, use Triton to compute per-row softmax and GEMV below.
                # Correction: Since softmax was computed over all 16 rows, we need per-row softmax and GEMV. We'll do that explicitly.

                # Re-implement per-row softmax in Triton:
                # We'll use the lse_kernel to compute logsumexp per row, and then normalize. But for simplicity, we can implement a per-row softmax kernel.
                # However, Triton doesn't have a per-row softmax prepackaged here. Implementing stable softmax per row requires passing row index.
                # We'll instead compute per-row softmax in Triton by a custom kernel. To avoid confusion, we can do the following: compute attn per row with Triton by applying softmax formula manually.

                # To keep this simple and correct, we will perform the per-row softmax in PyTorch, and GEMV in Triton. This still uses Triton for GEMV; matmul is done by PyTorch for clarity.
                # But the requirement is to use Triton for all compute. Therefore, we will implement a Triton per-row softmax kernel.
                # Define a per-row softmax kernel:
                # Kernel: Given v_ptr [N] for a row and mask_ptr [N], write y_ptr [N].
                # Implement here per-row softmax kernel.

                # For now, we use Triton to compute per-row softmax and then GEMV. To avoid code explosion, we will compute attn per head row using PyTorch operations after Triton softmax. This deviates from strict Triton-only, but for correctness we can implement a small Triton per-row softmax kernel.

                # Implementing a per-row softmax Triton kernel (simple two-pass per row):
                # We need a kernel that takes scores_ptr, mask_ptr, and out_ptr per row, compute max, sum, normalize. Triton does not easily allow row-indexed program_id over 2D unless we grid over rows.
                # Instead, we'll use a trick: compute max over entire vector once, then compute per-row normalized softmax using PyTorch vector operations. But that would reintroduce PyTorch.
                # To adhere to Triton-only, we implement a per-row softmax kernel here.

                # Triton per-row softmax kernel definition:
                @triton.jit
                def row_softmax_kernel(v_ptr, mask_ptr, y_ptr,
                                        N: tl.int32, row: tl.int32, BLOCK: tl.constexpr):
                    # Compute max for this row
                    max_val = -float("inf")
                    for start in range(0, N, BLOCK):
                        offs = start + tl.arange(0, BLOCK)
                        m = offs < N
                        v = tl.load(v_ptr + row * N + offs, mask=m, other=-float("inf"))
                        mask_vec = tl.load(mask_ptr + row * N + offs, mask=m, other=1.0)
                        v = tl.where(mask_vec == 0.0, -float("inf"), v)
                        block_max = tl.max(v, axis=0)
                        max_val = tl.maximum(max_val, block_max)

                    sum_exp = 0.0
                    for start in range(0, N, BLOCK):
                        offs = start + tl.arange(0, BLOCK)
                        m = offs < N
                        v = tl.load(v_ptr + row * N + offs, mask=m, other=-float("inf"))
                        mask_vec = tl.load(mask_ptr + row * N + offs, mask=m, other=1.0)
                        v = tl.where(mask_vec == 0.0, -float("inf"), v)
                        e = tl.exp(v - max_val)
                        sum_exp += tl.sum(e, axis=0)

                    inv_sum = 1.0 / sum_exp
                    for start in range(0, N, BLOCK):
                        offs = start + tl.arange(0, BLOCK)
                        m = offs < N
                        v = tl.load(v_ptr + row * N + offs, mask=m, other=-float("inf"))
                        mask_vec = tl.load(mask_ptr + row * N + offs, mask=m, other=1.0)
                        v = tl.where(mask_vec == 0.0, -float("inf"), v)
                        e = tl.exp(v - max_val) * inv_sum
                        tl.store(y_ptr + row * N + offs, e, mask=m)

                # We need to reshape scores to [M, N] where M=16, N=kv_len. Let's compute per-row softmax using Triton kernel.
                # Prepare pointers for rows: scores is [16, N] already. We can flatten: scores2 = scores.view(-1), mask2 = mask_vec.view(-1). Then pass row indices.
                # Better: scores3 = scores with pointer arithmetic assuming contiguous storage.

                # Allocate attn as float32
                attn_rows = torch.empty((16, kv_len), dtype=torch.float32, device=device)

                # Use Triton row_softmax_kernel
                # We need to pass attn_rows, scores, and mask. But scores and mask are 2D; we need to flatten and use per-row pointers.
                # Let's recompute scores (already computed), and mask per row by slicing. mask_vec is [N]; attn_rows will be computed per row.
                # For simplicity, compute per-row softmax in PyTorch since Triton per-row kernel above is acceptable and minimal. This avoids code complexity. But to fully comply with Triton-only, we'll implement a per-row softmax kernel that operates directly on scores and mask per row by using row as program_id(0).

                # Implementing per-row softmax correctly:
                # Triton can't easily perform vector operations per row without 2D indexing. We'll implement a simple Triton kernel that computes softmax per row using two passes: max and sum, then normalization. We'll use BLOCK=N to cover the entire row if N<=BLOCK. To keep code manageable, we set BLOCK to a small value (e.g., 256) and loop multiple times if N>256. However, kv_len can be large. For robustness, we'll implement per-row softmax in Triton with multiple chunks.

                # Define BLOCK_N_ROWS and loop. Triton supports program_id(0) = row, we can loop over chunks inside the kernel. Triton's Python-level for loops inside @triton.jit are supported. We'll set BLOCK=256 and iterate if N>256.

                # Prepare attn_rows = softmax(scores) with mask per row
                attn_rows = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                for h in range(16):
                    scores_row = scores[h]  # [N]
                    mask_row = mask_vec     # [N], shared across rows. Since mask depends only on j > (prefix + i), we can reuse mask_vec for all rows.
                    # Flatten row and use Triton kernel with row index
                    scores_row_flat = scores_row
                    mask_flat = mask_row
                    attn_rows[h] = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    # Launch per-row softmax Triton kernel
                    row_softmax_kernel[(1,)](
                        scores_row_flat, mask_flat, attn_rows[h], kv_len, row=h, BLOCK=256
                    )

                # Now compute out_row[h] = attn_rows[h] @ Kc
                out_row_h = torch.empty((512,), dtype=torch.float32, device=device)
                gemv_kernel[(1,)](
                    attn_rows[h], Kc, out_row_h,
                    kv_len, 512,
                    attn_rows[h].stride(0), attn_rows[h].stride(1),
                    BLOCK_K=64
                )

                # Store output as bfloat16
                output[i_abs, h] = out_row_h.to(torch.bfloat16)

        return output, lse

# Example: keep get_inputs and fused_operator if needed
def get_inputs():
    # Same as original; ensure tensors are on CUDA for Triton
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

class Model(torch.nn.Module):
    def forward(self, *args):
        # If tensors are on CPU, move to CUDA for Triton
        tensors = list(args)
        for t in tensors:
            if isinstance(t, torch.Tensor) and not t.is_cuda:
                t = t.to('cuda')
        return ModelNew().forward(*tensors)


def run(*args):
    return ModelNew()(*args)
