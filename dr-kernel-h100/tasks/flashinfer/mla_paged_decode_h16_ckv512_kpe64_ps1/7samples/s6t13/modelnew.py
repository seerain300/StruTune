import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel A: Initialize qn and qp device vectors (float32), one per (b, h).
# Reads q_nope[b, h, :] and q_pe[b, h, :], casts to float32, stores into qn_out[0:Hc] and qp_out[0:Hp].
@triton.jit
def qnqp_init_kernel(
    qn_ptr,           # *float32, shape [Hc]
    qp_ptr,           # *float32, shape [Hp]
    qn_out_ptr,       # *float32, shape [Hc]
    qp_out_ptr,       # *float32, shape [Hp]
    Hc: tl.constexpr,
    Hp: tl.constexpr
):
    # Single program, linear indexing to write entire vectors
    # We don't have pid here; this kernel is called once per (b,h). We simply copy qn and qp to qn_out/qp_out.
    # The "for" loops are Triton-friendly when Hc/Hp are constexpr.
    for i in range(Hc):
        qni = tl.load(qn_ptr + i)
        tl.store(qn_out_ptr + i, qni)
    for j in range(Hp):
        qpj = tl.load(qp_ptr + j)
        tl.store(qp_out_ptr + j, qpj)


# Kernel B: Row-wise softmax + logsumexp for a single row (logits vector), for given qn and Kc/Kp.
# Computes:
#   m = max(logits), sum_exp = sum(exp(logits - m)), lse = log(sum_exp) / log(2), then writes attn[i] = exp(logits[i] - m) / sum_exp.
# Inputs:
#   qn_ptr: [Hc] float32 (per-head query)
#   Kc_ptr: [L, Hc] float32
#   Kp_ptr: [L, Hp] float32
#   lse_ptr: [1] float32 (scalar output)
#   attn_ptr: [L] float32 (output softmax)
#   L: int runtime (number of tokens)
#   Hc: constexpr
#   Hp: constexpr
#   Kc_stride0, Kc_stride1: strides for Kc
#   Kp_stride0, Kp_stride1: strides for Kp
# Pass 1: compute max
# Pass 2: compute sum_exp
# Pass 3: write attn
@triton.jit
def rowwise_softmax_logsumexp_kernel(
    qn_ptr,                # *float32, length Hc
    qp_ptr,                # *float32, length Hp
    Kc_ptr,                # *float32, shape [L, Hc]
    Kp_ptr,                # *float32, shape [L, Hp]
    lse_ptr,               # *float32, scalar
    attn_ptr,              # *float32, length L
    sm_scale,              # float32
    L: tl.constexpr,       # number of tokens (constexpr for loops)
    Hc: tl.constexpr,
    Hp: tl.constexpr,
    Kc_stride0, Kc_stride1,  # strides for Kc
    Kp_stride0, Kp_stride1,  # strides for Kp
):
    # Pass 1: compute max of (qn @ Kc.T + qp @ Kp.T)
    max_val = -float("inf")
    for t in range(L):
        acc = 0.0
        for j in range(0, Hc, 64):
            col_ids = j + tl.arange(0, 64)
            mask = col_ids < Hc
            kc = tl.load(Kc_ptr + t * Kc_stride0 + col_ids * Kc_stride1, mask=mask, other=0.0)
            qn_chunk = tl.load(qn_ptr + col_ids, mask=mask, other=0.0)
            acc += tl.sum(qn_chunk * kc, axis=0)
        for k in range(0, Hp, 32):
            col_ids_k = k + tl.arange(0, 32)
            mask_k = col_ids_k < Hp
            kp = tl.load(Kp_ptr + t * Kp_stride0 + col_ids_k * Kp_stride1, mask=mask_k, other=0.0)
            qp_chunk = tl.load(qp_ptr + col_ids_k, mask=mask_k, other=0.0)
            acc += tl.sum(qp_chunk * kp, axis=0)
        val = acc * sm_scale
        if val > max_val:
            max_val = val

    # Pass 2: compute sum of exp(logits - max_val)
    sum_exp = 0.0
    for t in range(L):
        acc = 0.0
        for j in range(0, Hc, 64):
            col_ids = j + tl.arange(0, 64)
            mask = col_ids < Hc
            kc = tl.load(Kc_ptr + t * Kc_stride0 + col_ids * Kc_stride1, mask=mask, other=0.0)
            qn_chunk = tl.load(qn_ptr + col_ids, mask=mask, other=0.0)
            acc += tl.sum(qn_chunk * kc, axis=0)
        for k in range(0, Hp, 32):
            col_ids_k = k + tl.arange(0, 32)
            mask_k = col_ids_k < Hp
            kp = tl.load(Kp_ptr + t * Kp_stride0 + col_ids_k * Kp_stride1, mask=mask_k, other=0.0)
            qp_chunk = tl.load(qp_ptr + col_ids_k, mask=mask_k, other=0.0)
            acc += tl.sum(qp_chunk * kp, axis=0)
        val = acc * sm_scale
        sum_exp += tl.exp(val - max_val)
    # Write lse = log(sum_exp) / log(2)
    lse = tl.log(sum_exp) * (1.0 / math.log(2.0))
    tl.store(lse_ptr, lse)

    # Pass 3: write attn = exp(val - max_val) / sum_exp
    inv_sum = 1.0 / sum_exp
    for t in range(L):
        acc = 0.0
        for j in range(0, Hc, 64):
            col_ids = j + tl.arange(0, 64)
            mask = col_ids < Hc
            kc = tl.load(Kc_ptr + t * Kc_stride0 + col_ids * Kc_stride1, mask=mask, other=0.0)
            qn_chunk = tl.load(qn_ptr + col_ids, mask=mask, other=0.0)
            acc += tl.sum(qn_chunk * kc, axis=0)
        for k in range(0, Hp, 32):
            col_ids_k = k + tl.arange(0, 32)
            mask_k = col_ids_k < Hp
            kp = tl.load(Kp_ptr + t * Kp_stride0 + col_ids_k * Kp_stride1, mask=mask_k, other=0.0)
            qp_chunk = tl.load(qp_ptr + col_ids_k, mask=mask_k, other=0.0)
            acc += tl.sum(qp_chunk * kp, axis=0)
        val = acc * sm_scale
        p = tl.exp(val - max_val) * inv_sum
        tl.store(attn_ptr + t, p)


# Kernel C: Gather matvec for output per head: out_row[h, :] = attn[:, h] @ Kc[:, :], where Kc is gathered per batch (tok_idx).
# We implement this by looping over tokens in chunks; for each output column j in [0..Hc), compute:
#   out_row[j] += attn[t] * Kc[t, j] for all t. This avoids building full matrices and keeps Triton-only math.
@triton.jit
def gather_matvec_kernel(
    attn_ptr,          # *float32, length L
    Kc_ptr,            # *float32, shape [L, Hc]
    out_ptr,           # *float32, length Hc
    Hc: tl.constexpr,
    L: tl.constexpr,
    Kc_stride0, Kc_stride1,
):
    # Single program writes out_row; we loop over output columns in chunks for parallelism
    BLOCK_N = 128
    for j in range(0, Hc, BLOCK_N):
        j_offsets = j + tl.arange(0, BLOCK_N)
        mask_j = j_offsets < Hc
        # Accumulator for this chunk
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # Loop over tokens
        for t in range(L):
            # Load attn[t] and Kc[t, j_offsets]
            at = tl.load(attn_ptr + t)
            kc_chunk = tl.load(Kc_ptr + t * Kc_stride0 + j_offsets * Kc_stride1, mask=mask_j, other=0.0)
            acc += at * kc_chunk
        # Store the chunk
        tl.store(out_ptr + j_offsets, acc, mask=mask_j)


# ModelNew: Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all compute in Triton

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA."
        B = q_nope.shape[0]
        Hc = q_nope.shape[2]
        Hp = q_pe.shape[2]
        num_qo_heads = q_nope.shape[1]
        # Ensure contiguity
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        # Prepare output and lse
        output = torch.empty((B, num_qo_heads, Hc), dtype=torch.float32, device=device)  # we will cast to bfloat16 at the end
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Constants for caching dims
        Hc_const = Hc
        Hp_const = Hp

        for b in range(B):
            # Compute token indices and lengths
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = int(page_end - page_beg)
            if L_tokens <= 0:
                # No tokens for this batch element; output zeros, lse -inf
                output[b] = torch.zeros((num_qo_heads, Hc), dtype=torch.float32)
                lse[b] = torch.full((num_qo_heads,), -float("inf"), dtype=torch.float32)
                continue

            # Gather Kc and Kp for this batch element
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # indices into cache
            Kc_all = ckv_cache.squeeze(1).to(torch.float32)          # [num_pages, Hc] -> [L_tokens, Hc] via gather below
            Kp_all = kpe_cache.squeeze(1).to(torch.float32)
            # Collect Kc and Kp per token
            Kc = torch.empty((L_tokens, Hc), dtype=torch.float32, device=device)
            Kp = torch.empty((L_tokens, Hp), dtype=torch.float32, device=device)
            for i, idx in enumerate(tok_idx):
                Kc[i] = Kc_all[idx]  # broadcasting over dim
                Kp[i] = Kp_all[idx]

            # For Triton kernels, we pass qn/qp and Kc/Kp. We will read qn and qp for each head in Triton.
            # We need qn and qp tensors to pass into kernels. We can create views or pass pointers to slices.
            # We will launch kernels per head h.

            for h in range(num_qo_heads):
                # 1) Initialize qn and qp buffers (float32)
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [Hc]
                qp = q_pe[b, h].to(torch.float32).contiguous()   # [Hp]
                qn_buf = torch.empty(Hc, dtype=torch.float32, device=device)
                qp_buf = torch.empty(Hp, dtype=torch.float32, device=device)

                # Launch qnqp_init_kernel: copies qn and qp to qn_buf/qp_buf
                qnqp_init_kernel[(1,)](
                    qn, qp, qn_buf, qp_buf,
                    Hc=Hc_const, Hp=Hp_const,
                    num_warps=1, num_stages=1
                )

                # 2) Compute lse and attn
                attn = torch.empty(L_tokens, dtype=torch.float32, device=device)
                lse_vec = torch.empty(1, dtype=torch.float32, device=device)
                rowwise_softmax_logsumexp_kernel[(1,)](
                    qn_buf, qp_buf, Kc, Kp, lse_vec, attn, sm_scale,
                    L=L_tokens, Hc=Hc_const, Hp=Hp_const,
                    Kc_stride0=Kc.stride(0), Kc_stride1=Kc.stride(1),
                    Kp_stride0=Kp.stride(0), Kp_stride1=Kp.stride(1),
                    num_warps=1, num_stages=1
                )
                # Store lse[b, h]
                lse[b, h] = lse_vec[0]

                # 3) Compute output[b, h, :] = attn @ Kc using Triton gather_matvec_kernel
                out_row = torch.empty(Hc, dtype=torch.float32, device=device)
                gather_matvec_kernel[(1,)](
                    attn, Kc, out_row,
                    Hc=Hc_const, L=L_tokens, Kc_stride0=Kc.stride(0), Kc_stride1=Kc.stride(1),
                    num_warps=4, num_stages=2
                )
                output[b, h, :] = out_row

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


# Helper to produce inputs on CUDA (for testing)
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
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    return ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)