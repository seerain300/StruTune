import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_query_kernel(
    # Input pointers (original dtype)
    q_nope_ptr,       # *bf16, shape [Q_total, 16, 512], row-major
    q_pe_ptr,         # *bf16, shape [Q_total, 16, 64], row-major
    Kc_sel_ptr,       # *bf16, shape [kv_len, 512], row-major (selected tokens)
    Kp_sel_ptr,       # *bf16, shape [kv_len, 64], row-major (selected tokens)
    output_ptr,       # *bf16, shape [Q_total, 16, 512], row-major
    lse_ptr,          # *fp32, shape [Q_total, 16]
    # runtime scalar
    q_start,          # int32, start query index in global q_nope
    # constexpr meta-parameters
    q_len: tl.constexpr,    # number of queries in this batch element (compile-time for this program)
    kv_len: tl.constexpr,   # number of selected KV tokens (compile-time for this program)
    sm_scale: tl.constexpr,         # fp32 scaling factor
    ln2_inv: tl.constexpr,          # fp32 = 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,        # 16
    HEAD_DIM_CKV: tl.constexpr,     # 512
    HEAD_DIM_KPE: tl.constexpr,     # 64
):
    # This program handles one specific query i within a batch element.
    # The grid is (1, q_len); b is implicit via q_start and i is the second grid dimension.

    # Determine query absolute index
    q_abs = q_start  # Triton scalar; we can add i at runtime. However Triton expects scalar args.
    # Triton allows only compile-time loops; we pass i via program_id(1). Triton provides no direct access to loop variable; use program_id(1) as i.
    # To avoid dynamic indexing confusion, compute i from program_id(1):
    # Note: Triton provides program_id(0), program_id(1). We use program_id(1) as i.
    i = tl.program_id(1)

    q_abs = q_start + i

    # Load qn[h, :] and qp[h, :] for all heads h, convert to fp32
    qn = tl.zeros((NUM_HEADS, HEAD_DIM_CKV), dtype=tl.float32)
    qp = tl.zeros((NUM_HEADS, HEAD_DIM_KPE), dtype=tl.float32)

    for h in range(NUM_HEADS):
        k_vec = tl.arange(0, HEAD_DIM_CKV)
        offset_qn = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV + k_vec
        # Load original dtype (bf16), cast to float32
        qn[h, :] = tl.load(q_nope_ptr + offset_qn, mask=k_vec < HEAD_DIM_CKV, other=0.0).to(tl.float32)

        kpe_vec = tl.arange(0, HEAD_DIM_KPE)
        offset_qp = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE + kpe_vec
        qp[h, :] = tl.load(q_pe_ptr + offset_qp, mask=kpe_vec < HEAD_DIM_KPE, other=0.0).to(tl.float32)

    # Compute logits per head: [NUM_HEADS, kv_len]
    logits = tl.zeros((NUM_HEADS, kv_len), dtype=tl.float32)

    for j in range(kv_len):
        # Load Kc_sel[j, :] and Kp_sel[j, :]
        k_vec = tl.arange(0, HEAD_DIM_CKV)
        Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0).to(tl.float32)
        dot_qn = tl.sum(qn * Kc_j[None, :], axis=1)  # [NUM_HEADS]

        kpe_vec = tl.arange(0, HEAD_DIM_KPE)
        Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + kpe_vec, mask=kpe_vec < HEAD_DIM_KPE, other=0.0).to(tl.float32)
        dot_qp = tl.sum(qp * Kp_j[None, :], axis=1)  # [NUM_HEADS]

        logits[:, j] = dot_qn + dot_qp

    # Scale
    logits = logits * sm_scale

    # Causal mask: valid j >= prefix_len + i + 1, where prefix_len = kv_len - q_len
    prefix_len = kv_len - q_len
    valid_start = prefix_len + i + 1
    j_vec = tl.arange(0, kv_len)
    causal_mask = j_vec >= valid_start
    logits = tl.where(causal_mask, -float("inf"), logits)

    # logsumexp in log2
    m = tl.max(logits, axis=1)  # [NUM_HEADS]
    sumexp = tl.sum(tl.exp(logits - m[:, None]), axis=1)
    lse_val = m + tl.log(sumexp) * ln2_inv  # [NUM_HEADS]
    for h in range(NUM_HEADS):
        tl.store(lse_ptr + q_abs * NUM_HEADS + h, lse_val[h])

    # Softmax (stable)
    logits = logits - m[:, None]
    exp_logits = tl.exp(logits)
    denom = tl.sum(exp_logits, axis=1)[:, None]  # [NUM_HEADS, 1]
    softmax = exp_logits / denom  # [NUM_HEADS, kv_len]

    # Output: out[h, :] = sum_j softmax[h,j] * Kc_sel[j, :]
    out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
    for h in range(NUM_HEADS):
        for j in range(kv_len):
            k_vec = tl.arange(0, HEAD_DIM_CKV)
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            out_vec += softmax[h, j] * Kc_j
        # Store as bf16
        # Triton will write as bf16 if output_ptr is bf16; cast explicitly
        tl.store(output_ptr + (q_abs * NUM_HEADS + h) * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), out_vec.to(tl.bfloat16))

# Entry point: ModelNew.forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Assertions
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16"
        assert q_nope.shape[2] == 512, "head_dim_ckv must be 512"
        assert q_pe.shape[2] == 64, "head_dim_kpe must be 64"
        device = q_nope.device
        total_q = int(qo_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        assert len_indptr >= 2, "len_indptr must be >= 2"
        batch_size = len_indptr - 1

        # Allocate outputs
        output = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        ln2_inv = 1.0 / math.log(2.0)

        # For each batch element, select KV tokens and launch Triton kernels per query
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end:
                continue

            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # tok_idx = kv_indices[kv_start:kv_end]
            tok_idx = kv_indices[kv_start:kv_end]  # keep as int64 tensor (Triton will treat as indices)

            # Select Kc_sel and Kp_sel per batch element (no torch math or dtype conversions in forward)
            # Ensure tensors are on device and contiguous
            Kc_sel = ckv_cache[tok_idx].contiguous()  # [kv_len, 512], original dtype (e.g., bf16)
            Kp_sel = kpe_cache[tok_idx].contiguous()  # [kv_len, 64], original dtype

            # Launch one program per query i in this batch element
            grid = (1, q_len)
            _forward_query_kernel[grid](
                q_nope, q_pe, Kc_sel, Kp_sel, output, lse,
                q_start,
                q_len=q_len, kv_len=kv_len,
                sm_scale=float(sm_scale), ln2_inv=float(ln2_inv),
                NUM_HEADS=16, HEAD_DIM_CKV=512, HEAD_DIM_KPE=64,
                num_warps=4, num_stages=2,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
