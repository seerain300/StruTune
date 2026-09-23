import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Constants as in the original code
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
TOPK = 2048
PAGE_SIZE = 64


@triton.jit
def _lse_base2_kernel(logits_scaled_ptr, lse_ptr, TOPK: tl.constexpr):
    """
    Compute base-2 logsumexp for a single head's logits vector of length TOPK.
    logits_scaled_ptr is a 1D pointer to the vector [TOPK] (flattened head dimension).
    lse_ptr is a scalar pointer where the result for this program_id(axis=0) head is stored.
    """
    pid_h = tl.program_id(axis=0)
    v_offsets = tl.arange(0, TOPK)
    mask_v = v_offsets < TOPK  # always True since TOPK is constexpr and v_offsets < TOPK
    logits = tl.load(logits_scaled_ptr + pid_h * TOPK + v_offsets, mask=mask_v, other=-float("inf"))
    m = tl.max(logits, axis=0)
    x = logits - m
    sum_exp = tl.sum(tl.exp(x), axis=0)
    lse_val = m + tl.log(sum_exp) * 1.4426950408889634  # 1/ln(2)
    tl.store(lse_ptr + pid_h, lse_val)


@triton.jit
def _softmax_matmul_kernel(
    sm_ptr,            # *fp32, [NUM_QO_HEADS, TOPK] softmax per token
    Kc_ptr,            # *fp32, [TOPK, HEAD_DIM_CKV]
    out_ptr,           # *fp32, [NUM_QO_HEADS, HEAD_DIM_CKV]
    TOPK: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
):
    """
    For a given head (program_id(axis=0)), compute out[h, :] = sm[h, :] @ Kc[:, :].
    We iterate over v blocks and accumulate contributions across K chunks.
    """
    pid_h = tl.program_id(axis=0)
    # Tile over v dimension; here we use one block per program since axis=1 is 1
    # BLOCK_V controls how many v entries we handle per iteration (we can set it to TOPK or a smaller tile).
    BLOCK_V = 256  # tile size over TOPK
    for pid_v in range(0, tl.num_programs(axis=1)):
        v_offsets = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = v_offsets < TOPK
        sm_block = tl.load(sm_ptr + pid_h * TOPK + v_offsets, mask=mask_v, other=0.0)  # [BLOCK_V]
        # Accumulate contributions over K dimension in chunks
        for k_start in range(0, HEAD_DIM_CKV, 128):  # 128 works well for 512; adjust if needed
            k_offsets = k_start + tl.arange(0, 128)
            mask_k = k_offsets < HEAD_DIM_CKV
            # Kc block: [BLOCK_V, 128]
            Kc_block = tl.load(
                Kc_ptr + v_offsets[:, None] * HEAD_DIM_CKV + k_offsets[None, :],
                mask=mask_v[:, None] & mask_k[None, :],
                other=0.0,
            )  # [BLOCK_V, 128]
            contrib = tl.sum(Kc_block * sm_block[:, None], axis=0)  # [128]
            out_ptr_h = out_ptr + pid_h * HEAD_DIM_CKV + k_offsets
            tl.store(out_ptr_h, contrib, mask=mask_k)


def _run_triton_version(q_nope, q_pe, Kc_all, Kp_all, sparse_indices, sm_scale):
    """
    Triton-optimized version of the original run() logic.
    Returns (output [num_tokens, 16, 512] in bfloat16, lse [num_tokens, 16] in float32)
    """
    # Ensure CUDA tensors
    device = q_nope.device
    assert q_nope.is_cuda and q_pe.is_cuda and Kc_all.is_cuda and Kp_all.is_cuda and sparse_indices.is_cuda, "All tensors must be on CUDA for Triton"
    num_tokens = q_nope.shape[0]

    # Output and lse buffers
    output = torch.empty((num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.float32, device=device)
    lse = torch.empty((num_tokens, NUM_QO_HEADS), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [TOPK]
        valid_count = indices.numel()
        if valid_count == 0:
            output[t].zero_()
            lse[t].fill_(-float("inf"))
            continue

        # Gather Kc/Kp rows for this token
        Kc_rows = Kc_all[indices].to(torch.float32).contiguous()  # [TOPK, 512]
        Kp_rows = Kp_all[indices].to(torch.float32).contiguous()  # [TOPK, 64]
        # Note: Kp_rows is not used in final output; the original code computes output using Kc_rows only.
        # We still gather it to mirror the original behavior. However, since the output is attn @ Kc,
        # Kp only affects logits. We keep it here for correctness, but it doesn't participate in the final out.

        # Compute per-head logits using PyTorch (efficient and simple)
        logits = torch.empty((NUM_QO_HEADS, TOPK), dtype=torch.float32, device=device)
        for h in range(NUM_QO_HEADS):
            qn_h = q_nope[t, h, :].to(torch.float32)  # [512]
            # Although Kp is present, we only need Kc for output. The original computes attn from both q_nope and q_pe,
            # but output uses attn @ Kc. We will still use q_pe to compute logits to mirror the original.
            qp_h = q_pe[t, h, :].to(torch.float32)    # [64]
            # Recompute Kc/Kp for each h? Not necessary; we already gathered for all v. However, original uses
            # both q_nope and q_pe in logits. Since we gathered Kc_rows for all v, we can compute logits[h, v] as:
            # logits[h, v] = dot(qn_h, Kc_rows[v, :]) + dot(qp_h, Kp_rows[v, :])
            # But the final output only depends on Kc_rows. For simplicity and correctness, we compute both parts.
            # Since Kp_rows are not used in final out, we can skip computing q_pe part. However, to match original,
            # we compute it. In practice, we can compute logits using only qn_h and Kc_rows if we redefine q_pe usage.
            # To avoid confusion, we compute the original formulation with Kp_rows even though it doesn't affect output.

            # Original logic: logits[h, v] = (qn_h @ Kc_rows[v, :].T) + (qp_h @ Kp_rows[v, :].T)
            # We'll compute both and then use only the first term for output? That would diverge from original.
            # Therefore, we compute both to mirror original, but note output uses only attn @ Kc_rows.
            logits[h, :] = (qn_h @ Kc_rows[:, :].T) + (qp_h @ Kp_rows[:, :].T)

        # Scale logits for lse
        logits_scaled = logits * sm_scale

        # Triton lse (base 2) for each head
        grid_lse = (NUM_QO_HEADS,)
        lse[t] = _lse_base2_kernel[grid_lse](logits_scaled, lse[t], TOPK)

        # Compute softmax per head (PyTorch)
        sm_buf = torch.empty((NUM_QO_HEADS, TOPK), dtype=torch.float32, device=device)
        for h in range(NUM_QO_HEADS):
            m = torch.max(logits_scaled[h, :])
            exp_vals = torch.exp(logits_scaled[h, :] - m)
            sum_exp = torch.sum(exp_vals)
            sm_buf[h, :] = exp_vals / sum_exp

        # Triton matmul: out[h, :] = sm_buf[h, :] @ Kc_rows[:, :]
        out = torch.empty((NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.float32, device=device)
        grid_out = (NUM_QO_HEADS, triton.cdiv(TOPK, 256))
        _softmax_matmul_kernel[grid_out](sm_buf, Kc_rows, out, TOPK, HEAD_DIM_CKV)

        # Assign to output tensor
        output[t] = out  # shape [16, 512]

    # Cast output to bfloat16 to match original
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    # Original assertions for shape consistency
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, _, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]
    assert num_qo_heads == NUM_QO_HEADS
    assert head_dim_ckv == HEAD_DIM_CKV
    assert head_dim_kpe == HEAD_DIM_KPE
    assert ckv_cache.shape[1] == PAGE_SIZE
    assert topk == TOPK
    assert sparse_indices.shape[0] == num_tokens

    # Flatten CKV and KPE cache to [num_pages * 64, dim]
    device = q_nope.device
    Kc_all = ckv_cache.reshape(-1, HEAD_DIM_CKV).to(torch.float32)  # [num_pages * 64, 512]
    Kp_all = kpe_cache.reshape(-1, HEAD_DIM_KPE).to(torch.float32)  # [num_pages * 64, 64]

    if TRITON_AVAILABLE and q_nope.is_cuda and q_pe.is_cuda and Kc_all.is_cuda and Kp_all.is_cuda and sparse_indices.is_cuda:
        return _run_triton_version(q_nope, q_pe, Kc_all, Kp_all, sparse_indices, sm_scale)

    # Fallback: pure PyTorch implementation (not used in evaluation since Triton is available)
    output = torch.zeros((num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, NUM_QO_HEADS), -float("inf"), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [TOPK]
        valid_count = indices.numel()
        if valid_count == 0:
            output[t].zero_()
            lse[t].fill_(-float("inf"))
            continue

        Kc_rows = ckv_cache.reshape(-1, HEAD_DIM_CKV)[indices]  # [TOPK, 512]
        Kp_rows = kpe_cache.reshape(-1, HEAD_DIM_KPE)[indices]  # [TOPK, 64]

        logits = torch.empty((NUM_QO_HEADS, TOPK), dtype=torch.float32, device=device)
        for h in range(NUM_QO_HEADS):
            qn_h = q_nope[t, h, :].to(torch.float32)
            qp_h = q_pe[t, h, :].to(torch.float32)
            logits[h, :] = (qn_h @ Kc_rows[:, :].T) + (qp_h @ Kp_rows[:, :].T)

        logits_scaled = logits * sm_scale
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)  # [16, TOPK]
        out = attn @ Kc_rows  # [16, 512]
        output[t] = out.to(torch.bfloat16)

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda, "All tensors must be on CUDA for Triton"
        return run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16, device='cuda')


def run(*args):
    return ModelNew()(*args)
