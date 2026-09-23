import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_scaled_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_total, D], contiguous (L_total = len(kv_indices[b]))
    Kp_ptr,         # *fp32, shape [L_total, Dp], contiguous
    tok_idx_ptr,    # *int32, shape [L_total]
    logits_scaled_ptr,  # *fp32, flat buffer [B*H*L_total]
    B: tl.int32,
    H: tl.int32,
    D: tl.int32,
    Dp: tl.int32,
    L_total: tl.int32,
    b: tl.int32,    # batch element id (runtime scalar)
    sm_scale: tl.float32,
):
    # 2D grid over (b, h)
    h = tl.program_id(1)
    # We need b from grid as well; Triton passes program_id(0) as b
    b_id = tl.program_id(0)
    if (b_id >= B) or (h >= H):
        return

    # Compute logits_scaled[b_id, h, t] for all t in 0..L_total-1
    for t in range(0, L_total):
        # Load qn row for head h: qn[h, :] is contiguous over D
        acc1 = 0.0
        # dot product over D
        for i in range(0, D):
            q = tl.load(qn_ptr + h * D + i)
            K = tl.load(Kc_ptr + tl.load(tok_idx_ptr + t) * D + i)
            acc1 += q * K

        # Load qp row for head h: qp[h, :] is contiguous over Dp
        acc2 = 0.0
        for j in range(0, Dp):
            q = tl.load(qp_ptr + h * Dp + j)
            K = tl.load(Kp_ptr + tl.load(tok_idx_ptr + t) * Dp + j)
            acc2 += q * K

        val = acc1 + acc2
        # Store to flat buffer: index = ((b * H) + h) * L_total + t
        idx = (b_id * H + h) * L_total + t
        tl.store(logits_scaled_ptr + idx, val * sm_scale)


@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr,  # *fp32, flat buffer [B*H*L_total]
    lse_ptr,            # *fp32, flat buffer [B*H]
    B: tl.int32,
    H: tl.int32,
    L_total: tl.int32,
    b: tl.int32,
):
    h = tl.program_id(1)  # second grid dim is head
    b_id = tl.program_id(0)  # first grid dim is batch
    if (b_id >= B) or (h >= H):
        return

    # Compute max over tokens for numerical stability
    m = -float("inf")
    base = (b_id * H + h) * L_total
    for t in range(0, L_total):
        v = tl.load(logits_scaled_ptr + base + t)
        m = tl.maximum(m, v)

    # Compute sum of exp(v - m)
    sumexp = 0.0
    for t in range(0, L_total):
        v = tl.load(logits_scaled_ptr + base + t)
        sumexp += tl.exp(v - m)

    lse_val = tl.log(sumexp) / 1.4426950408889634  # log(2)
    tl.store(lse_ptr + b_id * H + h, lse_val)


@triton.jit
def compute_output_kernel(
    logits_scaled_ptr,  # *fp32, flat buffer [B*H*L_total]
    Kc_ptr,             # *fp32, shape [L_total, D], contiguous
    tok_idx_ptr,        # *int32, shape [L_total]
    output_ptr,         # *fp32, flat buffer [B*H*D]
    B: tl.int32,
    H: tl.int32,
    D: tl.int32,
    L_total: tl.int32,
    b: tl.int32,
):
    h = tl.program_id(1)  # head id
    b_id = tl.program_id(0)  # batch id
    if (b_id >= B) or (h >= H):
        return

    # Compute denominator of softmax over tokens
    denom = 0.0
    base = (b_id * H + h) * L_total
    for t in range(0, L_total):
        v = tl.load(logits_scaled_ptr + base + t)
        denom += tl.exp(v)  # since we didn't scale, use raw exp

    # Accumulate output over tokens: output[h, :] = sum_t softmax(v) * Kc[t, :]
    for d in range(0, D):
        acc = 0.0
        for t in range(0, L_total):
            v = tl.load(logits_scaled_ptr + base + t)
            attn_t = tl.exp(v) / denom
            tok_idx_t = tl.load(tok_idx_ptr + t)
            Kc_val = tl.load(Kc_ptr + tok_idx_t * D + d)
            acc += attn_t * Kc_val
        # Store output[h, d]
        tl.store(output_ptr + (b_id * H + h) * D + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, D], q_pe: [B, H, Dp], ckv_cache: [N_total, 1, D],
                kpe_cache: [N_total, 1, Dp], kv_indptr: [len_indptr], kv_indices: [L_tokens_total],
                sm_scale: float.
        Returns: output [B, H, D] bfloat16, lse [B, H] float32.
        """
        assert q_nope.dim() == 3 and q_pe.dim() == 3
        assert ckv_cache.dim() == 3 and kpe_cache.dim() == 3
        assert kv_indptr.dim() == 1 and kv_indices.dim() == 1
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D = q_nope.shape[2]
        Dp = q_pe.shape[2]
        N_total = ckv_cache.shape[0]
        device = q_nope.device

        # Cast inputs to float32 for computation
        q_nope_f32 = q_nope.to(torch.float32).contiguous()  # [B, H, D]
        q_pe_f32 = q_pe.to(torch.float32).contiguous()      # [B, H, Dp]
        Kc_all_f32 = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [N_total, D]
        Kp_all_f32 = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [N_total, Dp]

        # Prepare per-batch tok_idx from kv_indptr and kv_indices
        # Note: The provided inputs use kv_indptr of length 2 and kv_indices of length L_tokens.
        # We treat per-batch token range as kv_indices[0:] directly (since start=0,end=1 -> L_tokens=108 in tests).
        # But to match general logic, we reconstruct tok_idx per b using kv_indptr:
        # tok_idx[b] = kv_indices[kv_indptr[b]: kv_indptr[b+1]]
        tok_idx_list = []
        # Compute L_total per batch (assuming kv_indptr[b] and kv_indptr[b+1] define range)
        starts = kv_indptr[:-1].to(torch.int32).to(device)
        ends = kv_indptr[1:].to(torch.int32).to(device)
        L_total_list = (ends - starts).tolist()
        # Gather per-batch token indices
        for b in range(B):
            start = int(starts[b].item())
            end = int(ends[b].item())
            L_total = end - start
            tok_idx_list.append(kv_indices[start:end].to(torch.int32).to(device))
        # Stack tok_idx per batch
        tok_idx = torch.stack(tok_idx_list, dim=0)  # [B, L_total]

        # Allocate buffers
        logits_scaled = torch.empty(B * H * max(L_total_list), dtype=torch.float32, device=device)  # we will slice per b
        lse = torch.empty(B * H, dtype=torch.float32, device=device)
        output = torch.empty(B * H * D, dtype=torch.float32, device=device)

        # Launch kernels
        # 1) Compute logits_scaled[b, h, t] for all b, h, t
        for b in range(B):
            L_total = L_total_list[b]
            # Slice buffers per batch
            logits_scaled_b = logits_scaled[(b * H * L_total):(b * H * L_total + H * L_total)]
            _compute_logits_scaled = compute_logits_scaled_kernel[(B, H)](
                q_nope_f32[b], q_pe_f32[b], Kc_all_f32, Kp_all_f32, tok_idx[b],
                logits_scaled_b,
                B, H, D, Dp, L_total, b, float(sm_scale),
                num_warps=4
            )

        # 2) Compute lse[b, h]
        _compute_lse = compute_lse_kernel[(B, H)](
            logits_scaled, lse, B, H, max(L_total_list), 0,  # 'b' is ignored here since we use per-batch slice; pass 0
            num_warps=4
        )

        # 3) Compute output[b, h, :]
        for b in range(B):
            L_total = L_total_list[b]
            output_b = output[(b * H * D):(b * H * D + H * D)]
            _compute_output = compute_output_kernel[(B, H)](
                logits_scaled[(b * H * L_total):(b * H * L_total + H * L_total)],
                Kc_all_f32, tok_idx[b], output_b,
                B, H, D, L_total, b,
                num_warps=4
            )

        # Reshape and return
        output = output.view(B, H, D).to(torch.bfloat16)
        lse = lse.view(B, H)
        return output, lse


def run(*args):
    return ModelNew()(*args)
