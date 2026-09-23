import torch
import math
import triton
import triton.language as tl


# Triton kernel 1: compute logits_scaled[h, t] = qn[h] · Kc[t] + qp[h] · Kp[t]
@triton.jit
def compute_logits_kernel(
    logits_ptr,     # *f32 [H, T]
    qn_ptr,         # *f32 [H, Dq]
    qp_ptr,         # *f32 [H, Dp]
    Kc_ptr,         # *f32 [T, Dq]
    Kp_ptr,         # *f32 [T, Dp]
    H: tl.constexpr, T: tl.constexpr,
    Dq: tl.constexpr, Dp: tl.constexpr,
    BLOCK_T: tl.constexpr = 128
):
    h = tl.program_id(0)
    t_tile = tl.program_id(1)

    t_offsets = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)  # power-of-two
    mask_t = t_offsets < T

    # Accumulate contributions
    dot_qn = tl.zeros([BLOCK_T], dtype=tl.float32)
    dot_qp = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Reduce over Dq and Dp
    # qn[h, :] and qp[h, :] are 1D vectors of length Dq/Dp
    # We load qn[h, d] and sum over d
    for d in range(0, Dq):
        qn_d = tl.load(qn_ptr + h * Dq + d)
        # Kc[t, d] for this tile
        Kc_vals = tl.load(Kc_ptr + t_offsets * Dq + d, mask=mask_t, other=0.0)
        dot_qn += qn_d * Kc_vals

    for d in range(0, Dp):
        qp_d = tl.load(qp_ptr + h * Dp + d)
        Kp_vals = tl.load(Kp_ptr + t_offsets * Dp + d, mask=mask_t, other=0.0)
        dot_qp += qp_d * Kp_vals

    logits_tile = dot_qn + dot_qp  # [BLOCK_T]
    # Store logits_scaled[h, t_offsets]
    tl.store(logits_ptr + h * T + t_offsets, logits_tile, mask=mask_t)


# Triton kernel 2: row-wise softmax over tokens for each head
@triton.jit
def softmax_row_kernel(
    attn_ptr,         # *f32 [H, T]
    logits_scaled_ptr,  # *f32 [H, T]
    H: tl.constexpr, T: tl.constexpr,
    inv_ln2: tl.constexpr,  # float32 scalar, 1 / ln(2)
    BLOCK_T: tl.constexpr = 128
):
    h = tl.program_id(0)
    t_tile = tl.program_id(1)

    t_offsets = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    logits = tl.load(logits_scaled_ptr + h * T + t_offsets, mask=mask_t, other=-float('inf'))
    row_max = tl.max(logits, axis=0)  # scalar
    logits_shift = logits - row_max
    exp_logits = tl.exp(logits_shift)
    sum_exp = tl.sum(exp_logits, axis=0)  # scalar
    attn = exp_logits / sum_exp
    # Divide by ln(2) as per original: attn = softmax(logits_scaled) but the scaled logits here are logits_scaled; the original applies softmax after scaling, but we scaled in host. We implement softmax on scaled logits. Since the original code computes lse on scaled logits, we need to apply the same scaling. However, we can keep attn as softmax of scaled logits for later lse. To match original: we directly compute attn on logits_scaled. We will instead compute softmax on logits_scaled scaled by inv_ln2? No, original does softmax on scaled logits: logits_scaled = logits * sm_scale. We should have computed scaled logits outside. Here we just do softmax on logits_scaled. For correctness, we need to use scaled logits. We will adjust: compute scaled_logits first then softmax. The previous code did that. Fix by computing attn on logits_scaled scaled by inv_ln2? We only have logits_scaled. The original computes softmax on logits_scaled = logits * sm_scale. Our kernel doesn't have sm_scale. We'll store attn as softmax on logits_scaled directly.
    attn_tile = attn
    tl.store(attn_ptr + h * T + t_offsets, attn_tile, mask=mask_t)


# Triton kernel 3: row-wise logsumexp over tokens for each head, divide by ln(2)
@triton.jit
def lse_row_kernel(
    lse_ptr,          # *f32 [H]
    logits_scaled_ptr,  # *f32 [H, T]
    H: tl.constexpr, T: tl.constexpr,
    inv_ln2: tl.constexpr,
    BLOCK_T: tl.constexpr = 128
):
    h = tl.program_id(0)

    row_max = -float('inf')
    for t_tile in range(0, tl.cdiv(T, BLOCK_T)):
        t_offsets = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T
        logits = tl.load(logits_scaled_ptr + h * T + t_offsets, mask=mask_t, other=-float('inf'))
        local_max = tl.max(logits, axis=0)
        row_max = tl.maximum(row_max, local_max)

    sum_exp = 0.0
    for t_tile in range(0, tl.cdiv(T, BLOCK_T)):
        t_offsets = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T
        logits = tl.load(logits_scaled_ptr + h * T + t_offsets, mask=mask_t, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(logits - row_max), axis=0)

    lse_val = row_max + tl.log(sum_exp) * inv_ln2
    tl.store(lse_ptr + h, lse_val)


# Triton kernel 4: per-head GEMV: out[h, d] = sum_t attn[h, t] * Kc[t, d]
# 2D grid over (h, d_tile). Each program computes a tile of output dimension for one head.
@triton.jit
def perhead_gemv_kernel(
    out_ptr,           # *f32 [H, Dq]
    attn_ptr,          # *f32 [H, T]
    Kc_ptr,            # *f32 [T, Dq]
    H: tl.constexpr, Dq: tl.constexpr, T: tl.constexpr,
    BLOCK_D: tl.constexpr = 128
):
    h = tl.program_id(0)
    d_tile = tl.program_id(1)

    d_offsets = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]
    mask_d = d_offsets < Dq

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Loop over tokens tiles to accumulate
    for t_tile in range(0, tl.cdiv(T, 128)):  # 128 is BLOCK_T; we loop tiles across T
        t_offsets = t_tile * 128 + tl.arange(0, 128)  # power-of-two
        mask_t = t_offsets < T
        attn_tile = tl.load(attn_ptr + h * T + t_offsets, mask=mask_t, other=0.0)  # [BLOCK_T]
        # Load Kc block: [BLOCK_T, BLOCK_D]
        Kc_block = tl.load(Kc_ptr + t_offsets[:, None] * Dq + d_offsets[None, :],  # shape [128, BLOCK_D]
                           mask=mask_t[:, None] & mask_d[None, :], other=0.0)
        # Multiply and reduce across tokens dimension
        partial = tl.sum(attn_tile[:, None] * Kc_block, axis=0)  # [BLOCK_D]
        acc += partial

    tl.store(out_ptr + h * Dq + d_offsets, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and proper dtypes
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be CUDA."
        device = q_nope.device

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dq = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Output and lse
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # we'll cast to bfloat16 at the end
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        ln2 = math.log(2.0)
        inv_ln2 = 1.0 / ln2  # for Triton as constexpr-like scalar, we pass as Python float

        # For each batch element
        for b in range(B):
            # Compute number of tokens in this batch element and gather indices
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV cache for this batch element; set output to zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32)

            # Gather Kc and Kp for this batch element (flatten [T, Dq] and [T, Dp])
            Kc_b = ckv_cache[tok_idx].squeeze(1).contiguous().to(torch.float32)  # [T, Dq]
            Kp_b = kpe_cache[tok_idx].squeeze(1).contiguous().to(torch.float32)  # [T, Dp]

            # Prepare qn and qp for this batch element
            qn_b = q_nope[b].contiguous().to(torch.float32)  # [H, Dq]
            qp_b = q_pe[b].contiguous().to(torch.float32)    # [H, Dp]

            T = L_tokens

            # Allocate intermediate tensors
            logits_scaled = torch.empty((H, T), dtype=torch.float32, device=device)
            attn = torch.empty((H, T), dtype=torch.float32, device=device)

            # Launch Triton kernel to compute logits_scaled[h, t]
            grid_log = (H, triton.cdiv(T, 128))
            compute_logits_kernel[grid_log](
                logits_scaled, qn_b, qp_b, Kc_b, Kp_b,
                H=H, T=T, Dq=Dq, Dp=Dp,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Apply scaling: logits_scaled = logits * sm_scale
            logits_scaled.mul_(sm_scale)

            # Launch Triton softmax_row_kernel to compute attn[h, t] = softmax(logits_scaled[h, :])
            grid_softmax = (H, triton.cdiv(T, 128))
            softmax_row_kernel[grid_softmax](
                attn, logits_scaled,
                H=H, T=T,
                inv_ln2=inv_ln2,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Launch Triton lse_row_kernel to compute per-head lse
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                lse[b], logits_scaled,
                H=H, T=T,
                inv_ln2=inv_ln2,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Per-head GEMV: out[h, :] = attn[h, :] @ Kc[:, :]
            out_b = torch.empty((H, Dq), dtype=torch.float32, device=device)
            grid_gemv = (H, triton.cdiv(Dq, 128))
            perhead_gemv_kernel[grid_gemv](
                out_b, attn, Kc_b,
                H=H, Dq=Dq, T=T,
                BLOCK_D=128,
                num_warps=4, num_stages=2
            )

            # Store output[b, :, :]
            output[b] = out_b

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse

# Optional helpers if needed by harness
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)], 0).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope.cuda(), q_pe.cuda(), ckv_cache.cuda(), kpe_cache.cuda(), kv_indptr.cuda(), kv_indices.cuda(), sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
