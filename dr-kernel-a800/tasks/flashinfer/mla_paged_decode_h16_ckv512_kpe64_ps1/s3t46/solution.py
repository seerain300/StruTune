import torch
import math
import triton
import triton.language as tl


# Triton kernel: compute logits[h, t] = qn[h] · Kc[t] + qp[h] · Kp[t]
@triton.jit
def compute_logits_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
    H: tl.constexpr, Dq: tl.constexpr, Dp: tl.constexpr, T: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr,
):
    h_block = tl.program_id(0)
    t_block = tl.program_id(1)
    hs = h_block * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    ts = t_block * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]

    logits = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)

    # Load qn and qp for all h in this block
    qn = tl.load(qn_ptr + hs[:, None] * Dq + tl.arange(0, Dq)[None, :],
                 mask=hs[:, None] < H, other=0.0)  # [BLOCK_H, Dq]
    qp = tl.load(qp_ptr + hs[:, None] * Dp + tl.arange(0, Dp)[None, :],
                 mask=hs[:, None] < H, other=0.0)  # [BLOCK_H, Dp]

    # Loop over tokens in tiles
    for t0 in range(0, T, BLOCK_T):
        ts_vec = t0 + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        mask_t = ts_vec < T

        # Load Kc and Kp for tokens in this tile
        Kc = tl.load(Kc_ptr + ts_vec[:, None] * Dq + tl.arange(0, Dq)[None, :],
                     mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dq]
        Kp = tl.load(Kp_ptr + ts_vec[:, None] * Dp + tl.arange(0, Dp)[None, :],
                     mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dp]

        # Accumulate dot products
        acc1 = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        for d in range(0, Dq, 128):
            qsub = qn[:, d:(d+128)]  # [BLOCK_H, 128]
            Ksub = Kc[d:(d+128), :]  # [128, BLOCK_T]
            acc1 += tl.sum(qsub[:, :, None] * Ksub[None, :, :], axis=1)
        acc2 = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        for d in range(0, Dp, 64):
            qsub = qp[:, d:(d+64)]  # [BLOCK_H, 64]
            Ksub = Kp[d:(d+64), :]  # [64, BLOCK_T]
            acc2 += tl.sum(qsub[:, :, None] * Ksub[None, :, :], axis=1)
        logits += acc1 + acc2

    # Store results
    store_mask = (hs[:, None] < H) & (ts[None, :] < T)
    tl.store(out_ptr + hs[:, None] * T + ts[None, :], logits, mask=store_mask)


# Triton kernel: row-wise softmax over tokens T for each head h
@triton.jit
def softmax_row_kernel(scaled_logits_ptr, attn_ptr,
                        H: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    if h >= H:
        return
    idx = tl.arange(0, BLOCK_T)
    row = tl.load(scaled_logits_ptr + h * T + idx,
                  mask=idx < T, other=-float("inf"))  # [BLOCK_T]
    row_max = tl.max(row, axis=0)
    row_exp = tl.exp(row - row_max)
    row_sum = tl.sum(row_exp, axis=0)
    attn_row = row_exp / row_sum
    tl.store(attn_ptr + h * T + idx, attn_row, mask=idx < T)


# Triton kernel: row-wise logsumexp over tokens T for each head h, divide by ln(2)
@triton.jit
def lse_row_kernel(scaled_logits_ptr, lse_ptr,
                   H: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    if h >= H:
        return
    idx = tl.arange(0, BLOCK_T)
    row = tl.load(scaled_logits_ptr + h * T + idx,
                  mask=idx < T, other=-float("inf"))  # [BLOCK_T]
    row_max = tl.max(row, axis=0)
    row_exp = tl.exp(row - row_max)
    row_sum = tl.sum(row_exp, axis=0)
    lse_val = row_max + tl.log(row_sum) * (1.0 / 1.4426950408889634)  # 1/ln(2)
    tl.store(lse_ptr + h, lse_val)


# Triton kernel: per-head GEMV: out[h, :] = attn[h, :] @ Kc[:, :]
@triton.jit
def perhead_gemv_kernel(attn_ptr, Kc_ptr, out_ptr,
                        H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr,
                        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    if h >= H:
        return
    out_vec = tl.zeros((Dq,), dtype=tl.float32)
    for t0 in range(0, T, BLOCK_T):
        ts = t0 + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        mask_t = ts < T
        attn_row = tl.load(attn_ptr + h * T + ts, mask=mask_t, other=0.0)  # [BLOCK_T]
        Kc_block = tl.load(Kc_ptr + ts[:, None] * Dq + tl.arange(0, Dq)[None, :],
                           mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dq]
        # Accumulate contributions
        for t_i in range(0, BLOCK_T):
            t_idx = t0 + t_i
            if t_idx < T:
                out_vec += attn_row[t_i] * Kc_block[t_i, :]
    tl.store(out_ptr + h * Dq + tl.arange(0, Dq), out_vec, mask=tl.arange(0, Dq) < Dq)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Constants (asserted in original): num_qo_heads == 16, head_dim_ckv == 512, head_dim_kpe == 64
        H = 16
        Dq = 512
        Dp = 64

        B = q_nope.shape[0]

        # Allocate outputs
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # compute in float32
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].contiguous()  # int32
            # Gather Kc and Kp for these tokens (float32)
            Kc_b = ckv_cache[tok_idx].squeeze(1).contiguous().to(torch.float32)  # [L_tokens, Dq]
            Kp_b = kpe_cache[tok_idx].squeeze(1).contiguous().to(torch.float32)  # [L_tokens, Dp]

            # qn and qp for this batch (float32)
            qn_b = q_nope[b].contiguous().to(torch.float32)  # [H, Dq]
            qp_b = q_pe[b].contiguous().to(torch.float32)   # [H, Dp]

            # Allocate intermediate tensors
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch Triton kernel to compute logits = qn @ Kc^T + qp @ Kp^T
            BLOCK_H = 16
            BLOCK_T = 128
            grid_log = (triton.cdiv(H, BLOCK_H), triton.cdiv(L_tokens, BLOCK_T))
            compute_logits_kernel[grid_log](
                qn_b, qp_b, Kc_b, Kp_b, logits,
                H=H, Dq=Dq, Dp=Dp, T=L_tokens,
                BLOCK_H=BLOCK_H, BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )

            # Scale
            logits_scaled = logits * sm_scale

            # Row-wise softmax over tokens
            grid_sm = (H,)
            softmax_row_kernel[grid_sm](
                logits_scaled, attn,
                H=H, T=L_tokens,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Row-wise lse per head, divide by ln(2)
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                logits_scaled, lse[b],
                H=H, T=L_tokens,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Per-head GEMV: out[h, :] = attn[h, :] @ Kc[:, :]
            out_b = torch.empty((H, Dq), dtype=torch.float32, device=device)
            grid_gemv = (H,)
            perhead_gemv_kernel[grid_gemv](
                attn, Kc_b, out_b,
                H=H, T=L_tokens, Dq=Dq,
                BLOCK_T=128, BLOCK_D=128,
                num_warps=4, num_stages=2
            )

            # Assign to output
            output[b] = out_b

        # Cast output to bfloat16 as in original
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
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
