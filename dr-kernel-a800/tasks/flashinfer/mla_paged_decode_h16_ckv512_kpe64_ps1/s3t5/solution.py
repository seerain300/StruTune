import math
import torch

import triton
import triton.language as tl


@triton.jit
def fused_logits_kernel(
    q_nope_ptr, q_pe_ptr, Kc_ptr, Kp_ptr, logits_ptr,
    H: tl.constexpr, Dq: tl.constexpr, Dp: tl.constexpr, T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(axis=0)  # one program per head
    # Preload q vectors for this head using compile-time Dq/Dp
    qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq))  # [Dq]
    qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp))    # [Dp]

    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        # Accumulate logits for this tile [BLOCK_T]
        logits_tile = tl.zeros([BLOCK_T], dtype=tl.float32)
        # Dot over Dq: qn @ Kc_sub
        for d in range(0, Dq):
            # Kc_sub: [BLOCK_T]
            Kc_sub = tl.load(Kc_ptr + t_idx * Dq + d, mask=mask_t, other=0.0)
            logits_tile += qn[d] * Kc_sub
        # Dot over Dp: qp @ Kp_sub
        for d in range(0, Dp):
            Kp_sub = tl.load(Kp_ptr + t_idx * Dp + d, mask=mask_t, other=0.0)
            logits_tile += qp[d] * Kp_sub
        # Store logits[h, t]
        # Note: logits_ptr is row-major [H, T] with row stride = T
        tl.store(logits_ptr + h * T + t_idx, logits_tile, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    logits_ptr, attn_ptr,
    H: tl.constexpr, T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(axis=0)  # one program per head
    # Compute row_max via tiled reduction
    row_max = tl.full((), -float('inf'), dtype=tl.float32)
    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        logits_sub = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))
        tile_max = tl.max(logits_sub, axis=0)
        row_max = tl.maximum(row_max, tile_max)

    # Compute sum of exp(logits - row_max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        logits_sub = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))
        exp_sub = tl.exp(logits_sub - row_max)
        sum_exp += tl.sum(exp_sub, axis=0)

    # Write normalized attn = exp(logits - row_max) / sum_exp
    inv_sum = 1.0 / sum_exp
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        logits_sub = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))
        attn_sub = tl.exp(logits_sub - row_max) * inv_sum
        tl.store(attn_ptr + h * T + t_idx, attn_sub, mask=mask_t)


@triton.jit
def lse_row_kernel(
    logits_ptr, lse_ptr,
    H: tl.constexpr, T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(axis=0)
    # Pass 1: compute row_max
    row_max = tl.full((), -float('inf'), dtype=tl.float32)
    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        logits_sub = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))
        tile_max = tl.max(logits_sub, axis=0)
        row_max = tl.maximum(row_max, tile_max)

    # Pass 2: compute sum of exp(logits - row_max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        logits_sub = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(logits_sub - row_max), axis=0)

    # lse = log(sum_exp) + row_max, divided by ln(2)
    lse_val = row_max + tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
    # Store as float32
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def attn_matmul_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H: tl.constexpr, Dq: tl.constexpr, T: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr,
):
    h = tl.program_id(axis=0)  # one program per head
    # Accumulator for output [Dq]
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    offs_d = tl.arange(0, BLOCK_D)
    # Loop over D tiles
    for d_start in range(0, Dq, BLOCK_D):
        d_idx = d_start + offs_d
        mask_d = d_idx < Dq
        # For each token tile, accumulate contributions into acc
        for t_start in range(0, T, BLOCK_T):
            t_idx = t_start + tl.arange(0, BLOCK_T)
            mask_t = t_idx < T
            # attn_sub: [BLOCK_T]
            attn_sub = tl.load(attn_ptr + h * T + t_idx, mask=mask_t, other=0.0)
            # Kc_sub: [BLOCK_T, BLOCK_D]
            Kc_sub = tl.load(
                Kc_ptr + t_idx[:, None] * Dq + d_idx[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            )
            # Accumulate over tokens in this tile
            # attn_sub[:, None] * Kc_sub -> [BLOCK_T, BLOCK_D]; sum over axis=0 -> [BLOCK_D]
            acc += tl.sum(attn_sub[:, None] * Kc_sub, axis=0)
        # Store acc for valid d
        tl.store(out_ptr + h * Dq + d_idx, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs on CUDA (Triton requires CUDA)
        device = q_nope.device
        if device.type != 'cuda':
            device = torch.device('cuda')
            q_nope = q_nope.to(device)
            q_pe = q_pe.to(device)
            ckv_cache = ckv_cache.to(device)
            kpe_cache = kpe_cache.to(device)
            kv_indptr = kv_indptr.to(device)
            kv_indices = kv_indices.to(device)

        # Cast to float32 for compute (original code casts anyway)
        q_nope_f32 = q_nope.contiguous().to(torch.float32)  # [B, H, Dq]
        q_pe_f32 = q_pe.contiguous().to(torch.float32)      # [B, H, Dp]
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [N, Dq]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [N, Dp]

        B = q_nope_f32.shape[0]
        H = q_nope_f32.shape[1]
        Dq = q_nope_f32.shape[2]
        Dp = q_pe_f32.shape[2]

        # Output tensors (compute in float32, cast to bfloat16 at end)
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Determine number of used tokens
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg
            if L_tokens <= 0:
                # No KV entries used for this batch element; output zeros and skip kernels
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc and Kp for used tokens
            tok_idx = kv_indices[page_beg:page_end].to(torch.long).contiguous()  # [T]
            Kc = Kc_all[tok_idx]  # [T, Dq]
            Kp = Kp_all[tok_idx]  # [T, Dp]

            # q vectors for this batch element
            qn = q_nope_f32[b]  # [H, Dq]
            qp = q_pe_f32[b]    # [H, Dp]

            # Allocate intermediate tensors
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # [H, T]
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)    # [H, T]

            # Launch fused_logits_kernel: compute logits[h, t] = qn[h, :] @ Kc[t, :] + qp[h, :] @ Kp[t, :]
            BLOCK_T = 256
            grid = (H,)
            fused_logits_kernel[grid](
                qn, qp, Kc, Kp, logits,
                H=H, Dq=Dq, Dp=Dp, T=L_tokens,
                BLOCK_T=BLOCK_T,
                num_warps=4,
            )

            # Scale logits by sm_scale
            logits_scaled = logits * sm_scale

            # Compute softmax per row (attention) via Triton
            attn_mat = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # [H, T]
            softmax_row_kernel[grid](
                logits_scaled, attn_mat,
                H=H, T=L_tokens,
                BLOCK_T=BLOCK_T,
                num_warps=4,
            )

            # Compute lse per row (logsumexp) via Triton
            lse_b = torch.empty((H,), dtype=torch.float32, device=device)
            lse_row_kernel[grid](
                logits_scaled, lse_b,
                H=H, T=L_tokens,
                BLOCK_T=BLOCK_T,
                num_warps=4,
            )
            # Store per head lse
            lse[b] = lse_b

            # Compute out[h, :] = attn_mat[h, :] @ Kc[:, :] per head via Triton
            BLOCK_D = 128
            out_tmp = torch.empty((H, Dq), dtype=torch.float32, device=device)
            attn_matmul_kernel[grid](
                attn_mat, Kc, out_tmp,
                H=H, Dq=Dq, T=L_tokens,
                BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T,
                num_warps=4,
            )
            output[b] = out_tmp

        # Match original output dtypes
        output = output.to(torch.bfloat16)
        lse = lse  # float32 as in original
        return output, lse

# The following helper functions mirror the original for evaluation.
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

# Entry point class required by evaluator
class Model(torch.nn.Module):
    def forward(self, *args):
        return fused_operator(*args)


def run(*args):
    return ModelNew()(*args)
