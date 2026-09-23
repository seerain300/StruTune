import torch
import math
import triton
import triton.language as tl


@triton.jit
def qk_dot_kernel(
    Q_ptr, K_ptr, LOGITS_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute LOGITS = Q @ K^T for each segment:
      - Q: [Nq, Hq, D], float32
      - K: [Nk, Hq, D], float32
      - LOGITS: [Nq, Hq, Nk], float32
    Grid: (pid0 tiles over Nq, pid1 tiles over Nk, pid2 over Hq)
    """
    pid0 = tl.program_id(0)  # tile over q
    pid1 = tl.program_id(1)  # tile over k
    pid2 = tl.program_id(2)  # head index in 0..Hq-1

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        mask_d = d_offsets < D

        # Load Q tile: [BLOCK_M, BLOCK_D] from Q[pid2]
        Q_tile = tl.load(
            Q_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0
        )
        # Load K tile: [BLOCK_N, BLOCK_D] from K[:, pid2]
        K_tile = tl.load(
            K_ptr + k_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0
        )

        # acc += Q_tile @ K_tile^T -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(Q_tile, tl.trans(K_tile))

    # Scale logits
    acc = acc * sm_scale

    # Store: LOGITS[pid0, pid2, pid1] tile
    # LOGITS linear index = q * (Hq * Nk) + h * Nk + k
    tl.store(
        LOGITS_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def softmax_lastdim_kernel(
    X_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Softmax over the last dimension (Nk) of X for each (q, h):
      - X: [Nq, Hq, Nk], float32
      - Y: [Nq, Hq, Nk], float32
    Grid: (tiles over Nq, Hq)
    """
    pid0 = tl.program_id(0)  # tile over q
    pid1 = tl.program_id(1)  # head index

    q_start = pid0 * BLOCK_M
    q_offsets = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq

    # First pass: compute max over Nk per q
    max_val = tl.full((BLOCK_M,), -1e20, tl.float32)
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        x = tl.load(
            X_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-1e20
        )
        tile_max = tl.max(x, axis=1)  # [BLOCK_M]
        max_val = tl.maximum(max_val, tile_max)

    # Second pass: compute sum of exp(X - max) over Nk per q
    sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        x = tl.load(
            X_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-1e20
        )
        x = x - max_val[:, None]
        exp_x = tl.exp(x)
        tile_sum = tl.sum(exp_x, axis=1)  # [BLOCK_M]
        sum_exp += tile_sum

    inv_sum = 1.0 / sum_exp  # [BLOCK_M]

    # Third pass: write normalized values
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        x = tl.load(
            X_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-1e20
        )
        x = x - max_val[:, None]
        exp_x = tl.exp(x)
        y = exp_x * inv_sum[:, None]
        tl.store(
            Y_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            y,
            mask=mask_q[:, None] & mask_k[None, :]
        )


@triton.jit
def lse_lastdim_kernel(
    X_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr,
    scale: tl.float32,  # 1.0 / log(2.0)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute per (q, h): lse = log(sum(exp(X * scale))) * scale
    Here X is already scaled by sm_scale in host code, and we multiply by scale=1/log(2) to divide by ln(2).
    Grid: (tiles over Nq, Hq)
    """
    pid0 = tl.program_id(0)  # tile over q
    pid1 = tl.program_id(1)  # head index

    q_start = pid0 * BLOCK_M
    q_offsets = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq

    sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        x = tl.load(
            X_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0
        )
        # x is scaled in host; we don't need to apply sm_scale here
        sum_exp += tl.sum(tl.exp(x), axis=1)

    # Compute lse = log(sum_exp) / log(2) = log(sum_exp) * scale
    lse_val = tl.log(sum_exp) * scale  # [BLOCK_M]
    # Store lse per (q, h): linear index = q * Hq + h
    # LSE_ptr is [Nq, Hq] contiguous
    for i in range(BLOCK_M):
        q_i = q_start + i
        if q_i < Nq:
            tl.store(LSE_ptr + q_i * Hq + pid1, lse_val[i])


@triton.jit
def matvec_output_kernel(
    WEIGHT_ptr, VEXP_ptr, OUT_ptr,
    Nq: tl.int32, Nk: tl.int32, D: tl.constexpr,
    Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute OUT = WEIGHT @ VEXP along last dim:
      - WEIGHT: [Nq, Hq, Nk], float32 (softmax output)
      - VEXP: [Nk, Hq, D], float32 (expanded V with 4x heads)
      - OUT: [Nq, Hq, D], float32
    Grid: (tiles over Nq, Hq)
    We loop over Nk in tiles and accumulate over D.
    """
    pid0 = tl.program_id(0)  # tile over q
    pid1 = tl.program_id(1)  # head index

    q_start = pid0 * BLOCK_M
    q_offsets = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq

    out_acc = tl.zeros((BLOCK_M, Hq, D), dtype=tl.float32)

    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk

        # Load WEIGHT tile: [BLOCK_M, BLOCK_N]
        weight = tl.load(
            WEIGHT_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0
        )

        # Loop over D in chunks and accumulate
        for d_start in range(0, D, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            mask_d = d_offsets < D

            # VEXP tile: [BLOCK_N, BLOCK_D], vexp[k, h, d]
            vexp_tile = tl.load(
                VEXP_ptr + k_offsets[:, None] * (Hq * D) + pid1 * D + d_offsets[None, :],
                mask=mask_k[:, None] & mask_d[None, :],
                other=0.0
            )

            # out_acc += weight[:, :, None] * vexp_tile[None, :, :]
            # That is: for each (q,k), add vexp_tile across D to out_acc[q, h, d]
            out_acc += weight[:, :, None] * vexp_tile[None, :, :]

    # Store out_acc (float32); we'll cast to bfloat16 in host after
    # OUT linear index = q * (Hq * D) + h * D + d
    # We store full out_acc: q dimension
    for q_i in range(BLOCK_M):
        q_i_val = q_start + q_i
        if q_i_val < Nq:
            for h in range(Hq):
                for d_start in range(0, D, BLOCK_D):
                    d_offsets = d_start + tl.arange(0, BLOCK_D)
                    mask_d = d_offsets < D
                    vals = out_acc[q_i, h, d_offsets]
                    tl.store(
                        OUT_ptr + q_i_val * (Hq * D) + h * D + d_offsets,
                        vals,
                        mask=mask_d
                    )


def _run_triton(q, k, v, qo_indptr, kv_indptr, sm_scale):
    """
    q: [Nq, 32, 128], bfloat16
    k: [Nkv, 8, 128], bfloat16
    v: [Nkv, 8, 128], bfloat16
    qo_indptr: [len_indptr], int32
    kv_indptr: [len_indptr], int32
    sm_scale: float32
    returns (output: [Nq, 32, 128] bfloat16, lse: [Nq, 32] float32)
    """
    device = q.device
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16

    # Prepare expanded K and V
    num_qo_heads = 32
    num_kv_heads = 8
    gqa_ratio = num_qo_heads // num_kv_heads
    k_expanded = k.repeat_interleave(gqa_ratio, dim=1)  # [Nkv, 32, 128]
    v_expanded = v.repeat_interleave(gqa_ratio, dim=1)  # [Nkv, 32, 128]

    total_q = q.shape[0]
    total_kv = k.shape[0]
    len_indptr = qo_indptr.shape[0]

    # Allocate outputs
    output = torch.empty((total_q, num_qo_heads, 128), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

    # Constants for Triton
    Hq = num_qo_heads
    D = 128
    # Tiling sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 32

    # Launch per segment
    for b in range(1, len_indptr):
        q_start = int(qo_indptr[b - 1].item())
        q_end = int(qo_indptr[b].item())
        kv_start = int(kv_indptr[b - 1].item())
        kv_end = int(kv_indptr[b].item())

        if q_start >= q_end or kv_start >= kv_end:
            continue

        Nq = q_end - q_start
        Nk = kv_end - kv_start

        # Extract batched Q and K (float32 for compute)
        q_batch = q[q_start:q_end].to(torch.float32)  # [Nq, 32, 128]
        k_batch = k_expanded[kv_start:kv_end].to(torch.float32)  # [Nk, 32, 128]
        vexp_batch = v_expanded[kv_start:kv_end].to(torch.float32)  # [Nk, 32, 128]

        # Compute and store logits [Nq, 32, Nk], float32
        logits = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=device)
        grid_qk = (triton.cdiv(Nq, BLOCK_M), triton.cdiv(Nk, BLOCK_N), Hq)
        qk_dot_kernel[grid_qk](
            q_batch, k_batch, logits,
            Nq, Nk,
            Hq, D,
            sm_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2
        )

        # Apply causal mask in Triton by filling logits with -inf where not causal
        # causal: kv_idx < (q_idx + 1 + delta)
        delta = Nk - Nq  # segment-level delta
        # We'll do this by another Triton kernel over (q,h) tiles that computes mask and writes masked logits.
        # To keep code simple, we implement mask in host using torch ops on logits (but we can do in Triton by recomputing).
        # Instead, we implement a Triton kernel that reads logits and writes masked logits to a buffer.
        # However, we can avoid extra buffers by doing mask inside softmax: we can't read original after softmax, so we'll create a masked buffer from logits.
        # Create masked_logits = logits
        masked_logits = logits.clone()

        # Triton mask kernel: set positions not in causal range to -1e20
        # Define mask kernel:
        @triton.jit
        def apply_causal_mask_kernel(LOGITS_ptr, MASKED_ptr, Nq: tl.int32, Nk: tl.int32, delta: tl.int32, Hq: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
            pid0 = tl.program_id(0)  # tile over q
            pid1 = tl.program_id(1)  # head

            q_start = pid0 * BLOCK_M
            q_offsets = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
            mask_q = q_offsets < Nq

            for k_start in range(0, Nk, BLOCK_N):
                k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
                mask_k = k_offsets < Nk

                x = tl.load(
                    LOGITS_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
                    mask=mask_q[:, None] & mask_k[None, :],
                    other=0.0
                )
                # causal mask: kv_idx < (q_idx + 1 + delta)
                # delta is per segment scalar
                # For each q_offsets[:, None], compute condition: k_offsets[None, :] < (q_offsets[:, None] + 1 + delta)
                cond = k_offsets[None, :] < (q_offsets[:, None] + 1 + delta)
                # Set non-causal to -1e20
                x = tl.where(cond, x, -1e20)
                tl.store(
                    MASKED_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
                    x,
                    mask=mask_q[:, None] & mask_k[None, :]
                )

        # Launch causal mask kernel
        masked_logits = torch.empty_like(logits)
        apply_causal_mask_kernel[(triton.cdiv(Nq, BLOCK_M), Hq)](
            logits, masked_logits, Nq, Nk, delta, Hq,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Softmax over last dim (Nk) for each (q,h) using Triton
        softmax_out = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=device)
        # For softmax_lastdim_kernel, we need grid over tiles of Nq and Hq
        BLOCK_M_sm = 64
        BLOCK_N_sm = 64
        softmax_lastdim_kernel[(triton.cdiv(Nq, BLOCK_M_sm), Hq)](
            masked_logits, softmax_out,
            Nq, Nk, Hq,
            BLOCK_M=BLOCK_M_sm, BLOCK_N=BLOCK_N_sm,
            num_warps=4, num_stages=2
        )

        # Compute LSE per (q,h): logsumexp over masked_logits * sm_scale, divide by ln(2)
        lse_seg = torch.empty((Nq, Hq), dtype=torch.float32, device=device)
        scale_div_ln2 = 1.0 / math.log(2.0)  # 1.0 / ln(2)
        lse_lastdim_kernel[(triton.cdiv(Nq, BLOCK_M), Hq)](
            masked_logits, lse_seg,
            Nq, Nk, Hq,
            scale_div_ln2,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Now compute output = softmax(masked_logits) @ v_expanded
        # Prepare WEIGHT = softmax_out and VEXP = v_expanded[kv_start:kv_end], shape [Nk, 32, 128]
        # We need OUT: [Nq, 32, 128], float32
        out = torch.empty((Nq, Hq, D), dtype=torch.float32, device=device)
        matvec_output_kernel[(triton.cdiv(Nq, BLOCK_M), Hq)](
            softmax_out, vexp_batch, out,
            Nq, Nk, D, Hq,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2
        )

        # Write out to final output
        # output[q_start:q_end, :, :] = out
        # We need to place out into output with correct q offsets
        for q_i in range(Nq):
            q_idx = q_start + q_i
            # out[q_i, :, :] is already [Hq, D]
            # output is [Nq, Hq, D]
            # torch.copy_ is allowed (it's not a torch compute op; it's data movement)
            output[q_idx].copy_(out[q_i])

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        return _run_triton(q, k, v, qo_indptr, kv_indptr, sm_scale)

# The following helpers can remain the same
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16)
    k = torch.randn([1, 8, 128], dtype=torch.bfloat16)
    v = torch.randn([1, 8, 128], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k, v, qo_indptr, kv_indptr, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
