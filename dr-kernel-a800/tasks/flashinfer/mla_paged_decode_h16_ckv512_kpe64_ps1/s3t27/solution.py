import torch
import math
import triton
import triton.language as tl


@triton.jit
def fused_logits_kernel(
    qn_ptr,  # *f32 [H, Dq]
    qp_ptr,  # *f32 [H, Dp]
    Kc_ptr,  # *f32 [T, Dq]
    Kp_ptr,  # *f32 [T, Dp]
    logits_ptr,  # *f32 [H, T]
    H: tl.constexpr,     # number of heads (16)
    T: tl.constexpr,     # number of tokens
    Dq: tl.constexpr,    # 512
    Dp: tl.constexpr,    # 64
    BLOCK_T: tl.constexpr,  # e.g., 128 (power-of-two)
):
    h = tl.program_id(0)  # one program per head
    # Load qn[h, :] and qp[h, :]
    offs_qn = tl.arange(0, Dq)  # Dq is constexpr (512)
    qn_row = tl.load(qn_ptr + h * Dq + offs_qn)  # [Dq]
    offs_qp = tl.arange(0, Dp)  # Dp is constexpr (64)
    qp_row = tl.load(qp_ptr + h * Dp + offs_qp)  # [Dp]

    # Compute logits[h, t] = dot(qn_row, Kc[t, :]) + dot(qp_row, Kp[t, :])
    # Iterate over tokens in tiles
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        mask_t = offs_t < T
        # Load Kc and Kp tiles
        Kc_tile = tl.load(
            Kc_ptr + offs_t[:, None] * Dq + tl.arange(0, Dq)[None, :],
            mask=mask_t[:, None],
            other=0.0
        )  # [BLOCK_T, Dq]
        Kp_tile = tl.load(
            Kp_ptr + offs_t[:, None] * Dp + tl.arange(0, Dp)[None, :],
            mask=mask_t[:, None],
            other=0.0
        )  # [BLOCK_T, Dp]

        # Accumulate dot products
        dot1 = tl.sum(Kc_tile * qn_row[None, :], axis=1)  # [BLOCK_T]
        dot2 = tl.sum(Kp_tile * qp_row[None, :], axis=1)  # [BLOCK_T]

        logits_row = dot1 + dot2  # [BLOCK_T]
        # Store with mask
        tl.store(logits_ptr + h * T + offs_t, logits_row, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    logits_ptr,   # *f32 [H, T]
    attn_ptr,     # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,  # e.g., 128
):
    h = tl.program_id(0)  # one program per head
    # Load the row
    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < T
    logits_row = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))  # [BLOCK_T]
    # Numerically stable softmax
    row_max = tl.max(logits_row, axis=0)
    logits_stable = logits_row - row_max
    exp_row = tl.exp(logits_stable)
    sum_exp = tl.sum(exp_row, axis=0)
    attn_row = exp_row / sum_exp
    # Store
    tl.store(attn_ptr + h * T + offs_t, attn_row, mask=mask_t)


@triton.jit
def lse_row_kernel(
    logits_ptr,   # *f32 [H, T]
    lse_ptr,      # *f32 [H]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,  # e.g., 128
):
    h = tl.program_id(0)  # one program per head
    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < T
    logits_row = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))  # [BLOCK_T]
    row_max = tl.max(logits_row, axis=0)
    sum_exp = tl.sum(tl.exp(logits_row - row_max), axis=0)
    lse_val = row_max + tl.log(sum_exp)  # logsumexp
    # Store lse scaled by 1/ln(2) if needed
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def matmul_row_kernel(
    attn_ptr,     # *f32 [H, T]
    Kc_ptr,       # *f32 [T, Dq]
    out_ptr,      # *f32 [H, Dq]
    H: tl.constexpr,
    T: tl.constexpr,
    Dq: tl.constexpr,
    BLOCK_D: tl.constexpr,  # e.g., 128
):
    h = tl.program_id(0)  # one program per head
    # Load attn[h, :]
    offs_t = tl.arange(0, BLOCK_T)  # BLOCK_T is 128 here
    mask_t = offs_t < T
    attn_row = tl.load(attn_ptr + h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]

    # Accumulator for out[h, :]
    out_vec = tl.zeros((Dq,), dtype=tl.float32)

    # Tile over features
    for d_start in range(0, Dq, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        # For each feature tile, compute contribution from each token t
        # out_vec[d] += sum_{t} attn_row[t] * Kc[t, d]
        for d_idx in range(0, BLOCK_D):
            d = d_start + d_idx
            # mask for d
            mask_d = d < Dq
            acc_d = tl.zeros((), dtype=tl.float32)
            # Loop t over tiles
            for t_start in range(0, T, BLOCK_T):
                offs_t_sub = t_start + tl.arange(0, BLOCK_T)
                mask_t_sub = offs_t_sub < T
                # Load Kc for this feature d across tokens in the tile
                Kc_sub = tl.load(
                    Kc_ptr + offs_t_sub * Dq + d,
                    mask=mask_t_sub,
                    other=0.0
                )  # [BLOCK_T]
                # Multiply by attn_row in the same tile
                for t_idx in range(0, BLOCK_T):
                    t = t_start + t_idx
                    mask_t_elem = t < T
                    val = tl.load(Kc_ptr + t * Dq + d, mask=mask_t_elem & mask_d, other=0.0)
                    acc_d += attn_row[t] * val
            out_vec[d] = acc_d

    # Store out_vec
    offs_out = tl.arange(0, Dq)
    tl.store(out_ptr + h * Dq + offs_out, out_vec, mask=offs_out < Dq)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Shapes
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    device = q_nope.device

    # Check fixed assertions
    assert num_qo_heads == 16, "num_qo_heads must be 16"
    assert head_dim_ckv == 512, "head_dim_ckv must be 512"
    assert head_dim_kpe == 64, "head_dim_kpe must be 64"

    # Allocate output tensors (compute in fp32)
    output = torch.empty((batch_size, 16, 512), dtype=torch.float32, device=device)
    lse = torch.empty((batch_size, 16), dtype=torch.float32, device=device)

    for b in range(batch_size):
        # Compute number of tokens for this batch element
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L_tokens <= 0:
            output[b].zero_()
            lse[b].zero_()
            continue

        # Gather Kc and Kp for this batch element
        tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.long)
        Kc_b = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, 512]
        Kp_b = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, 64]

        # qn and qp for this batch
        qn = q_nope[b].contiguous().to(torch.float32)  # [16, 512]
        qp = q_pe[b].contiguous().to(torch.float32)    # [16, 64]

        # Allocate intermediate
        logits = torch.empty((16, L_tokens), dtype=torch.float32, device=device)

        # Launch fused_logits_kernel: one program per head
        grid_logits = (16,)
        fused_logits_kernel[grid_logits](
            qn, qp, Kc_b, Kp_b, logits,
            H=16, T=L_tokens, Dq=512, Dp=64,
            BLOCK_T=128,
            num_warps=4, num_stages=2
        )

        # Softmax per head
        attn = torch.empty((16, L_tokens), dtype=torch.float32, device=device)
        grid_softmax = (16,)
        softmax_row_kernel[grid_softmax](
            logits, attn,
            H=16, T=L_tokens,
            BLOCK_T=128,
            num_warps=4, num_stages=2
        )

        # LogSumExp per head
        lse_b = torch.empty((16,), dtype=torch.float32, device=device)
        grid_lse = (16,)
        lse_row_kernel[grid_lse](
            logits, lse_b,
            H=16, T=L_tokens,
            BLOCK_T=128,
            num_warps=4, num_stages=2
        )
        lse[b] = lse_b

        # Per-head matmul: out[h, :] = attn[h, :] @ Kc_b[:, :]
        out_b = torch.empty((16, 512), dtype=torch.float32, device=device)
        grid_matmul = (16,)
        matmul_row_kernel[grid_matmul](
            attn, Kc_b, out_b,
            H=16, T=L_tokens, Dq=512,
            BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        # Store output
        output[b] = out_b

    # Cast output back to bfloat16 to match original run dtype
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA device for Triton kernels"
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
