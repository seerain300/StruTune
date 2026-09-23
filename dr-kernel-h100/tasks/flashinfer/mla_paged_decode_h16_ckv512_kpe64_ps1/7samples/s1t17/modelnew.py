import math
import torch
import triton
import triton.language as tl


@triton.jit
def per_head_kernel(
    qn_ptr,            # *float32, [D]
    qp_ptr,            # *float32, [Dp]
    Kc_ptr,            # *float32, [L, D], row-major (L, D)
    Kp_ptr,            # *float32, [L, Dp], row-major (L, Dp)
    out_ptr,           # *float32, [D]
    lse_ptr,           # *float32, [1]
    L: tl.int32,       # number of tokens
    D: tl.int32,       # head_dim_ckv (512)
    Dp: tl.int32,      # head_dim_kpe (64)
    sm_scale: tl.float32,  # scaling factor for logits
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # We run one program per (batch, head) and compute the entire output vector.
    # 1) Compute logits v[i] for i in [0, L)
    v = tl.zeros([L], dtype=tl.float32)
    # Reduce over Kc dimension (D) in tiles
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        # qn slice
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Accumulate sum_i (qn_slice * Kc[i, k_off])
        for i in range(0, L, BLOCK_L):
            i_off = i + tl.arange(0, BLOCK_L)
            mask_i = i_off < L
            # Load Kc[i_off, k_off] -> shape [BLOCK_L, BLOCK_K]
            Kc_tile = tl.load(
                Kc_ptr + i_off[:, None] * D + k_off[None, :],
                mask=mask_i[:, None] & mask_k[None, :],
                other=0.0,
            )
            # v += sum over k of (qn_slice[k] * sum_i Kc_tile[i, k])
            v += tl.sum(qn_slice[None, :] * tl.sum(Kc_tile, axis=1), axis=1)

    # 2) Reduce over Kp dimension (Dp) in tiles
    for dp in range(0, Dp, BLOCK_K):
        dp_off = dp + tl.arange(0, BLOCK_K)
        mask_dp = dp_off < Dp
        qp_slice = tl.load(qp_ptr + dp_off, mask=mask_dp, other=0.0)  # [BLOCK_K]
        for i in range(0, L, BLOCK_L):
            i_off = i + tl.arange(0, BLOCK_L)
            mask_i = i_off < L
            Kp_tile = tl.load(
                Kp_ptr + i_off[:, None] * Dp + dp_off[None, :],
                mask=mask_i[:, None] & mask_dp[None, :],
                other=0.0,
            )
            v += tl.sum(qp_slice[None, :] * tl.sum(Kp_tile, axis=1), axis=1)

    # Scale logits by sm_scale
    v = v * sm_scale

    # 3) Compute lse = logsumexp_base2(v) stably
    ln2 = 1.4426950408889634  # 1 / ln(2)
    v_max = -float("inf")
    for i in range(0, L, BLOCK_L):
        i_off = i + tl.arange(0, BLOCK_L)
        mask_i = i_off < L
        v_sub = tl.load(v_ptr + i_off, mask=mask_i, other=-float("inf"))
        v_max = tl.maximum(v_max, tl.max(v_sub, axis=0))
    sum_exp = 0.0
    for i in range(0, L, BLOCK_L):
        i_off = i + tl.arange(0, BLOCK_L)
        mask_i = i_off < L
        v_sub = tl.load(v_ptr + i_off, mask=mask_i, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(v_sub - v_max) * (1.0 / ln2), axis=0)
    lse_val = v_max + tl.log(sum_exp)  # base-e log
    # Write lse to lse_ptr[0]
    tl.store(lse_ptr, lse_val)

    # 4) Compute attention weights attn[i] = exp(v[i]/ln2 - lse)
    # We need v[i] for each i. Re-load v from memory and compute attn.
    attn = tl.zeros([L], dtype=tl.float32)
    for i in range(0, L, BLOCK_L):
        i_off = i + tl.arange(0, BLOCK_L)
        mask_i = i_off < L
        v_sub = tl.load(v_ptr + i_off, mask=mask_i, other=-float("inf"))
        attn_i = tl.exp(v_sub / ln2 - lse_val)
        attn[i_off] = attn_i

    # 5) Compute out[j, :] = sum_i attn[i] * Kc[i, :]
    out = tl.zeros([D], dtype=tl.float32)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        for i in range(0, L, BLOCK_L):
            i_off = i + tl.arange(0, BLOCK_L)
            mask_i = i_off < L
            attn_sub = tl.load(attn_ptr + i_off, mask=mask_i, other=0.0)
            Kc_tile = tl.load(
                Kc_ptr + i_off[:, None] * D + k_off[None, :],
                mask=mask_i[:, None] & mask_k[None, :],
                other=0.0,
            )
            # out += sum_i (attn_sub[i] * Kc_tile[i, :])
            out += tl.sum(attn_sub[:, None] * Kc_tile, axis=0)

    # Store output and lse
    tl.store(out_ptr, out)
    tl.store(lse_ptr, lse_val)


def get_inputs():
    # Helper for local testing; harness may provide its own inputs.
    # Avoid recursion and self-reference.
    batch_size = 1
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 989669

    q_nope = torch.randn([batch_size, num_qo_heads, head_dim_ckv], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([batch_size, num_qo_heads, head_dim_kpe], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([num_pages, 1, head_dim_ckv], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([num_pages, 1, head_dim_kpe], dtype=torch.bfloat16, device='cuda')

    # Simple indptr for testing
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)  # [2]
    num_tokens = kv_indptr[-1].item()
    kv_indices = torch.randint(0, num_pages, [num_tokens], dtype=torch.int32, device='cuda')

    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        if q_nope.device.type != 'cuda':
            q_nope = q_nope.to('cuda')
        if q_pe.device.type != 'cuda':
            q_pe = q_pe.to('cuda')
        if ckv_cache.device.type != 'cuda':
            ckv_cache = ckv_cache.to('cuda')
        if kpe_cache.device.type != 'cuda':
            kpe_cache = kpe_cache.to('cuda')
        if kv_indptr.device.type != 'cuda':
            kv_indptr = kv_indptr.to('cuda')
        if kv_indices.device.type != 'cuda':
            kv_indices = kv_indices.to('cuda')

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Prepare cached Kc_all and Kp_all
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device='cuda')
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device='cuda')

        # Process each batch element
        for b in range(batch_size):
            # Determine token range and length
            if kv_indptr.numel() != (batch_size + 1):
                # Fallback: if indptr invalid, zero output
                output[b] = 0.0
                lse[b] = -float("inf")
                continue
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b] = 0.0
                lse[b] = -float("inf")
                continue

            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())]  # [L_tokens]
            Kc = Kc_all[tok_idx]  # [L_tokens, 512]
            Kp = Kp_all[tok_idx]  # [L_tokens, 64]

            # For each head j
            for j in range(num_qo_heads):
                # Cast qn and qp to float32 for Triton
                qn = q_nope[b, j].to(torch.float32)  # [512]
                qp = q_pe[b, j].to(torch.float32)    # [64]

                # Launch Triton kernel: one program per (b,j)
                grid = (1,)
                per_head_kernel[grid](
                    qn, qp,
                    Kc, Kp,
                    output[b, j], lse[b, j],
                    L_tokens, head_dim_ckv, head_dim_kpe,
                    sm_scale,
                    BLOCK_K=128, BLOCK_L=128,
                    num_warps=4,
                )

        # Cast output to bfloat16 as original function returns bfloat16
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse