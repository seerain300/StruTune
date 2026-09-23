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
    D: tl.int32,       # head_dim_ckv
    Dp: tl.int32,      # head_dim_kpe
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # One program handles the entire vector for this head and batch
    # Compute v[i] for all i in [0, L), then reduce to lse, then compute attn[i] and final output.
    # We'll iterate in tiles of BLOCK_L over i (token index) and BLOCK_K over feature reduction.

    # 1) Compute max over v for numerical stability
    v_max = -float("inf")
    for l0 in range(0, L, BLOCK_L):
        offs_i = l0 + tl.arange(0, BLOCK_L)
        mask_i = offs_i < L
        sum1 = 0.0
        sum2 = 0.0
        # Reduce over Kc dimension (D)
        for k in range(0, D, BLOCK_K):
            k_off = k + tl.arange(0, BLOCK_K)
            mask_k = k_off < D
            qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
            Kc_ptr_tile = Kc_ptr + offs_i[:, None] * D + k_off[None, :]
            kc_tile = tl.load(Kc_ptr_tile, mask=mask_i[:, None] & mask_k[None, :], other=0.0)
            sum1 += tl.sum(qn_slice[None, :] * kc_tile, axis=1)  # [BLOCK_L]
        # Reduce over Kp dimension (Dp)
        for kp in range(0, Dp, BLOCK_K):
            kp_off = kp + tl.arange(0, BLOCK_K)
            mask_kp = kp_off < Dp
            qp_slice = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)  # [BLOCK_K]
            Kp_ptr_tile = Kp_ptr + offs_i[:, None] * Dp + kp_off[None, :]
            kp_tile = tl.load(Kp_ptr_tile, mask=mask_i[:, None] & mask_kp[None, :], other=0.0)
            sum2 += tl.sum(qp_slice[None, :] * kp_tile, axis=1)  # [BLOCK_L]
        v_tile = sum1 + sum2
        # Apply mask for valid i
        v_tile = tl.where(mask_i, v_tile, -float("inf"))
        # Tile-wise max
        v_max = tl.maximum(v_max, tl.max(v_tile, axis=0))
    # 2) Compute sum(exp(v / ln(2) - v_max)) * ln(2)
    lse_sum = 0.0
    for l0 in range(0, L, BLOCK_L):
        offs_i = l0 + tl.arange(0, BLOCK_L)
        mask_i = offs_i < L
        sum1 = 0.0
        sum2 = 0.0
        for k in range(0, D, BLOCK_K):
            k_off = k + tl.arange(0, BLOCK_K)
            mask_k = k_off < D
            qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)
            Kc_ptr_tile = Kc_ptr + offs_i[:, None] * D + k_off[None, :]
            kc_tile = tl.load(Kc_ptr_tile, mask=mask_i[:, None] & mask_k[None, :], other=0.0)
            sum1 += tl.sum(qn_slice[None, :] * kc_tile, axis=1)
        for kp in range(0, Dp, BLOCK_K):
            kp_off = kp + tl.arange(0, BLOCK_K)
            mask_kp = kp_off < Dp
            qp_slice = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)
            Kp_ptr_tile = Kp_ptr + offs_i[:, None] * Dp + kp_off[None, :]
            kp_tile = tl.load(Kp_ptr_tile, mask=mask_i[:, None] & mask_kp[None, :], other=0.0)
            sum2 += tl.sum(qp_slice[None, :] * kp_tile, axis=1)
        v_tile = sum1 + sum2
        v_tile = tl.where(mask_i, v_tile, -float("inf"))
        exp_tile = tl.exp((v_tile - v_max) * (1.0 / math.log(2.0)))  # pass 1/log(2) from host
        lse_sum += tl.sum(exp_tile, axis=0)
    lse_val = v_max + math.log(lse_sum)  # logsumexp_base_2(v) = v_max + log(sum(exp(v - v_max)/ln2) * ln2), but here sum scaled, so lse = v_max + log(lse_sum)

    # 3) Write lse for this head
    tl.store(lse_ptr, lse_val)

    # 4) Compute attn[i] and final output vector out[:] = sum_i attn[i] * Kc[i, :]
    inv_ln2 = 1.0 / math.log(2.0)
    for l0 in range(0, L, BLOCK_L):
        offs_i = l0 + tl.arange(0, BLOCK_L)
        mask_i = offs_i < L
        sum1 = 0.0
        sum2 = 0.0
        for k in range(0, D, BLOCK_K):
            k_off = k + tl.arange(0, BLOCK_K)
            mask_k = k_off < D
            qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)
            Kc_ptr_tile = Kc_ptr + offs_i[:, None] * D + k_off[None, :]
            kc_tile = tl.load(Kc_ptr_tile, mask=mask_i[:, None] & mask_k[None, :], other=0.0)
            sum1 += tl.sum(qn_slice[None, :] * kc_tile, axis=1)
        for kp in range(0, Dp, BLOCK_K):
            kp_off = kp + tl.arange(0, BLOCK_K)
            mask_kp = kp_off < Dp
            qp_slice = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)
            Kp_ptr_tile = Kp_ptr + offs_i[:, None] * Dp + kp_off[None, :]
            kp_tile = tl.load(Kp_ptr_tile, mask=mask_i[:, None] & mask_kp[None, :], other=0.0)
            sum2 += tl.sum(qp_slice[None, :] * kp_tile, axis=1)
        v_tile = sum1 + sum2
        v_tile = tl.where(mask_i, v_tile, -float("inf"))
        attn_tile = tl.exp((v_tile - lse_val) * inv_ln2)  # base-2 normalization with lse
        # Accumulate output: out[h] += sum_i attn[i] * Kc[i, :]
        for k in range(0, D, BLOCK_K):
            k_off = k + tl.arange(0, BLOCK_K)
            mask_k = k_off < D
            # Kc column vector for this k
            Kc_col = tl.load(Kc_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
            # For each i in tile, multiply attn[i] and accumulate over k
            for l in range(0, BLOCK_L):
                i_idx = l0 + l
                if i_idx < L:
                    attn_i = attn_tile[l]  # scalar
                    out_ptr_k = out_ptr + k_off
                    out_partial = attn_i * Kc_col  # [BLOCK_K]
                    # tl.atomic_add does not exist; we do elementwise accumulation
                    # Since we write all elements in one go below, this loop just accumulates per k. Simplify by writing directly.
                    # Instead, we'll compute acc_k = sum_{offs_i} attn[offs_i] * Kc[offs_i, k]
                    # We need to compute acc_k across all i. For simplicity, we can recompute below.
                    pass
        # Directly accumulate per k using attn_tile
        # For each k, we need sum_i attn_i * Kc[i,k]. We recompute using loaded Kc_col
        # Implement accumulation over i tiles:
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for l0_acc in range(0, L, BLOCK_L):
            offs_i = l0_acc + tl.arange(0, BLOCK_L)
            mask_i = offs_i < L
            sum1 = 0.0
            sum2 = 0.0
            for k_acc in range(0, D, BLOCK_K):
                k_off_acc = k_acc + tl.arange(0, BLOCK_K)
                mask_k_acc = k_off_acc < D
                qn_slice_acc = tl.load(qn_ptr + k_off_acc, mask=mask_k_acc, other=0.0)
                Kc_ptr_tile_acc = Kc_ptr + offs_i[:, None] * D + k_off_acc[None, :]
                kc_tile_acc = tl.load(Kc_ptr_tile_acc, mask=mask_i[:, None] & mask_k_acc[None, :], other=0.0)
                sum1 += tl.sum(qn_slice_acc[None, :] * kc_tile_acc, axis=1)
            for kp_acc in range(0, Dp, BLOCK_K):
                kp_off_acc = kp_acc + tl.arange(0, BLOCK_K)
                mask_kp_acc = kp_off_acc < Dp
                qp_slice_acc = tl.load(qp_ptr + kp_off_acc, mask=mask_kp_acc, other=0.0)
                Kp_ptr_tile_acc = Kp_ptr + offs_i[:, None] * Dp + kp_off_acc[None, :]
                kp_tile_acc = tl.load(Kp_ptr_tile_acc, mask=mask_i[:, None] & mask_kp_acc[None, :], other=0.0)
                sum2 += tl.sum(qp_slice_acc[None, :] * kp_tile_acc, axis=1)
            v_acc = sum1 + sum2
            v_acc = tl.where(mask_i, v_acc, -float("inf"))
            attn_acc = tl.exp((v_acc - lse_val) * inv_ln2)
            for l_acc in range(0, BLOCK_L):
                i_idx = l0_acc + l_acc
                if i_idx < L:
                    for k_acc2 in range(0, D, BLOCK_K):
                        k_off_acc2 = k_acc2 + tl.arange(0, BLOCK_K)
                        mask_k_acc2 = k_off_acc2 < D
                        Kc_col_acc = tl.load(Kc_ptr + i_idx * D + k_off_acc2, mask=mask_k_acc2, other=0.0)
                        attn_i = attn_acc[l_acc]
                        acc += attn_i * Kc_col_acc
        # Store out[h, :]
        for k_out in range(0, D, BLOCK_K):
            k_off = k_out + tl.arange(0, BLOCK_K)
            mask_k = k_off < D
            # acc is [BLOCK_K]; out_ptr points to out[h, k_off]
            tl.store(out_ptr + k_off, acc[k_out:], mask=mask_k)


def get_inputs():
    # Helper for local testing; harness may provide its own inputs.
    axes = {
        "batch_size": 1,
        "num_pages": 989669,
        "len_indptr": 2,
        "num_kv_indices": 8,
    }
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda').to(torch.float32)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda').to(torch.float32)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda').to(torch.float32)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda').to(torch.float32)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)  # [2]
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and float32 for compute
        device = q_nope.device
        q_nope_f32 = q_nope.to(torch.float32).contiguous()
        q_pe_f32 = q_pe.to(torch.float32).contiguous()
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

        batch_size = q_nope_f32.shape[0]
        num_qo_heads = q_nope_f32.shape[1]
        D = q_nope_f32.shape[2]  # 512
        Dp = q_pe_f32.shape[2]   # 64

        # Prepare output and lse buffers
        output = torch.zeros((batch_size, num_qo_heads, D), dtype=torch.float32, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # For each batch element
        for b in range(batch_size):
            # Determine token range from kv_indptr
            if kv_indptr.numel() != (batch_size + 1):
                # Fallback to zero output if indptr malformed
                output[b] = 0.0
                lse[b] = -float("inf")
                continue
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = max(0, page_end - page_beg)
            if L == 0:
                output[b] = 0.0
                lse[b] = -float("inf")
                continue

            # Gather tokens
            tok_idx = kv_indices[page_beg:page_end]  # [L]
            Kc = Kc_all[tok_idx]  # [L, 512], contiguous
            Kp = Kp_all[tok_idx]  # [L, 64], contiguous

            # One Triton program per head; grid size = num_qo_heads
            grid = (num_qo_heads,)
            per_head_kernel[grid](
                q_nope_f32[b],                      # qn_ptr: [D]
                q_pe_f32[b],                       # qp_ptr: [Dp]
                Kc,                                 # Kc_ptr: [L, D]
                Kp,                                 # Kp_ptr: [L, Dp]
                output[b],                          # out_ptr: [D]
                lse[b],                             # lse_ptr: [1]
                L, D, Dp,
                BLOCK_K=128, BLOCK_L=128,
                sm_scale=sm_scale,
            )

        # Cast output to bfloat16 as original function returns bfloat16 output
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
