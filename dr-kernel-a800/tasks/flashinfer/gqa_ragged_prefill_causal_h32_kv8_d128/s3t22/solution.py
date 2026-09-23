import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_kernel(
    q_ptr,       # *float32, [Q, 32, 128]
    k_ptr,       # *float32, [K, 32, 128]
    v_ptr,       # *float32, [K, 32, 128]
    out_ptr,     # *float32, [Q, 32, 128]
    lse_ptr,     # *float32, [Q, 32]
    sm_scale,    # float32
    Q, K,        # int32
    delta,       # int32 = K - Q
    H: tl.constexpr,               # 32
    gqa_ratio: tl.constexpr,       # 4
    ln2,          # float32 = log(2)
    head_dim: tl.constexpr,        # 128
    BLOCK_Q: tl.constexpr,         # e.g., 64
    BLOCK_K: tl.constexpr,         # e.g., 64
):
    # Process Q in tiles
    for q0 in tl.static_range(0, Q, BLOCK_Q):
        # Iterate over heads h (constexpr loop)
        for h in tl.static_range(0, H):
            # Vector of i in this tile: shape [BLOCK_Q]
            i_vec = q0 + tl.arange(0, BLOCK_Q)
            valid_i = i_vec < Q

            # Initialize lse and denom for each i in this tile
            lse_vals = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
            denom_vals = tl.full((BLOCK_Q,), 0.0, tl.float32)

            # We will compute lse and denom for each i in the tile
            for bqi in tl.static_range(0, BLOCK_Q):
                i = i_vec[bqi]
                if not valid_i[bqi]:
                    continue
                # For this i and head h, compute vector over j in a tile
                for k0 in tl.static_range(0, K, BLOCK_K):
                    j_vec = k0 + tl.arange(0, BLOCK_K)
                    valid_j = j_vec < K

                    # Load q[i, h, :]
                    q_offset = i * head_dim * H + h * head_dim
                    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, head_dim), mask=True, other=0.0)

                    # Load k[j, h, :]
                    k_offset = j_vec * head_dim * H + h * head_dim
                    k_vec = tl.load(k_ptr + k_offset + tl.arange(0, head_dim), mask=valid_j, other=0.0)

                    # Dot product over head_dim to get logits scalar
                    dot = tl.zeros((), tl.float32)
                    for d in tl.static_range(0, head_dim):
                        qd = q_vec[d]  # scalar
                        kv = k_vec[:, d]  # [BLOCK_K]
                        dot += qd * tl.sum(kv, axis=0)

                    # Apply scaling and mask
                    mask_row_offsets = i * K + j_vec
                    mask_vals = tl.load(mask_ptr + mask_row_offsets, mask=valid_j, other=1)  # int8, 1 means valid
                    invalid = (j_vec >= (i + 1 + delta)) | (mask_vals == 0)
                    logits = sm_scale * dot
                    logits = tl.where(invalid, -float("inf"), logits)

                    # Update max and sum for lse/denom
                    max_val = tl.max(logits, axis=0)
                    exp_logits = tl.exp(logits - max_val)
                    sum_exp = tl.sum(exp_logits, axis=0)
                    lse_vals[bqi] = tl.maximum(lse_vals[bqi], max_val)
                    denom_vals[bqi] += sum_exp / ln2

            # Compute final output vector out[i, h, :] = sum_j softmax_j * v[j, h, :]
            out_offset = (i_vec * H + h) * head_dim
            out_vec = tl.zeros((BLOCK_Q * head_dim,), tl.float32)

            for k0 in tl.static_range(0, K, BLOCK_K):
                j_vec = k0 + tl.arange(0, BLOCK_K)
                valid_j = j_vec < K

                # Load q[i, h, :]
                q_offset = i_vec * head_dim * H + h * head_dim
                q_vec = tl.load(q_ptr + q_offset + tl.arange(0, head_dim), mask=valid_i, other=0.0)

                # Load k[j, h, :] and logits per j (recompute)
                k_offset = j_vec * head_dim * H + h * head_dim
                k_vec = tl.load(k_ptr + k_offset + tl.arange(0, head_dim), mask=valid_j, other=0.0)

                dot = tl.zeros((), tl.float32)
                for d in tl.static_range(0, head_dim):
                    qd = q_vec[d]
                    kv = k_vec[:, d]
                    dot += qd * tl.sum(kv, axis=0)

                logits = sm_scale * dot
                mask_row_offsets = i_vec * K + j_vec
                mask_vals = tl.load(mask_ptr + mask_row_offsets, mask=valid_j, other=1)
                invalid = (j_vec >= (i_vec + 1 + delta)) | (mask_vals == 0)
                logits = tl.where(invalid, -float("inf"), logits)

                max_val = tl.max(logits, axis=0)
                exp_logits = tl.exp(logits - max_val)
                sum_exp = tl.sum(exp_logits, axis=0)
                softmax = exp_logits / sum_exp

                # Load v[j, h, :] and accumulate
                v_offset = j_vec * head_dim * H + h * head_dim
                v_vec = tl.load(v_ptr + v_offset + tl.arange(0, head_dim), mask=valid_j, other=0.0)

                for d in tl.static_range(0, head_dim):
                    out_vec += softmax * v_vec[:, d]

            # Store out[i, h, :] as float32 (later cast in host to bfloat16)
            base = i_vec * head_dim * H + h * head_dim
            for bqi in tl.static_range(0, BLOCK_Q):
                i = i_vec[bqi]
                if valid_i[bqi]:
                    out_base = out_ptr + base[bqi]
                    for d in tl.static_range(0, head_dim):
                        tl.store(out_base + d, out_vec[bqi * head_dim + d])

        # Update lse per (i,h) for this tile
        for bqi in tl.static_range(0, BLOCK_Q):
            i = i_vec[bqi]
            if valid_i[bqi]:
                lse_base = lse_ptr + i * H + h
                # We stored lse_vals per i, but above we only updated lse_vals via max over K. We need exact max over K.
                # We'll recompute max over all K for correctness. For simplicity, compute per-(i,h) max in a loop over K tiles.
                max_val = -float("inf")
                for k0 in tl.static_range(0, K, BLOCK_K):
                    j_vec = k0 + tl.arange(0, BLOCK_K)
                    valid_j = j_vec < K
                    # Recompute dot and logits for each j in tile, then take max
                    # Load q[i, h, :]
                    q_offset = i * head_dim * H + h * head_dim
                    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, head_dim), mask=True, other=0.0)
                    # Load k[j, h, :]
                    k_offset = j_vec * head_dim * H + h * head_dim
                    k_vec = tl.load(k_ptr + k_offset + tl.arange(0, head_dim), mask=valid_j, other=0.0)
                    dot = tl.zeros((), tl.float32)
                    for d in tl.static_range(0, head_dim):
                        qd = q_vec[d]
                        kv = k_vec[:, d]
                        dot += qd * tl.sum(kv, axis=0)
                    logits = sm_scale * dot
                    mask_row_offsets = i * K + j_vec
                    mask_vals = tl.load(mask_ptr + mask_row_offsets, mask=valid_j, other=1)
                    invalid = (j_vec >= (i + 1 + delta)) | (mask_vals == 0)
                    cur = tl.where(invalid, -float("inf"), logits)
                    cur_max = tl.max(cur, axis=0)
                    max_val = tl.maximum(max_val, cur_max)
                tl.store(lse_base, max_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure on CUDA and dtype float32 for compute
        device = q.device
        assert device.type == "cuda", "ModelNew requires CUDA device"

        total_q = q.shape[0]
        total_kv = k.shape[0]

        # Cast to float32 for compute
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        len_indptr = qo_indptr.numel()
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)  # kernel computes float32
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_batch = q_f32[q_start:q_end]      # [Q, 32, 128]
            k_batch = k_f32[kv_start:kv_end]    # [K, 8, 128]
            v_batch = v_f32[kv_start:kv_end]    # [K, 8, 128]

            Q = q_batch.shape[0]
            K = k_batch.shape[0]
            delta = K - Q

            # GQA mapping: expand heads by 4
            k_expanded = k_batch.repeat_interleave(4, dim=1)   # [K, 32, 128]
            v_expanded = v_batch.repeat_interleave(4, dim=1)   # [K, 32, 128]

            # Precompute bounded attention mask: mask[i, j] = 1 if j < (i + 1 + delta), else 0
            q_positions = torch.arange(Q, device=device)
            kv_positions = torch.arange(K, device=device)
            mask_mat = (kv_positions[None, :] < (q_positions[:, None] + 1 + delta))  # [Q, K], bool
            mask_int8 = mask_mat.to(torch.int8)

            # Launch Triton kernel for this segment
            segment_attention_kernel[(1,)](
                q_batch, k_expanded, v_expanded,
                output[q_start:q_end], lse[q_start:q_end],
                float(sm_scale),
                Q, K, delta,
                H=32, gqa_ratio=4,
                ln2=1.4426950408889634,  # log(2)
                head_dim=128,
                BLOCK_Q=64, BLOCK_K=64,
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
