import math
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    Q_ptr,  # *fp32, shape [B, 32, D]
    K_ptr,  # *fp32, shape [T, D]
    V_ptr,  # *fp32, shape [T, D]
    out_ptr,  # *fp32, shape [B, 32, D]
    lse_ptr,  # *fp32, shape [B, 32]
    num_tokens_b: tl.int32,
    sm_scale: tl.float32,
    D: tl.int32,
    H: tl.int32,  # num_kv_heads, not used for loading but kept for signature consistency
):
    # One program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Numerically stable logsumexp over scaled logits
    running_max = -float("inf")
    running_sum = 0.0

    # First pass: compute lse
    for t in range(0, num_tokens_b):
        # Load q_vec = Q[b, h]
        q_base = Q_ptr + b * 3 * D + h * D  # strides: [B, 32, D] -> stride0=3*D, stride1=D, stride2=1
        q_vec = tl.load(q_base + tl.arange(0, D))

        # Load k_vec and v_vec for this token and kv_head mapping (GQA: kv_head = h // 4)
        kv_head = h // (32 // 8)  # gqa ratio = 4
        k_base = K_ptr + t * D
        k_vec = tl.load(k_base + tl.arange(0, D))
        v_base = V_ptr + t * D
        v_vec = tl.load(v_base + tl.arange(0, D))

        # Dot product
        dot_val = tl.sum(q_vec * k_vec, axis=0)

        # Scaled logits
        scaled = dot_val * sm_scale
        # Update running_max and running_sum (stable logsumexp)
        new_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.exp(scaled - new_max)
        running_max = new_max

    # Compute lse = log(running_sum) + running_max, then divide by ln(2)
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * 1.0 / 1.44269504  # 1 / ln(2)

    # Store lse
    tl.store(lse_ptr + b * 32 + h, lse_val)

    # Second pass: compute output vector
    for t in range(0, num_tokens_b):
        kv_head = h // (32 // 8)
        q_base = Q_ptr + b * 3 * D + h * D
        q_vec = tl.load(q_base + tl.arange(0, D))

        k_base = K_ptr + t * D
        k_vec = tl.load(k_base + tl.arange(0, D))
        v_base = V_ptr + t * D
        v_vec = tl.load(v_base + tl.arange(0, D))

        dot_val = tl.sum(q_vec * k_vec, axis=0)
        scaled = dot_val * sm_scale
        attn = tl.exp(scaled - lse_val)  # softmax over scaled logits

        out_base = out_ptr + b * 32 * D + h * D
        # out[b, h, :] += attn * v_vec
        out_vec = tl.load(out_base + tl.arange(0, D))
        out_vec = out_vec + attn * v_vec
        tl.store(out_base + tl.arange(0, D), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is computed in Triton

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, 32, 128], bfloat16 or float32
        k_cache: [N, 8, 128], bfloat16 or float32 (evaluation axes show H=8, D=128)
        v_cache: [N, 8, 128], bfloat16 or float32
        kv_indptr: [len_indptr], int32, with len_indptr = B + 1
        kv_indices: [num_kv_indices], int32
        sm_scale: float32 scalar, e.g., 1.0 / sqrt(128)
        Returns:
        - output: [B, 32, 128], bfloat16
        - lse: [B, 32], float32
        """
        B = q.shape[0]
        num_qo_heads = 32
        D = 128
        H = 8  # from axes; evaluation uses 8 kv heads

        # Prepare outputs
        output = torch.empty((B, num_qo_heads, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Compute num_tokens per batch element: num_tokens_b = kv_indptr[b+1] - kv_indptr[b]
        # kv_indptr shape [B+1]
        num_tokens_b_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b_list.append(end - start)
        # Now gather K_t and V_t for each batch b
        # Note: k_cache/v_cache are [N, 8, 128]; we use kv_indices[start:start+num_tokens_b] for each b
        for b in range(B):
            num_tokens_b = num_tokens_b_list[b]
            start = int(kv_indptr[b].item())
            indices_b = kv_indices[start:start + num_tokens_b]  # [num_tokens_b], int32

            # Gather K_t and V_t: shape becomes [num_tokens_b, 8, 128], but we'll reduce to [num_tokens_b, 128] per head
            # The evaluation axes show k_cache/v_cache [N, 8, 128]; torch.gather supports gather along dim 0:
            # K_t = k_cache[indices_b, :, :] -> [num_tokens_b, 8, 128]
            K_t = k_cache[indices_b, :, :].to(torch.float32).contiguous()  # [T, 8, 128]
            V_t = v_cache[indices_b, :, :].to(torch.float32).contiguous()  # [T, 8, 128]

            # We need per-token 128-d vectors for k and v for kv_head mapping. Reduce by selecting the corresponding head.
            # GQA mapping: kv_head = h // 4, but here we reduce for each token. Since GQA uses a single kv head per token,
            # K_t[:, :, :] are already per-token. We'll pass K_t and V_t and slice inside the kernel by computing kv_head from h.
            # However, Triton kernel expects [T, D] for k/v. We create K_t_flat and V_t_flat by selecting the kv_head for each
            # token: we cannot do that on host easily without creating per-token tensors. Instead, we pass K_t and V_t as is
            # and inside kernel compute per token the desired kv_head slice. To do that robustly, we will flatten to [T, D]
            # by taking K_t[:, 0, :] if we had a size-1, but k_cache has H=8. Therefore, we'll create K_t_flat and V_t_flat
            # by concatenating all heads for each token. But that would be 8x128 per token and complicate kernel. Simpler:
            # In the kernel, for each t, we compute kv_head = h // 4, then fetch K_t[t, kv_head, :] and V_t[t, kv_head, :].
            # To support that, we pass K_t and V_t as [T, H, D], and load per t using kv_head index. torch.gather gave [T, 8, 128]
            # which is exactly what we want: K_t[t, h2, :] for h2 in 0..7.

            # Launch Triton kernel: one program per (b, h)
            grid = (B, num_qo_heads)
            softmax_and_attention_single_bh[grid](
                q[b].to(torch.float32).contiguous(),           # Q_ptr: [32, 128] for this batch
                K_t,                                           # [T, 8, 128]
                V_t,                                           # [T, 8, 128]
                output,                                        # [B, 32, 128]
                lse,                                          # [B, 32]
                num_tokens_b,                                 # int32
                sm_scale,                                     # float32
                D,                                            # int32
                H,                                            # int32 (num_kv_heads)
                num_warps=4,
            )

        # Cast output to bfloat16 to match original return dtype
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
