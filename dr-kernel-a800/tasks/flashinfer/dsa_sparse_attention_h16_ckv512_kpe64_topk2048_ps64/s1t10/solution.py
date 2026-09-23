import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    # Shapes and assertions
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Constants and checks
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Flatten paged KV cache to [num_total_kv, dim] and cast to float32 for compute
    total_kv = num_pages * page_size
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32).contiguous()  # [total_kv, 512]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32).contiguous()  # [total_kv, 64]

    # Prepare output and lse
    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    # Convert q tensors to float32 for compute
    q_nope_f32 = q_nope.to(torch.float32).contiguous()
    q_pe_f32 = q_pe.to(torch.float32).contiguous()
    sparse_i32 = sparse_indices.to(torch.int32).contiguous()

    # Loop over tokens
    for t in range(num_tokens):
        indices = sparse_i32[t]  # [topk]
        valid_mask = indices != -1

        # Initialize logits for this token
        logits = torch.empty((num_qo_heads, topk), dtype=torch.float32, device=device)
        for h in range(num_qo_heads):
            # Accumulate contributions for each k
            for k in range(topk):
                idx = int(indices[k])
                valid = valid_mask[k]
                if valid:
                    # contrib1: q_nope[t, h, :] @ Kc_all[idx, :]
                    contrib1 = torch.dot(q_nope_f32[t, h, :], Kc_all[idx, :])
                    # contrib2: q_pe[t, h, :] @ Kp_all[idx, :]
                    contrib2 = torch.dot(q_pe_f32[t, h, :], Kp_all[idx, :])
                    logits[h, k] = (contrib1 + contrib2) * sm_scale
                else:
                    logits[h, k] = 0.0  # padding does not contribute

        # Compute lse = logsumexp(logits_scaled) / ln(2) per head
        ln2 = 1.0 / math.log(2.0)
        for h in range(num_qo_heads):
            m = torch.max(logits[h, :])
            sum_exp = torch.sum(torch.exp(logits[h, :] - m))
            lse[t, h] = m + torch.log(sum_exp) * ln2

        # Compute attn = softmax(logits, dim=1) per head
        for h in range(num_qo_heads):
            attn = torch.softmax(logits[h, :] - lse[t, h], dim=0)  # shape [topk]

            # Final output vector for this head: sum over k of attn_k * Kc_all[idx, :]
            out_vec = torch.zeros((head_dim_ckv,), dtype=torch.float32, device=device)
            for k in range(topk):
                idx = int(indices[k])
                valid = valid_mask[k]
                if valid:
                    kc_vec = Kc_all[idx, :]
                    out_vec += attn[k] * kc_vec
            output[t, h, :] = out_vec.to(torch.bfloat16)

    return output, lse


def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([8462, 64, 64], dtype=torch.bfloat16)
    sparse_indices = torch.randint(0, 541568, [1, 2048], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
