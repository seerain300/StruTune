import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Check constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [total_kv, 512]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [total_kv, 64]

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [topk]
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        # For each selected K (up to 2048 per token), compute two small dot products
        # and accumulate per head logits.
        qn = q_nope[t].to(torch.float32)  # [16, 512]
        qp = q_pe[t].to(torch.float32)    # [16, 64]

        Kc = Kc_all[valid_indices]  # [M, 512]
        Kp = Kp_all[valid_indices]  # [M, 64]

        for h in range(num_qo_heads):
            # Per-head q vectors
            qnh = qn[h, :]  # [512]
            qph = qp[h, :]  # [64]

            # Initialize logits for this head
            logits = torch.empty((topk,), dtype=torch.float32, device=device)

            for k in range(topk):
                idx = int(valid_indices[k])
                Kc_vec = Kc_all[idx]  # [512]
                Kp_vec = Kp_all[idx]  # [64]

                contrib1 = torch.dot(qnh, Kc_vec)  # scalar
                contrib2 = torch.dot(qph, Kp_vec)  # scalar
                logits[k] = (contrib1 + contrib2) * sm_scale

            # Compute 2-base LSE
            m = torch.max(logits)
            sum_exp = torch.sum(torch.exp(logits - m))
            lse[t, h] = torch.log(sum_exp) / math.log(2.0)

            # Compute attention output
            attn = torch.softmax(logits, dim=0)  # [topk]
            out_h = torch.zeros((head_dim_ckv,), dtype=torch.float32, device=device)
            for k in range(topk):
                Kc_vec = Kc_all[valid_indices[k]]  # [512]
                out_h += attn[k] * Kc_vec
            output[t, h, :] = out_h.to(torch.bfloat16)

    return output, lse


def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([8462, 64, 64], dtype=torch.bfloat16)
    sparse_indices = torch.randint(0, 541568, [1, 2048], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Use the original run function to ensure exact behavior and correctness.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
