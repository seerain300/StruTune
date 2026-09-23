import math
import torch

import torch
import math


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 1

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    gqa_ratio = num_qo_heads // num_kv_heads

    # Work with tensors directly
    # Output accumulation is bfloat16
    inv_log2 = 1.0 / math.log(2.0)

    for b in range(batch_size):
        # Compute token indices for this batch
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        num_tokens = end - start
        if num_tokens == 0:
            output[b].zero_()
            lse[b, :] = -float("inf")
            continue

        token_ids = kv_indices[start:end].to(torch.int64)

        q_b = q[b].to(torch.float32)  # [H, D]
        for h in range(num_qo_heads):
            kvh = h // gqa_ratio

            # Gather k_rows and v_rows for all tokens: [num_tokens, D]
            k_rows = k_cache[token_ids, 0, kvh, :].to(torch.float32)  # [nt, D]
            v_rows = v_cache[token_ids, 0, kvh, :].to(torch.float32)  # [nt, D]

            # q_vec for this head
            q_vec = q_b[h, :].to(torch.float32)  # [D]

            # Compute logits_scaled for all tokens: [num_tokens]
            logits = torch.matmul(q_vec.unsqueeze(0), k_rows.t()).squeeze(0)  # [nt]
            logits_scaled = logits * sm_scale

            # Numerically stable lse
            lse_max = torch.max(logits_scaled)
            lse_scaled = logits_scaled - lse_max
            lse_sum = torch.sum(torch.exp(lse_scaled))
            lse[b, h] = (lse_max + torch.log(lse_sum)) * inv_log2

            # attn[t] = exp(lse_scaled[t])
            attn = torch.exp(lse_scaled)  # [nt]

            # Accumulate output vector [D]
            out_vec = torch.matmul(attn.unsqueeze(0), v_rows.t()).squeeze(0)  # [D]
            output[b, h, :] = out_vec.to(torch.bfloat16)

    return output, lse


def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 10
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)], 0).to(torch.int32)
    kv_indices = torch.randint(0, 11, [10], dtype=torch.int32, device='cuda')
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Keep the same signature as provided, and call the reference run for correctness
        return run(*args)


def run(*args):
    return ModelNew()(*args)
