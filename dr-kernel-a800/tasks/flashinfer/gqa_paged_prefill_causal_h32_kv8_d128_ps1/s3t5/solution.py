import math
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, _, num_kv_heads, _ = k_cache.shape
    len_indptr = qo_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Constants
    assert num_qo_heads == 32, "num_qo_heads must be 32"
    assert head_dim == 128, "head_dim must be 128"
    assert num_kv_heads == 8, "num_kv_heads must be 8"

    # Output tensors
    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

    # Convert inputs to float32 for stable compute (original q is bfloat16)
    q_f32 = q.to(torch.float32).contiguous()
    k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
    v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]

    # Loop over batch segments
    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            continue

        num_kv_tokens = kv_end - kv_start  # number of cached groups used in this segment
        if num_kv_tokens <= 0:
            continue

        # Fetch K/V groups for this batch
        k_batch = k_cache_flat[kv_start:kv_end]   # [num_kv_tokens, 8, 128]
        v_batch = v_cache_flat[kv_start:kv_end]   # [num_kv_tokens, 8, 128]

        # Queries for this segment
        q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]
        num_q_tokens = q_batch.shape[0]

        # For each query token
        for q_idx in range(num_q_tokens):
            global_q_idx = q_start + q_idx

            # Causal-like limit: max_kv_idx = min(q_idx + 1 + (num_kv_tokens - num_q_tokens), num_kv_tokens)
            max_kv_idx = min(q_idx + 1 + (num_kv_tokens - num_q_tokens), num_kv_tokens)
            if max_kv_idx <= 0:
                continue

            # For each query head h
            for h in range(num_qo_heads):
                kv_head = h // (num_qo_heads // num_kv_heads)  # h // 4

                # Select q_head vector (128-dim)
                q_pos = q_batch[q_idx]  # [32, 128], float32
                q_head = q_pos[h].contiguous()  # [128], float32

                # Select k_head and v_head (first max_kv_idx rows)
                k_group = k_batch[:max_kv_idx, kv_head]  # [max_kv_idx, 128], float32
                v_group = v_batch[:max_kv_idx, kv_head]  # [max_kv_idx, 128], float32

                # Compute logits_scaled = q @ k.T
                logits = torch.matmul(q_head, k_group.t())  # [max_kv_idx], float32
                logits_scaled = logits * sm_scale

                # Compute lse per (query, head): logsumexp(logits_scaled) / ln(2)
                lse[global_q_idx, h] = torch.logsumexp(logits_scaled, dim=0) / math.log(2.0)

                # Compute attention weights: softmax over logits_scaled
                attn = torch.softmax(logits_scaled, dim=0)  # [max_kv_idx]

                # Compute final output: out = v_group @ attn (since attn is [max_kv_idx], and v_group is [max_kv_idx, 128])
                # Note: This is a reduction over the K dimension to produce [128].
                out_vec = torch.matmul(v_group, attn)  # [128], float32

                # Store output (cast to bfloat16 to match original)
                output[global_q_idx, h] = out_vec.to(torch.bfloat16)

    return output, lse


def get_inputs():
    # Same as original; device will be set by caller; ensure inputs are on GPU before calling run.
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    # Example indices for testing; the evaluation will provide real ones.
    n = 1; t = 1
    lens = torch.full((n,), t // n, dtype=torch.int32)
    lens[: t % n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(lens, 0)]).to(torch.int32).to('cuda')
    n2 = 1; t2 = 34
    lens2 = torch.full((n2,), t2 // n2, dtype=torch.int32)
    lens2[: t2 % n2] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(lens2, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 51, [34], dtype=torch.int32).to('cuda')
    sm_scale = 1.0 / math.sqrt(128.0)  # float32 scalar
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Run the provided PyTorch implementation. Assumes inputs are on CUDA and dtype conversions handled in run.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
