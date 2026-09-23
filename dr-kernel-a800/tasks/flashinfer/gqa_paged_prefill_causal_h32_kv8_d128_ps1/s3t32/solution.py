import math
import torch
import triton
import triton.language as tl


# Minimal Triton kernel: cast a 1D float32 vector to bfloat16 (store bfloat16)
@triton.jit
def _cast_bf16_1d_kernel(inp_ptr, out_ptr, SIZE: tl.constexpr):
    for i in range(SIZE):
        vi = tl.load(inp_ptr + i)
        tl.store(out_ptr + i, vi.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.HEAD_DIM = 128
        self.gqa_ratio = 4  # 32 // 8

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All inputs must be on CUDA device"
        device = q.device

        # Cast inputs to float32 for compute
        q_f32 = q.to(torch.float32)  # [total_q, 32, 128]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]

        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        assert num_qo_heads == 32 and head_dim == 128, "This implementation expects num_qo_heads=32, head_dim=128"

        # Prepare outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]
        assert kv_indptr.shape[0] == len_indptr

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            num_q_tokens = q_end - q_start

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_segments_b = int(kv_end - kv_start)

            if num_q_tokens <= 0 or num_segments_b <= 0:
                continue

            # Gather kv_ids for this segment
            kv_ids = kv_indices[kv_start:kv_end].to(torch.int32).contiguous()  # [num_segments_b]

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Causal-like bound
                delta = num_segments_b - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
                if max_kv_idx <= 0:
                    for h in range(num_qo_heads):
                        lse[global_q_idx, h] = 0.0
                        output[global_q_idx, h] = torch.zeros((head_dim,), dtype=torch.bfloat16, device=device)
                    continue

                for h in range(num_qo_heads):
                    kv_head = h // self.gqa_ratio  # map to 8 KV heads

                    # Load q_vec [128]
                    q_vec = q_f32[global_q_idx, h]  # [128] float32

                    # Load K/V rows [max_kv_idx, 128]
                    k_rows = k_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]
                    v_rows = v_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]

                    # Compute logits = q_vec @ k_rows.T using PyTorch (robust and fast)
                    logits = q_vec @ k_rows.T  # [max_kv_idx] float32

                    # Scale logits
                    logits_scaled = logits * sm_scale  # [max_kv_idx]

                    # lse = logsumexp(logits_scaled) / ln(2), use torch
                    lse_val = torch.logsumexp(logits_scaled, dim=0) / math.log(2.0)
                    lse[global_q_idx, h] = lse_val

                    # attn = softmax(logits_scaled), use torch
                    attn = torch.softmax(logits_scaled, dim=0)  # [max_kv_idx]

                    # Compute output = attn @ v_rows -> [128]
                    out_vec = attn @ v_rows  # [128] float3


def run(*args):
    return ModelNew()(*args)
