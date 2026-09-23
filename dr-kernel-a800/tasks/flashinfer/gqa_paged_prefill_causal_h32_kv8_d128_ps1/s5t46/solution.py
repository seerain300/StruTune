import torch

class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "Inputs must be CUDA tensors"
        # Constants
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        GQA_RATIO = num_qo_heads // num_kv_heads  # 4
        # Shapes
        total_q = q.shape[0]
        device = q.device
        # Cast to float32 for compute
        q_f32 = q.to(torch.float32)
        # Squeeze the (1,) dimension from k/v caches
        k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        # Allocate outputs
        out = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)
        # Prepare flattened indptr and kv_indices (Triton expects int32)
        qo_indptr_i32 = qo_indptr.to(torch.int32)
        kv_indptr_i32 = kv_indptr.to(torch.int32)
        kv_indices_i32 = kv_indices.to(torch.int32)
        # Launch Triton kernel: grid over (b, q_idx, h)
        MAX_Q = 4096
        MAX_KV = 256
        grid = (len_indptr := qo_indptr_i32.numel(), total_q, num_qo_heads)
        _attention_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat,
            qo_indptr_i32, kv_indptr_i32, kv_indices_i32,
            out, lse,
            GQA_RATIO, MAX_Q, MAX_KV, head_dim, sm_scale,
        )
        # Cast output to bfloat16 to match original
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
