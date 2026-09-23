class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.MAX_KV_TOKENS = 1024  # upper bound for kv tokens per segment; safe for demonstration

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale=None):
        # Ensure CUDA device
        assert q.is_cuda, "Inputs must be on CUDA device for Triton kernel"
        device = q.device

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32).contiguous()
        # Flatten cache to [num_pages, num_kv_heads, head_dim]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()

        total_q = q_f32.shape[0]
        num_qo_heads = self.num_qo_heads
        head_dim = self.head_dim
        num_segments = qo_indptr.numel() - 1

        # Allocate outputs: output [total_q, num_qo_heads, head_dim], lse [total_q, num_qo_heads]
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per segment
        grid = (num_segments,)
        compute_attention_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            output, lse,
            total_q, num_qo_heads, head_dim,
            k_cache_flat.shape[0], self.num_kv_heads,
            self.gqa_ratio,
            sm_scale if sm_scale is not None else self.sm_scale,
            MAX_KV_TOKENS=self.MAX_KV_TOKENS,
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
