class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, H, D] bfloat16 on CUDA
        k_cache: [N_total, 1, num_kv_heads, D] bfloat16 (N_total = kv_indptr[-1] - 1 in provided inputs)
        v_cache: [N_total, 1, num_kv_heads, D] bfloat16
        kv_indptr: [B+1] int32
        kv_indices: [N_total] int32
        sm_scale: float32 scalar (e.g., 1/sqrt(128))
        Returns:
        - output: [B, H, D], dtype bfloat16
        - lse: [B, H], dtype float32
        """
        assert q.dtype == torch.bfloat16, "q must be bfloat16"
        assert k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16, "k_cache/v_cache must be bfloat16"
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be CUDA tensors"

        B, H, D = q.shape
        assert H == 32, "num_qo_heads must be 32"
        assert D == 128, "head_dim must be 128"
        num_kv_heads = k_cache.shape[2]  # 8 in provided inputs
        gqa_ratio = H // num_kv_heads  # 4

        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Prepare output and lse
        out = torch.empty((B, H, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device).zero_()  # initialize to zeros for accumulation safety

        # For each batch element, extract token indices and launch the Triton kernel
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            actual_num_tokens = end - start
            kv_indices_b = kv_indices[start:end].to(torch.int32)

            _forward_b_kernel[(1,)](
                q, k_cache, v_cache, kv_indices_b,
                out, lse,
                D=128, H=32, gqa_ratio=4, num_kv_heads=8,
                sm_scale=sm_scale,
                N_max=actual_num_tokens,  # Triton constexpr loop bound
                num_warps=4, num_stages=2
            )

        return out, lse


def run(*args):
    return ModelNew()(*args)
