class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        if q.device.type != "cuda":
            q = q.cuda()
        if k_cache.device.type != "cuda":
            k_cache = k_cache.cuda()
        if v_cache.device.type != "cuda":
            v_cache = v_cache.cuda()
        if qo_indptr.device.type != "cuda":
            qo_indptr = qo_indptr.cuda()
        if kv_indptr.device.type != "cuda":
            kv_indptr = kv_indptr.cuda()
        if kv_indices.device.type != "cuda":
            kv_indices = kv_indices.cuda()

        # Make inputs contiguous and flatten k_cache/v_cache by squeezing time dim
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        k_cache_flat = k_cache.squeeze(1)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1)  # [num_pages, 8, 128]

        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Shape assertions must hold."
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        len_indptr = qo_indptr.shape[0]
        B = len_indptr - 1  # number of batches

        # Allocate output and lse buffers (fp32 for computation)
        output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per batch
        grid = (B,)
        _compute_all_batches_kernel[grid](
            q, k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            output, lse,
            sm_scale,
            total_q, num_qo_heads, head_dim, gqa_ratio,
            len_indptr,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
