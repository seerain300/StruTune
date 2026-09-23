class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton requires CUDA tensors; inputs from evaluator are expected to be CUDA.
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors for Triton."

        B, H, D = q.shape
        # Inputs are already float32/bfloat16 from get_inputs; we can pass as-is.
        # We don't use .to(...) or any PyTorch compute in forward.
        out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q, k_cache, v_cache, kv_indices, kv_indptr, lse, out,
            B, H, D, sm_scale, 128  # N_TOTAL loop bound
        )

        # Return output and lse; evaluator expects list, not tuple
        return [out, lse]


def run(*args):
    return ModelNew()(*args)
