import math
import torch
import triton
import triton.language as tl


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors and contiguity
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA device"
        device = q.device
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)

        # Expand k and v to 32 heads (GQA expands 8 heads to 32)
        k_exp = torch.repeat_interleave(k, repeats=4, dim=1)  # [total_kv, 32, 128]
        v_exp = torch.repeat_interleave(v, repeats=4, dim=1)  # [total_kv, 32, 128]

        total_q = q.shape[0]
        total_kv = k.shape[0]
        NUM_SEGMENTS = qo_indptr.numel() - 1

        # Allocate output and lse
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (i, h)
        grid = (total_q * 32,)
        attention_per_ih_kernel[grid](
            q, k_exp, v_exp, output, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
        )

        # Return output in bfloat16 and lse in float32 (base-2)
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
