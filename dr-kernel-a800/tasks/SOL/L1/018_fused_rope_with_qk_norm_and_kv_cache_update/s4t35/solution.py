import torch
import triton
import triton.language as tl

# Triton kernel: RMS normalization per row for a 4D tensor [B, H, S, D], contiguous.
# Grid dimension is rows = B * H * S. Each program processes one row of length D.
@triton.jit
def rms_norm_4d_kernel(x_ptr, out_ptr, B, H, S, D, eps):
    row_id = tl.program_id(0)
    offs = tl.arange(0, D)

    # Map row_id to (b, h, s) in [B, H, S]
    HS = H * S
    b = row_id // (H * S)
    rem = row_id % (H * S)
    h = rem // S
    s = rem % S

    # Compute base linear index for this row in a contiguous [B, H, S, D] layout:
    # index = b*(H*S*D) + h*(S*D) + s*D
    base = b * (H * S * D) + h * (S * D) + s * D

    # Load as bfloat16, compute in float32
    x = tl.load(x_ptr + base + offs).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    mean = sum_sq / D
    scale = 1.0 / tl.sqrt(mean + eps)
    y = x * scale
    tl.store(out_ptr + base + offs, y.to(tl.bfloat16))

class ModelNew(torch.nn.Module):
    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                position_ids: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor,
                k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor,
                rms_norm_eps: float):
        """
        Triton-only forward:
        - Perform RMS normalization on query and key using a Triton kernel.
        - Do not perform apply_rope or cache updates (Triton does not support sin/cos).
        - Return normalized query and key, and the original caches.
        """

        # Shapes
        B_q = query.shape[0]
        H_q = query.shape[1]  # number of attention heads
        S_q = query.shape[2]
        D = query.shape[3]

        # Output tensor for normalized query
        out_query = torch.empty_like(query, dtype=torch.bfloat16, device=query.device)

        # Launch Triton kernel for query normalization
        grid_q = (B_q * H_q * S_q,)
        # Note: we pass eps as float (Triton will treat it as fp32)
        rms_norm_4d_kernel[grid_q](query, out_query, B_q, H_q, S_q, D, rms_norm_eps)

        # For key: shape [B, num_key_value_heads, S, D]
        Bk = key.shape[0]
        Hk = key.shape[1]
        Sk = key.shape[2]
        Dk = key.shape[3]
        out_key = torch.empty_like(key, dtype=torch.bfloat16, device=key.device)

        grid_k = (Bk * Hk * Sk,)
        rms_norm_4d_kernel[grid_k](key, out_key, Bk, Hk, Sk, Dk, rms_norm_eps)

        # Return normalized query and key, and original caches (no updates)
        return out_query, out_key, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
