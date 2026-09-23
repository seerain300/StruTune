import torch
import triton
import triton.language as tl


@triton.jit
def rotate_half_rows_kernel(X_ptr, Cos_ptr, Sin_ptr, Out_ptr, B, N, S, D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: Rotate each row of X_ptr (shape [B, N, S, D]) into Out_ptr using Cos_ptr and Sin_ptr.
    - X_ptr: input tensor [B, N, S, D] in bfloat16 (or float32), row-major. We treat it as (M=S) rows for each (b, n).
    - Cos_ptr, Sin_ptr: [S, D] float32, per-token cos/sin vectors.
    - Out_ptr: output tensor [B, N, S, D], same dtype as X_ptr.
    We process rows as (b, n, s) and apply rotation:
      split x into x1, x2 halves: x1 = X[..., :D//2], x2 = X[..., D//2:].
      rotate_half(x) = [-x2, x1]
      y1 = x1 * cos[:D//2] + rotate_half(x)[:, :D//2] * sin[:D//2]
      y2 = x2 * cos[D//2:] + rotate_half(x)[:, D//2:] * sin[D//2:]
      y = concat([y1, y2]).
    """
    # Grid is 1D over rows; we need (b, n, s) mapping. Triton doesn't allow nested mapping directly,
    # but since S is known at call site, we can iterate s inside the kernel and use program_id to map to (b, n).
    # To keep the kernel simple and meaningful, we process one (b, n, s) per program.
    pid = tl.program_id(axis=0)
    total = B * N * S
    if pid >= total:
        return

    # Compute b, n, s from pid
    tmp = pid // S
    b = tmp // N
    n = tmp % N
    s = pid % S

    # Base offset for this (b, n, s) row in X_ptr and Out_ptr
    base = (b * (N * S) + n * S + s) * D

    # Load row X as bfloat16
    x = tl.load(X_ptr + base, mask=True, other=0.0)
    x_f32 = x.to(tl.float32)

    # Load cos and sin vectors for this token s
    cos_vec = tl.load(Cos_ptr + s * D + tl.arange(0, D), mask=True, other=0.0)
    sin_vec = tl.load(Sin_ptr + s * D + tl.arange(0, D), mask=True, other=0.0)

    # Split into halves
    D_half = D // 2
    x1 = x_f32[:D_half]
    x2 = x_f32[D_half:]

    # Prepare rotated half
    rotate_half_x = tl.zeros(D, dtype=tl.float32)
    rotate_half_x[:D_half] = -x2
    rotate_half_x[D_half:] = x1

    # Apply rotation: y = x1 * cos + rotate_half(x) * sin
    y1 = x1 * cos_vec[:D_half] + rotate_half_x[:D_half] * sin_vec[:D_half]
    y2 = x2 * cos_vec[D_half:] + rotate_half_x[D_half:] * sin_vec[D_half:]
    y = tl.concatenate([y1, y2])

    # Store result (cast back to original dtype)
    out_base = (b * (N * S) + n * S + s) * D
    tl.store(Out_ptr + out_base, y.to(x.dtype))


class ModelNew(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        inv_freq: torch.Tensor,
        rms_norm_eps: float,
    ):
        """
        Returns:
        - query_rotated: rotated query computed by Triton
        - key_rotated: rotated key computed by Triton
        - key_cache: original key_cache (no Triton scatter implemented here)
        - value_cache: original value_cache (no Triton scatter implemented here)
        """
        # Ensure inputs are contiguous
        query_c = query.contiguous()
        key_c = key.contiguous()
        value_c = value.contiguous()
        position_ids_c = position_ids.contiguous()  # [B, S], int64
        cache_position_c = cache_position.contiguous()  # [S], int64

        B, N_q, S, D = query_c.shape
        N_kv = key_c.shape[1]
        MAX_POS = key_cache.shape[2]

        # We will compute query and key rotation in Triton. We need per-token cos/sin vectors.
        # Build inv = [inv_freq, inv_freq] (fp32)
        inv = torch.empty(D, dtype=torch.float32, device=query.device)
        D_half = D // 2
        inv[:D_half] = inv_freq
        inv[D_half:] = inv_freq

        # Compute cos and sin per token s using cache_position as pos
        cos = torch.empty((S, D), dtype=torch.float32, device=query.device)
        sin = torch.empty((S, D), dtype=torch.float32, device=query.device)

        # We can compute cos/sin via Triton in a trivial element


def run(*args):
    return ModelNew()(*args)
