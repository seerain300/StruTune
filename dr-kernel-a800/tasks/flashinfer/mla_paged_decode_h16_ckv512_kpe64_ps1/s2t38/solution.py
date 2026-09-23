import torch
import triton
import triton.language as tl


@triton.jit
def _compute_output_bh(
    q_nope_ptr,        # *bf16, [B, H, D1] flattened as H*D1 rows
    q_pe_ptr,          # *bf16, [B, H, D2] flattened as H*D2 rows
    Kc_all_ptr,        # *bf16, [N, D1] flattened
    Kp_all_ptr,        # *bf16, [N, D2] flattened
    out_ptr,           # *bf32, 1D buffer to store output [B, H, D1]
    H: tl.constexpr,   # num heads (e.g., 16)
    D1: tl.constexpr,  # head_dim_ckv (e.g., 512)
    D2: tl.constexpr,  # head_dim_kpe (e.g., 64)
    MAX_T: tl.constexpr,  # max tokens processed per loop
):
    # One program per (b, h)
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Base offsets
    base_qn = b * (H * D1) + h * D1
    base_qp = b * (H * D2) + h * D2

    # Load qn and qp vectors (bf16) and cast to f32
    qn = tl.load(q_nope_ptr + base_qn + tl.arange(0, D1)).to(tl.float32)  # [D1]
    qp = tl.load(q_pe_ptr + base_qp + tl.arange(0, D2)).to(tl.float32)   # [D2]

    # Output vector for this (b, h)
    out_vec_ptr = out_ptr + pid * D1
    # Initialize output vector to zeros
    for d in tl.static_range(0, D1):
        tl.store(out_vec_ptr + d, 0.0)

    # Compute and accumulate output per token with softmax
    token_max = tl.full((), -float("inf"), dtype=tl.float32)  # scalar max
    token_sum = tl.zeros((), dtype=tl.float32)                # scalar sum

    for t in tl.static_range(0, MAX_T):
        valid = t < (kv_indptr[b + 1] - kv_indptr[b])  # device tensors are used to infer L_tokens
        # Load Kc_row and Kp_row for this token index; if invalid, use zeros
        # Note: Triton requires known shapes; here we assume N >= number of tokens, masked loads handle OOB.
        Kc_row = tl.load(Kc_all_ptr + t * D1 + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
        Kp_row = tl.load(Kp_all_ptr + t * D2 + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

        # Dot products
        dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
        dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
        logits_scalar = (dot1 + dot2) * sm_scale  # scalar

        # Masked update for LSE
        if valid:
            token_max = tl.maximum(token_max, logits_scalar)
            token_sum += tl.exp(logits_scalar - token_max)

    # Second pass: compute normalized attn and accumulate output
    for t in tl.static_range(0, MAX_T):
        valid = t < (kv_indptr[b + 1] - kv_indptr[b])
        Kc_row = tl.load(Kc_all_ptr + t * D1 + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
        Kp_row = tl.load(Kp_all_ptr + t * D2 + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

        dot1 = tl.sum(qn * Kc_row, axis=0)
        dot2 = tl.sum(qp * Kp_row, axis=0)
        logits_scalar = (dot1 + dot2) * sm_scale

        attn = 0.0
        if valid:
            attn = tl.exp(logits_scalar - token_max) / token_sum  # scalar
        out_vec_ptr_t = out_vec_ptr + t * D1  # not used, but keep context
        # Accumulate output: out += attn * Kc_row
        for d in tl.static_range(0, D1):
            tl.atomic_add(out_vec_ptr + d, attn * Kc_row[d])

    # No need to store lse here; forward returns output only.


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale: float = 1.0, max_t: int = 2048, num_warps: int = 4):
        super().__init__()
        self.sm_scale = float(sm_scale)
        self.max_t = int(max_t)
        self.num_warps = num_warps

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale=None):
        # Ensure all tensors are on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D1 = q_nope.shape[2]
        D2 = q_pe.shape[2]
        N = ckv_cache.shape[0]

        # Flattened views for Triton: no .to() on host; casting handled inside kernel
        q_nope_flat = q_nope.view(B * H * D1)               # [B*H*D1]
        q_pe_flat = q_pe.view(B * H * D2)                  # [B*H*D2]
        Kc_all_flat = ckv_cache.view(N * D1)               # [N*D1]
        Kp_all_flat = kpe_cache.view(N * D2)               # [N*D2]

        # Output buffer (float32 for numeric stability), 1D [B*H*D1]
        out = torch.empty(B * H * D1, dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * H,)
        _compute_output_bh[grid](
            q_nope_flat, q_pe_flat, Kc_all_flat, Kp_all_flat, out,
            H=H, D1=D1, D2=D2, MAX_T=self.max_t,
            sm_scale=self.sm_scale if sm_scale is None else float(sm_scale),
            num_warps=self.num_warps,
        )

        # Reshape and cast to bfloat16 to match original output dtype
        output = out.view(B, H, D1).to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
