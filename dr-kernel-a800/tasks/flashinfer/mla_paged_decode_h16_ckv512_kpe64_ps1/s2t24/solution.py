import math
import torch
import triton
import triton.language as tl


@triton.jit
def _lse_and_output_kernel(
    q_nope_ptr,            # *f32, shape [H*D1], contiguous (flattened)
    q_pe_ptr,              # *f32, shape [H*D2], contiguous (flattened)
    Kc_ptr,                # *f32, shape [N, D1], contiguous (we select rows by kv_indices)
    Kp_ptr,                # *f32, shape [N, D2], contiguous (we select rows by kv_indices)
    out_ptr,               # *f32, shape [H*D1], contiguous (flattened output)
    lse_ptr,               # *f32, shape [B*H], contiguous
    kv_indptr_ptr,         # *int32, shape [B+1]
    kv_indices_ptr,        # *int32, shape [num_kv_indices]
    B: tl.int32,           # batch size (runtime)
    H: tl.int32,           # number of heads (runtime)
    D1: tl.constexpr,      # 512
    D2: tl.constexpr,      # 64
    N: tl.int32,           # number of cache entries (runtime, not used for loads)
    MAX_T: tl.constexpr,   # maximum number of tokens per batch element (e.g., 2048)
    sm_scale: tl.float32,  # scaling factor
):
    # One Triton program per batch element b
    b = tl.program_id(axis=0)

    # Compute L_tokens for this batch element
    L_tokens = tl.load(kv_indptr_ptr + b + 1) - tl.load(kv_indptr_ptr + b)

    # Initialize per-column max and sum across tokens
    token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
    token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

    # Loop over heads h for this batch element
    for h in tl.static_range(0, H):
        # Load qn and qp vectors for head h
        qn = tl.load(q_nope_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # First pass: compute logsumexp across tokens
        for t in tl.static_range(0, MAX_T):
            if t >= L_tokens:
                break
            # Select index for this token: idx = kv_indices[page_beg + t]
            # For b: page_beg = kv_indptr[b]
            idx = tl.load(kv_indices_ptr + t)
            # Load Kc_row and Kp_row as 1D vectors
            Kc_row = tl.load(Kc_ptr + idx * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
            Kp_row = tl.load(Kp_ptr + idx * D2 + tl.arange(0, D2)).to(tl.float32)  # [D2]

            # Compute scalar logits for this token
            dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
            dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
            logits_scalar = (dot1 + dot2) * sm_scale  # scalar

            token_max_vec = tl.maximum(token_max_vec, logits_scalar)
            token_sum_vec += tl.exp(logits_scalar - token_max_vec)

        # Compute lse for head h
        lse_val = tl.log(token_sum_vec) / math.log(2.0) + token_max_vec
        tl.store(lse_ptr + b * H + h, lse_val)

        # Second pass: accumulate output with softmax scaling
        for t in tl.static_range(0, MAX_T):
            if t >= L_tokens:
                break
            idx = tl.load(kv_indices_ptr + t)
            Kc_row = tl.load(Kc_ptr + idx * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
            Kp_row = tl.load(Kp_ptr + idx * D2 + tl.arange(0, D2)).to(tl.float32)  # [D2]

            dot1 = tl.sum(qn * Kc_row, axis=0)
            dot2 = tl.sum(qp * Kp_row, axis=0)
            logits_scalar = (dot1 + dot2) * sm_scale

            attn = tl.exp(logits_scalar - token_max_vec) / token_sum_vec  # scalar
            out_vec = tl.load(out_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
            out_vec += attn * Kc_row
            tl.store(out_ptr + h * D1 + tl.arange(0, D1), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, max_t=2048):
        super().__init__()
        self.max_t = max_t

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # q_nope: [B, H, D1], q_pe: [B, H, D2], ckv_cache: [N, 1, D1], kpe_cache: [N, 1, D2]
        # kv_indptr: [B+1], kv_indices: [num_kv_indices], sm_scale: float
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D1 = q_nope.shape[2]
        D2 = q_pe.shape[2]
        N = ckv_cache.shape[0]
        # Ensure contiguous
        q_nope_flat = q_nope.to(torch.float32).contiguous().view(H * D1)        # [H*D1]
        q_pe_flat = q_pe.to(torch.float32).contiguous().view(H * D2)            # [H*D2]
        Kc = ckv_cache.to(torch.float32).contiguous()                           # [N, D1]
        Kp = kpe_cache.to(torch.float32).contiguous()                           # [N, D2]
        # Output buffer in float32 (will cast to bfloat16 at the end)
        out = torch.zeros(H * D1, dtype=torch.float32, device=device)
        lse = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        _lse_and_output_kernel[grid](
            q_nope_flat, q_pe_flat, Kc, Kp, out, lse, kv_indptr, kv_indices,
            B=B, H=H, D1=D1, D2=D2, N=N, MAX_T=self.max_t, sm_scale=float(sm_scale),
        )

        # Reshape and cast output to bfloat16 to match original
        output = out.view(B, H, D1).to(torch.bfloat16)
        lse = lse.view(B, H)
        return output, lse


def run(*args):
    return ModelNew()(*args)
