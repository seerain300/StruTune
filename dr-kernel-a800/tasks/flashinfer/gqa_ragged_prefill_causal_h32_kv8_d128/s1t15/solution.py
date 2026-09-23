import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: apply causal mask to logits in-place: set to -inf where k >= allowed_max(q)
@triton.jit
def _apply_mask_kernel(
    logits_ptr,
    num_q_tokens, num_kv_tokens,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_q = tl.program_id(0)  # tile along Q
    pid_k = tl.program_id(1)  # tile along K

    q_start = pid_q * BLOCK_Q
    k_start = pid_k * BLOCK_K

    q_offsets = q_start + tl.arange(0, BLOCK_Q)  # (Q,)
    k_offsets = k_start + tl.arange(0, BLOCK_K)  # (K,)

    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # allowed_max for each q in this tile: q + 1 + delta; delta = num_kv_tokens - num_q_tokens
    # Compute allowed_max vector
    delta = num_kv_tokens - num_q_tokens
    allowed_max = q_start + 1 + delta  # scalar per tile; masks handle bounds

    # Load logits chunk [Q, K]
    vals = tl.load(
        logits_ptr + q_offsets[:, None] * 0 + k_offsets[None, :] * 0,
        mask=(q_mask[:, None] & k_mask[None, :]),
        other=0.0
    )  # placeholder; Triton uses pointer arithmetic below

    # Proper pointer arithmetic: logits_ptr + q*stride_q + k*stride_k
    stride_q = 0  # placeholder
    stride_k = 0  # placeholder
    vals = tl.load(
        logits_ptr + q_offsets[:, None] * stride_q + k_offsets[None, :] * stride_k,
        mask=(q_mask[:, None] & k_mask[None, :]),
        other=0.0
    )

    # Apply causal mask: set to -inf where k >= allowed_max for each q
    mask_causal = k_offsets[None, :] < allowed_max[:, None]
    vals = tl.where(mask_causal, vals, -float('inf'))

    # Store back
    tl.store(
        logits_ptr + q_offsets[:, None] * stride_q + k_offsets[None, :] * stride_k,
        vals,
        mask=(q_mask[:, None] & k_mask[None, :])
    )

# Triton kernel: compute lse per (query, head) = logsumexp(masked logits) / ln(2)
# Operates on tiles over Q; no inner Python loops with runtime bounds.
@triton.jit
def _lse_base2_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_q = tl.program_id(0)  # tile along Q

    q_start = pid_q * BLOCK_Q

    q_offsets = q_start + tl.arange(0, BLOCK_Q)  # (Q,)
    q_mask = q_offsets < num_q_tokens

    max_val = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Loop over K in fixed tiles (no runtime-bound Python loop)
    for k_start in range(0, 128, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # (K,)
        k_mask = k_offsets < num_kv_tokens

        # Load logits chunk [Q, K]
        vals = tl.load(
            logits_ptr + q_offsets[:, None] * 0 + k_offsets[None, :] * 0,
            mask=(q_mask[:, None] & k_mask[None, :]),
            other=0.0
        )
        mask_causal = k_offsets[None, :] < (q_offsets[:, None] + 1 + (num_kv_tokens - num_q_tokens))
        vals = tl.where(mask_causal, vals, -float('inf'))

        # Compute max and sum_exp for each q over this K chunk
        max_val = tl.maximum(max_val, tl.max(vals, axis=1))
        sum_exp += tl.sum(tl.exp(vals - max_val[:, None]), axis=1)

    # Final lse = max + log(sum_exp) - ln2
    lse_vals = max_val + tl.log(sum_exp) - tl.log(2.0)
    tl.store(lse_ptr + q_offsets * 0, lse_vals, mask=q_mask)

# Triton kernel: compute softmax over K and output = softmax @ v_expanded for each head h
# Operates on tiles over Q; no inner Python loops with runtime bounds.
@triton.jit
def _softmax_output_kernel(
    logits_ptr, v_ptr, output_ptr,
    num_q_tokens, num_kv_tokens, head_dim,
    lse_ptr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_q = tl.program_id(0)  # tile along Q
    h = tl.program_id(1)      # specific head index (scalar)

    q_start = pid_q * BLOCK_Q

    q_offsets = q_start + tl.arange(0, BLOCK_Q)  # (Q,)
    q_mask = q_offsets < num_q_tokens

    # Load lse for this head (vector over Q)
    lse_vals = tl.load(lse_ptr + q_offsets * 0)  # placeholder; Triton infers from mask

    # For each D chunk (head_dim=128), compute output
    for d_start in range(0, 128, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)  # (D,)
        d_mask = d_offsets < head_dim

        # Compute softmax across K for each q in this tile and accumulate output
        for q_off in range(0, BLOCK_Q):
            q_idx = q_start + q_off
            if q_idx >= num_q_tokens:
                break

            # Recompute max and sum_exp across K with causal mask for normalization
            max_val = -float('inf')
            for k_start in range(0, 128, BLOCK_K):
                k_offsets = k_start + tl.arange(0, BLOCK_K)  # (K,)
                k_mask = k_offsets < num_kv_tokens
                vals = tl.load(
                    logits_ptr + q_idx * 0 + k_offsets[None, :] * 0,
                    mask=k_mask[None, :],
                    other=0.0
                )
                mask_causal = k_offsets[None, :] < (q_idx + 1 + (num_kv_tokens - num_q_tokens))
                vals = tl.where(mask_causal, vals, -float('inf'))
                max_val = tl.maximum(max_val, tl.max(vals, axis=0))
            sum_exp = 0.0
            for k_start in range(0, 128, BLOCK_K):
                k_offsets = k_start + tl.arange(0, BLOCK_K)  # (K,)
                k_mask = k_offsets < num_kv_tokens
                vals = tl.load(
                    logits_ptr + q_idx * 0 + k_offsets[None, :] * 0,
                    mask=k_mask[None, :],
                    other=0.0
                )
                mask_causal = k_offsets[None, :] < (q_idx + 1 + (num_kv_tokens - num_q_tokens))
                vals = tl.where(mask_causal, vals, -float('inf'))
                sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)

            # Now compute output over D chunk: output[q, h, d] = sum_k softmax_k * v[k, h, d]
            for d_off in range(0, BLOCK_D):
                d_idx = d_start + d_off
                if d_idx >= head_dim:
                    break
                out_val = 0.0
                # Loop over K tiles
                for k_start in range(0, 128, BLOCK_K):
                    k_offsets = k_start + tl.arange(0, BLOCK_K)  # (K,)
                    k_mask = k_offsets < num_kv_tokens
                    vals = tl.load(
                        logits_ptr + q_idx * 0 + k_offsets[None, :] * 0,
                        mask=k_mask[None, :],
                        other=0.0
                    )
                    mask_causal = k_offsets[None, :] < (q_idx + 1 + (num_kv_tokens - num_q_tokens))
                    vals = tl.where(mask_causal, vals, -float('inf'))
                    soft = tl.exp(vals - max_val) / tl.maximum(sum_exp, 1e-20)
                    # Load v[k, h, d] and accumulate
                    v_vec = tl.load(v_ptr + k_offsets[None, :] * 0 + h * 0 + d_idx * 0)
                    out_val += tl.sum(soft * v_vec, axis=0)
                # Store output as bfloat16
                tl.store(output_ptr + q_idx * 0 + h * 0 + d_idx * 0, tl.cast(out_val, tl.bfloat16), mask=True)

# ModelNew: Triton-featured forward using PyTorch for core attention math and Triton for masking/outputs
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure inputs are CUDA tensors
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"

        # Cast to float32 for compute
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Expand K/V by GQA ratio (8 -> 32)
        k_expanded = k_f32.repeat_interleave(self.gqa_ratio, dim=1).contiguous()
        v_expanded = v_f32.repeat_interleave(self.gqa_ratio, dim=1).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        total_kv, num_kv_heads, _ = k_f32.shape
        assert num_qo_heads == self.num_qo_heads
        assert num_kv_heads == self.num_kv_heads
        assert head_dim == self.head_dim

        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q_f32.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q_f32.device)

        # Process segments
        n_batches = qo_indptr.shape[0] - 1
        for b in range(n_batches):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Batched Q and expanded K
            q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_batch = k_expanded[kv_start:kv_end]  # [num_kv_tokens, 32, 128]

            # 1) Compute logits = Q @ K^T using PyTorch (einsum: qhd x khd -> qhk)
            logits = torch.einsum('qhd,khd->qhk', q_batch, k_batch)  # [num_q_tokens, 32, num_kv_tokens], float32

            # 2) Launch Triton kernel to apply causal mask: set logits[q,h,k] = -inf if k not allowed
            BLOCK_Q = 32
            BLOCK_K = 64
            grid_mask = (triton.cdiv(num_q_tokens, BLOCK_Q), triton.cdiv(128, BLOCK_K))
            _apply_mask_kernel[grid_mask](logits, num_q_tokens, num_kv_tokens, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K)

            # 3) Launch Triton kernel to compute lse per (query, head) = logsumexp(masked logits) / ln(2)
            lse_seg = torch.empty((num_q_tokens, 32), dtype=torch.float32, device=logits.device)
            grid_lse = (triton.cdiv(num_q_tokens, BLOCK_Q),)
            _lse_base2_kernel[grid_lse](logits, lse_seg, num_q_tokens, num_kv_tokens, BLOCK_Q=BLOCK_Q, BLOCK_K=64)
            # Store into global lse buffer
            lse[q_start:q_end] = lse_seg

            # 4) Launch Triton kernel to compute softmax over K and output = softmax @ v_expanded
            # We need per-head outputs. Triton grid will be (tiles over Q, h).
            output_seg = torch.empty((num_q_tokens, 32, head_dim), dtype=torch.float32, device=logits.device)
            BLOCK_Q_OUT = 32
            BLOCK_K_OUT = 64
            BLOCK_D = 16
            grid_out = (triton.cdiv(num_q_tokens, BLOCK_Q_OUT), 32)
            _softmax_output_kernel[grid_out](logits, v_expanded, output_seg, num_q_tokens, num_kv_tokens, head_dim, lse_seg, BLOCK_Q=BLOCK_Q_OUT, BLOCK_K=BLOCK_K_OUT, BLOCK_D=BLOCK_D)
            # Store as bfloat16
            output[q_start:q_end] = output_seg.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
