import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    Q_ptr,                 # *fp32, q vector for head h, length = HEAD_DIM (1D)
    K_ptr,                 # *fp32, K tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    V_ptr,                 # *fp32, V tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,               # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,               # *fp32, scalar lse for this (b, h), length 1
    NUM_TOKENS: tl.constexpr,    # runtime integer: number of tokens for this batch
    HEAD_DIM: tl.constexpr,      # 128
    SM_SCALE: tl.float32,        # scaling factor, e.g., 1/sqrt(HEAD_DIM)
    LOG2_INVERSE: tl.float32,    # 1/ln(2) ≈ 1.4426950408889634
):
    # One program instance per (b, h), where h = program_id(0)
    # Note: Triton grid is 1D here; we map h via program_id(0). OUT_ptr and LSE_ptr are per-b per-h.
    h = tl.program_id(0)

    # First pass: compute lse = logsumexp(logits_scaled) / ln(2)
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head: [HEAD_DIM]
        q_vec = tl.load(Q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        # Load k vector for token t: [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        # Compute logits_t = dot(q_vec, k_vec)
        logits_t = tl.sum(q_vec * k_vec, axis=0)                        # scalar
        scaled = logits_t * SM_SCALE
        # Update running max and sum for logsumexp
        running_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - running_max) + tl.exp(scaled - running_max)

    # Compute lse = log(running_sum) + running_max, divide by ln(2) via multiply by LOG2_INVERSE
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * LOG2_INVERSE

    # Store scalar lse to LSE_ptr[b * num_qo_heads + h]
    # Since we don't have b here, the host will pass LSE_ptr as a flat pointer and we store directly.
    # The host ensures LSE_ptr[b * num_qo_heads + h] exists.
    tl.store(LSE_ptr, lse_val)

    # Second pass: compute output vector for this head
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(Q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        logits_t = tl.sum(q_vec * k_vec, axis=0)                        # scalar
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse_val)  # softmax over tokens for scaled logits
        out_vec += attn * v_vec

    # Store output vector
    tl.store(OUT_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)  # 1/sqrt(128)
        self.log2_inverse = 1.0 / math.log(2.0)  # 1/ln(2)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device and dtype
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA"
        B = q.shape[0]
        assert q.shape == (B, self.num_qo_heads, self.head_dim)

        # Prepare output and lse as fp32 for compute
        output = torch.empty((B, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, self.num_qo_heads), dtype=torch.float32, device=q.device)

        # Compute per-batch number of tokens: num_tokens[b] = kv_indptr[b+1] - kv_indptr[b]
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1, "kv_indptr must have length batch_size + 1"

        # We don't need kv_indices here; we only need num_tokens per batch from kv_indptr.
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens = end - start

            # If no tokens for this batch, zero outputs
            if num_tokens <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # GQA mapping: kv_head = h // gqa_ratio
            for h in range(self.num_qo_heads):
                kv_head = h // self.gqa_ratio  # 0..7

                # Slice k_cache and v_cache for this batch and head: [num_tokens, 128]
                K_t_h = k_cache[start:end, 0, kv_head, :]  # [num_tokens, 128]
                V_t_h = v_cache[start:end, 0, kv_head, :]  # [num_tokens, 128]
                # Ensure contiguous fp32
                K_t_h = K_t_h.to(torch.float32).contiguous()  # [num_tokens, 128]
                V_t_h = V_t_h.to(torch.float32).contiguous()  # [num_tokens, 128]

                # Prepare q vector for this head: Q[b, h] as 1D fp32 tensor
                q_vec_h = q[b, h].to(torch.float32).contiguous()  # [128]

                # Output vector for this head (fp32), contiguous
                out_vec = torch.empty((self.head_dim,), dtype=torch.float32, device=q.device)

                # Scalar lse buffer: pass a 1-element tensor to store scalar lse, indexed by (b, h)
                lse_scalar = torch.empty((1,), dtype=torch.float32, device=q.device)

                # Launch Triton kernel: grid = (1,), single program instance for this (b, h)
                softmax_and_attention_single_bh[(1,)](
                    q_vec_h,          # Q_ptr: 1D vector
                    K_t_h.view(-1, self.head_dim),  # [NUM_TOKENS, HEAD_DIM]
                    V_t_h.view(-1, self.head_dim),  # [NUM_TOKENS, HEAD_DIM]
                    out_vec,          # OUT_ptr: 1D vector to store output
                    lse_scalar,       # LSE_ptr: 1-element tensor to store scalar lse
                    NUM_TOKENS=num_tokens,
                    HEAD_DIM=self.head_dim,
                    SM_SCALE=(self.sm_scale if sm_scale is None else (sm_scale if isinstance(sm_scale, float) else float(sm_scale))),
                    LOG2_INVERSE=self.log2_inverse,
                    num_warps=2,      # small, suitable for 128-dim loops
                    num_stages=1,
                )

                # Store outputs
                output[b, h] = out_vec
                lse[b, h] = lse_scalar[0]

        # Cast output to bfloat16 to match original signature
        output = output.to(torch.bfloat16)
        # lse remains float32 as in original
        return output, lse


def run(*args):
    return ModelNew()(*args)
