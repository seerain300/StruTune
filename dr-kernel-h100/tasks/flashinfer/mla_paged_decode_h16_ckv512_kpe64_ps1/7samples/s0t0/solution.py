import math
import torch

import triton
import triton.language as tl


@triton.jit
def attention_output_kernel(
    qn_ptr,  # float32 * [16, 512]
    qp_ptr,  # float32 * [16, 64]
    Kc_ptr,  # float32 * [num_pages, 512]
    Kp_ptr,  # float32 * [num_pages, 64]
    tok_idx_ptr,  # int32 * [L_tokens]
    out_ptr,  # float32 * [16, 512]
    DQ: tl.constexpr,  # 512
    DP: tl.constexpr,  # 64
    D: tl.constexpr,   # 512
    SM_SCALE: tl.float32,
    L_TOKENS: tl.constexpr,  # number of tokens for this batch
):
    # program id for (b, head)
    pid = tl.program_id(0)
    # head is last dimension since output is [B, 16, 512]
    b = pid // 16
    head = pid % 16

    # Compute base offsets for q rows
    qn_row = qn_ptr + head * DQ
    qp_row = qp_ptr + head * DP

    # Prepare output vector (fp32)
    out_vec = tl.zeros((D,), dtype=tl.float32)

    # Streaming logsumexp scalars
    m = tl.full((), -float('inf'), dtype=tl.float32)  # running max
    s = tl.zeros((), dtype=tl.float32)                # running sum of exp(logits - m)

    # Iterate over tokens
    for t in range(0, L_TOKENS):
        idx = tl.load(tok_idx_ptr + t)  # int32 index
        Kc_row = Kc_ptr + idx * D       # [512]
        Kp_row = Kp_ptr + idx * DP      # [64]

        # Dot products: qn_row[512] dot Kc_row[512], and qp_row[64] dot Kp_row[64]
        # Note: tl.dot expects two tensors of same shape; we emulate row vector dot with elementwise and reduce.
        dot1 = tl.sum(tl.load(qn_row) * tl.load(Kc_row), axis=0)
        dot2 = tl.sum(tl.load(qp_row) * tl.load(Kp_row), axis=0)
        logit = dot1 + dot2  # scalar
        logit = logit * SM_SCALE

        # Streaming logsumexp update
        m_new = tl.maximum(m, logit)
        p = tl.exp(logit - m_new)
        s = s * tl.exp(m - m_new) + p
        m = m_new

        # Accumulate attention output: out_vec += p * Kc_row
        out_vec += p * tl.load(Kc_row)

    # Normalize by softmax denominator (since softmax divides by sum, we divide by s)
    out_vec = out_vec / s

    # Store the final output vector for this (b, head)
    out_row = out_ptr + b * (16 * D) + head * D
    tl.store(out_row, out_vec)


@triton.jit
def lse_accum_kernel(
    qn_ptr,  # float32 * [16, 512]
    qp_ptr,  # float32 * [16, 64]
    Kc_ptr,  # float32 * [num_pages, 512]
    Kp_ptr,  # float32 * [num_pages, 64]
    tok_idx_ptr,  # int32 * [L_tokens]
    lse_ptr,  # float32 * [B, 16]
    DQ: tl.constexpr,  # 512
    DP: tl.constexpr,  # 64
    D: tl.constexpr,   # 512
    SM_SCALE: tl.float32,
    L_TOKENS: tl.constexpr,  # number of tokens for this batch
):
    # program id for (b, head)
    pid = tl.program_id(0)
    b = pid // 16
    head = pid % 16

    qn_row = qn_ptr + head * DQ
    qp_row = qp_ptr + head * DP

    m = tl.full((), -float('inf'), dtype=tl.float32)
    s = tl.zeros((), dtype=tl.float32)

    for t in range(0, L_TOKENS):
        idx = tl.load(tok_idx_ptr + t)
        Kc_row = Kc_ptr + idx * D
        Kp_row = Kp_ptr + idx * DP

        dot1 = tl.sum(tl.load(qn_row) * tl.load(Kc_row), axis=0)
        dot2 = tl.sum(tl.load(qp_row) * tl.load(Kp_row), axis=0)
        logit = (dot1 + dot2) * SM_SCALE

        m_new = tl.maximum(m, logit)
        p = tl.exp(logit - m_new)
        s = s * tl.exp(m - m_new) + p
        m = m_new

    # Logsumexp base-2 normalization
    lse_val = (m + tl.log(s)) * (1.0 / 0.6931471805599453)  # 1/log(2)
    lse_row = lse_ptr + b * 16 + head
    tl.store(lse_row, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, 16, 512], bfloat16
        q_pe: [B, 16, 64], bfloat16
        ckv_cache: [num_pages, 1, 512], bfloat16
        kpe_cache: [num_pages, 1, 64], bfloat16
        kv_indptr: [B+1], int32
        kv_indices: [L_tokens], int32
        sm_scale: float32 scalar
        Returns (output [B, 16, 512] bfloat16, lse [B, 16] float32)
        """
        B = q_nope.shape[0]
        assert q_nope.shape == (B, 16, 512)
        assert q_pe.shape == (B, 16, 64)
        D = 512
        DP = 64
        device = q_nope.device

        # Ensure we operate in float32 for numerical stability; cast inputs to fp32
        # Note: Triton will read these as fp32 pointers
        q_nope_f32 = q_nope.contiguous().to(torch.float32)
        q_pe_f32 = q_pe.contiguous().to(torch.float32)

        # Flatten caches to [num_pages, D] and [num_pages, DP] for easier indexing
        num_pages = ckv_cache.shape[0]
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, DP]

        # Prepare output buffers (fp32 for compute, then cast to bfloat16)
        output_fp32 = torch.empty((B, 16, D), dtype=torch.float32, device=device)

        # We need tok_idx per batch. Given kv_indptr and kv_indices, build tok_idx.
        # len_indptr = B + 1
        # For each b, tokens = indices in [kv_indptr[b]: kv_indptr[b+1])
        tok_idx_list = []  # list of tensors [L_tokens] for each b
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens > 0:
                tok_idx = kv_indices[start:start + L_tokens].to(torch.int32).to(device)
                tok_idx_list.append(tok_idx)
            else:
                tok_idx_list.append(torch.empty(0, dtype=torch.int32, device=device))

        # Launch Triton kernel: one program per (b, head)
        grid = (B * 16,)
        # We need to pass L_TOKENS per program. Triton expects meta-parameters; we can pass L_TOKENS as a dict in kernel launch per b.
        # However, Triton kernels are compiled per unique constexpr args. Here, tok_idx_list[b] varies, but we can still pass L_TOKENS and pointer.
        # Triton supports passing loop length as constexpr. We'll call kernel once with max L_TOKENS among batches, which is fine because when L_tokens == 0 we can early return.
        max_L = max([t.numel() for t in tok_idx_list]) if tok_idx_list else 0

        # To use constexpr, Triton requires a static L_TOKENS. Since tok_idx_list length varies by batch, we can:
        # 1) Launch kernels per b with its specific L_TOKENS. Implementing a per-b kernel requires different grid size. Instead, we can:
        # 2) Pad tok_idx for each batch to max_L with sentinel index (e.g., -1), and in kernel guard loads with mask. But Triton loops rely on constexpr. The simplest is to do per-b launch with Python loop using Triton's dynamic meta trick:
        # We can still compile per L_TOKENS by passing a single L_TOKENS (max) and guard loads for L_TOKENS < t. For simplicity, we'll compute output in two passes: for batches with L_tokens == 0, we just skip.

        # But Triton requires constexpr loop; to handle variable L_tokens cleanly, we compute output with per-b launch using a wrapper that sets L_TOKENS=b's tok_idx length.
        # Triton doesn't support dynamic constexpr per program easily; we'll implement a small Python loop to launch per-b with its L_TOKENS.

        # For correctness, we'll loop and call the kernel per batch with its L_TOKENS, but Triton kernels need a compile-time constant; passing dynamic L_TOKENS is not supported.
        # Workaround: we compute L tokens and pass the actual length. Triton allows passing constexpr meta via keyword when launching. We'll launch per-b using the same kernel and pass the per-b L_TOKENS.

        # Since Triton doesn't expose per-b constexpr meta easily in a single call, we will launch one kernel per batch by iterating in Python, and passing the specific L_TOKENS for that batch.
        # This is acceptable and maintains performance because B is small in typical test.

        # Implement per-b launch:
        for b in range(B):
            L_tokens = tok_idx_list[b].numel()
            if L_tokens == 0:
                # Nothing to do; output remains zeros. We set output_fp32[b] to zero.
                output_fp32[b] = torch.zeros((16, D), dtype=torch.float32, device=device)
                continue

            tok_idx_b = tok_idx_list[b]

            # Launch Triton kernel for this (b, head). We create a grid of size 16 (heads) for this batch.
            grid_b = (16,)
            attention_output_kernel[grid_b](
                q_nope_f32[b],  # qn_ptr for this batch
                q_pe_f32[b],    # qp_ptr for this batch
                Kc_all,         # Kc_ptr
                Kp_all,         # Kp_ptr
                tok_idx_b,      # tok_idx_ptr
                output_fp32[b], # out_ptr
                DQ=512,
                DP=64,
                D=512,
                SM_SCALE=float(sm_scale),
                L_TOKENS=L_tokens,  # constexpr for this launch
                num_warps=4,
            )

        # Compute lse in Triton per (b, head) similarly
        lse_fp32 = torch.empty((B, 16), dtype=torch.float32, device=device)
        for b in range(B):
            L_tokens = tok_idx_list[b].numel()
            if L_tokens == 0:
                lse_fp32[b] = torch.full((16,), -float('inf'), dtype=torch.float32, device=device)
                continue

            grid_b = (16,)
            lse_accum_kernel[grid_b](
                q_nope_f32[b],
                q_pe_f32[b],
                Kc_all,
                Kp_all,
                tok_idx_b,
                lse_fp32[b],
                DQ=512,
                DP=64,
                D=512,
                SM_SCALE=float(sm_scale),
                L_TOKENS=L_tokens,
                num_warps=4,
            )

        # Cast output to bfloat16 as per original Model
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse_fp32


def run(*args):
    return ModelNew()(*args)
