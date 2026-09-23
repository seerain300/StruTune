import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.int32, Dc: tl.constexpr):
    # Each program copies one token row from cache into out
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


@triton.jit
def gather_rows_p_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.int32, Dp: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


@triton.jit
def matvec_kernel(attn_ptr, K_ptr, out_ptr,
                   H: tl.constexpr, Dc: tl.constexpr, L: tl.int32, BLOCK_D: tl.constexpr):
    # One program per head i; compute out_vec[i] = attn[i, :] @ K
    i = tl.program_id(0)
    if i >= H:
        return
    acc = tl.zeros((Dc,), dtype=tl.float32)
    # Loop over tokens in chunks
    for t0 in range(0, L, BLOCK_D):
        offs_t = t0 + tl.arange(0, BLOCK_D)
        mask_t = offs_t < L
        # attn_chunk: [BLOCK_D]
        attn_chunk = tl.load(attn_ptr + i * L + offs_t, mask=mask_t, other=-float("inf"))
        # K_chunk: [BLOCK_D, Dc]
        K_chunk = tl.load(K_ptr + offs_t[:, None] * Dc + tl.arange(0, Dc), mask=mask_t[:, None], other=0.0)
        # acc += sum_j attn_chunk[j] * K_chunk[j, :]
        for j in range(0, BLOCK_D):
            kj = K_chunk[j, :]  # [Dc]
            a = attn_chunk[j]   # scalar
            acc += a * kj
    out_base = i * Dc
    for d in range(0, Dc):
        tl.store(out_ptr + out_base + d, acc[d])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and constants
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        assert H == 16, "num_qo_heads must be 16"
        Dc = q_nope.shape[2]
        assert Dc == 512, "head_dim_ckv must be 512"
        Dp = q_pe.shape[2]
        assert Dp == 64, "head_dim_kpe must be 64"

        # Squeezed caches to [P, Dc] / [P, Dp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Dp]

        # Output buffers
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Process each batch element
        for b in range(B):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV for this batch element: output zeros, lse zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather token indices
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather rows from caches into Kc_flat and Kp_flat (float32)
            Kc_flat = torch.empty((L_tokens * Dc,), dtype=torch.float32, device=q_nope.device)
            Kp_flat = torch.empty((L_tokens * Dp,), dtype=torch.float32, device=q_nope.device)

            grid_gather = (L_tokens,)
            gather_rows_c_kernel[grid_gather](Kc_all, tok_idx, Kc_flat, L_tokens, Dc)
            Kc = Kc_flat.view(L_tokens, Dc)  # [L_tokens, Dc]

            gather_rows_p_kernel[grid_gather](Kp_all, tok_idx, Kp_flat, L_tokens, Dp)
            Kp = Kp_flat.view(L_tokens, Dp)  # [L_tokens, Dp]

            # 2) For each head i: compute logits_scaled = qn[i] @ Kc.T + qp[i] @ Kp.T
            for i in range(H):
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]
                # GEMV: qn @ Kc.T -> [1, L_tokens]
                logits_qn = qn @ Kc.T
                # GEMV: qp @ Kp.T -> [1, L_tokens]
                logits_qp = qp @ Kp.T
                logits = (logits_qn + logits_qp).squeeze(0)  # [L_tokens]
                logits_scaled = logits * sm_scale            # [L_tokens]

                # 3) Softmax over tokens in PyTorch (robust and fast); attn: [L_tokens]
                attn = torch.softmax(logits_scaled, dim=0)

                # 4) Matvec: attn @ Kc -> [Dc], using Triton (one program per head)
                out_flat = torch.empty((H * Dc,), dtype=torch.float32, device=q_nope.device)
                matvec_kernel[(H,)](attn.contiguous(), Kc.contiguous(), out_flat, H, Dc, L_tokens, 128)
                out_vec = out_flat.view(H, Dc)  # [H, Dc]
                output[b, i] = out_vec[i].to(torch.bfloat16)

                # 5) lse (base-2): compute with torch for simplicity and correctness
                lse[b, i] = torch.logsumexp(logits_scaled) / math.log(2.0)

        return output, lse


def run(*args):
    return ModelNew()(*args)
