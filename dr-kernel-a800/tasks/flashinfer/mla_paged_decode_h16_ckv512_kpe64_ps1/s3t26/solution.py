import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels
if TRITON_AVAILABLE:

    @triton.jit
    def fused_logits_kernel(
        qn_ptr,        # *f32 [H, Dq]
        qp_ptr,        # *f32 [H, Dp]
        Kc_ptr,        # *f32 [T, Dq]
        Kp_ptr,        # *f32 [T, Dp]
        logits_ptr,    # *f32 [H, T]
        H: tl.constexpr,
        T: tl.constexpr,
        Dq: tl.constexpr,    # 512
        Dp: tl.constexpr,    # 64
        BLOCK_T: tl.constexpr   # power-of-two tile, e.g., 128
    ):
        h = tl.program_id(0)  # one program per head
        # Load qn[h, :] and qp[h, :]
        qn_row = tl.load(qn_ptr + h * Dq + tl.arange(0, Dq), mask=tl.arange(0, Dq) < Dq, other=0.0)
        qp_row = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)

        # Compute logits[h, t] = dot(qn_row, Kc[t, :]) + dot(qp_row, Kp[t, :])
        for t_start in range(0, T, BLOCK_T):
            offs_t = t_start + tl.arange(0, BLOCK_T)  # power-of-two arange size
            mask_t = offs_t < T
            # Load Kc_sub and Kp_sub: [BLOCK_T, Dq] and [BLOCK_T, Dp]
            Kc_sub = tl.load(
                Kc_ptr + offs_t[:, None] * Dq + tl.arange(0, Dq)[None, :],
                mask=mask_t[:, None],
                other=0.0
            )  # shape [BLOCK_T, Dq]
            Kp_sub = tl.load(
                Kp_ptr + offs_t[:, None] * Dp + tl.arange(0, Dp)[None, :],
                mask=mask_t[:, None],
                other=0.0
            )  # shape [BLOCK_T, Dp]

            # Dot products: (1x512) @ (512xBLOCK_T) -> (1xBLOCK_T), (1x64) @ (64xBLOCK_T) -> (1xBLOCK_T)
            dot1 = tl.sum(qn_row[None, :] * Kc_sub, axis=1)  # [BLOCK_T]
            dot2 = tl.sum(qp_row[None, :] * Kp_sub, axis=1)  # [BLOCK_T]
            logits_sub = dot1 + dot2  # [BLOCK_T]
            # Store to logits[h, offs_t]
            tl.store(logits_ptr + h * T + offs_t, logits_sub, mask=mask_t)


    @triton.jit
    def softmax_row_kernel(
        logits_ptr,   # *f32 [H, T]
        attn_ptr,     # *f32 [H, T]
        H: tl.constexpr,
        T: tl.constexpr,
        BLOCK_T: tl.constexpr  # power-of-two, e.g., 128
    ):
        h = tl.program_id(0)
        offs = tl.arange(0, BLOCK_T)
        # Compute row-wise max
        m = -float('inf')
        for t_start in range(0, T, BLOCK_T):
            idx = t_start + offs
            mask = idx < T
            logits_sub = tl.load(logits_ptr + h * T + idx, mask=mask, other=-float('inf'))
            m = tl.maximum(m, tl.max(logits_sub, axis=0))
        # Compute softmax
        for t_start in range(0, T, BLOCK_T):
            idx = t_start + offs
            mask = idx < T
            logits_sub = tl.load(logits_ptr + h * T + idx, mask=mask, other=-float('inf'))
            e = tl.exp(logits_sub - m)
            denom = tl.sum(e, axis=0)
            attn_sub = e / denom
            tl.store(attn_ptr + h * T + idx, attn_sub, mask=mask)


    @triton.jit
    def lse_row_kernel(
        logits_ptr,   # *f32 [H, T]
        lse_ptr,      # *f32 [H]
        H: tl.constexpr,
        T: tl.constexpr,
        BLOCK_T: tl.constexpr  # power-of-two, e.g., 128
    ):
        h = tl.program_id(0)
        offs = tl.arange(0, BLOCK_T)
        m = -float('inf')
        for t_start in range(0, T, BLOCK_T):
            idx = t_start + offs
            mask = idx < T
            logits_sub = tl.load(logits_ptr + h * T + idx, mask=mask, other=-float('inf'))
            m = tl.maximum(m, tl.max(logits_sub, axis=0))
        # sum exp(x - m)
        sum_exp = 0.0
        for t_start in range(0, T, BLOCK_T):
            idx = t_start + offs
            mask = idx < T
            logits_sub = tl.load(logits_ptr + h * T + idx, mask=mask, other=-float('inf'))
            sum_exp += tl.sum(tl.exp(logits_sub - m), axis=0)
        lse_val = m + tl.log(sum_exp)  # natural log
        # convert to base-2
        lse_val = lse_val / tl.log(2.0)
        tl.store(lse_ptr + h, lse_val)


    @triton.jit
    def matmul_row_kernel(
        attn_ptr,     # *f32 [H, T]
        Kc_ptr,       # *f32 [T, Dq]
        out_ptr,      # *f32 [H, Dq]
        H: tl.constexpr,
        T: tl.constexpr,
        Dq: tl.constexpr,      # 512
        BLOCK_D: tl.constexpr  # power-of-two, e.g., 128
    ):
        h = tl.program_id(0)
        # Accumulator for out[h, :]
        out_vec = tl.zeros((Dq,), dtype=tl.float32)
        offs_d = tl.arange(0, BLOCK_D)  # power-of-two
        for t_start in range(0, T, BLOCK_D):
            idx = t_start + offs_d
            mask_t = idx < T
            attn_sub = tl.load(attn_ptr + h * T + idx, mask=mask_t, other=0.0)  # [BLOCK_D]
            Kc_sub = tl.load(
                Kc_ptr + idx[:, None] * Dq + tl.arange(0, Dq)[None, :],
                mask=mask_t[:, None],
                other=0.0
            )  # [BLOCK_D, Dq]
            prod = attn_sub[:, None] * Kc_sub  # [BLOCK_D, Dq]
            out_vec += tl.sum(prod, axis=0)  # sum over BLOCK_D -> update accumulator
        tl.store(out_ptr + h * Dq + tl.arange(0, Dq), out_vec, mask=tl.arange(0, Dq) < Dq)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA
        device = q_nope.device
        assert device.type == 'cuda', "This Triton implementation requires CUDA tensors."

        B, H, Dq = q_nope.shape
        assert H == 16 and Dq == 512, "Expected q_nope shape [B, 16, 512]"
        _, H_qp, Dp = q_pe.shape
        assert H_qp == 16 and Dp == 64, "Expected q_pe shape [B, 16, 64]"
        N, num_p, Dq_cache = ckv_cache.shape
        assert num_p == 1 and Dq_cache == 512, "Expected ckv_cache shape [N, 1, 512]"
        N2, num_p2, Dp_cache = kpe_cache.shape
        assert num_p2 == 1 and Dp_cache == 64, "Expected kpe_cache shape [N, 1, 64]"

        # Output buffers
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # per-head output in f32, cast to bf16 at end
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV entries for this batch item: fill zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc and Kp for these tokens
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int64).contiguous()  # [L_tokens]
            Kc_b = ckv_cache[tok_idx].squeeze(1).contiguous().to(torch.float32)  # [L_tokens, 512]
            Kp_b = kpe_cache[tok_idx].squeeze(1).contiguous().to(torch.float32)  # [L_tokens, 64]

            # qn and qp for this batch
            qn = q_nope[b].contiguous().to(torch.float32)  # [16, 512]
            qp = q_pe[b].contiguous().to(torch.float32)    # [16, 64]

            # Allocate intermediates
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # [16, L_tokens]
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)    # [16, L_tokens]

            # Launch Triton kernels
            # 1) Fused logits
            grid_logits = (H,)
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # 2) LogSumExp per head
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                logits, lse[b],
                H=H, T=L_tokens,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # 3) Softmax per head
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                logits, attn,
                H=H, T=L_tokens,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # 4) Per-head matmul: out[h, :] = attn[h, :] @ Kc_b[:, :]
            out_b = torch.empty((H, Dq), dtype=torch.float32, device=device)
            matmul_row_kernel[(H,)](
                attn, Kc_b, out_b,
                H=H, T=L_tokens, Dq=Dq,
                BLOCK_D=128,
                num_warps=4, num_stages=2
            )

            # Store
            output[b] = out_b
            lse[b] = lse[b]  # already computed per head

        # Cast output back to bfloat16 to match original run’s output dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse.to(torch.float32)


def run(*args):
    return ModelNew()(*args)
