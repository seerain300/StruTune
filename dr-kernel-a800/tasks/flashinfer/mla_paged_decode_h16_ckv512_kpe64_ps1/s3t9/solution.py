import torch
import triton
import triton.language as tl
import math

# Triton kernel: compute logits[h, t] = qn[h, :] @ Kc[t, :] + qp[h, :] @ Kp[t, :]
# One program per head h.
@triton.jit
def fused_logits_kernel(
    q_nope_ptr,  # *f32, [B, H, Dq]
    q_pe_ptr,    # *f32, [B, H, Dp]
    Kc_ptr,      # *f32, [T, Dq]
    Kp_ptr,      # *f32, [T, Dp]
    logits_ptr,  # *f32, [H, T]
    B, T,        # int32 runtime
    Dq: tl.constexpr,  # 512
    Dp: tl.constexpr,  # 64
    BLOCK_T: tl.constexpr,  # 256
):
    h = tl.program_id(0)
    # Preload q vectors: qn[h, :] and qp[h, :]
    # Indices for q: q_nope is [B, H, Dq], q_pe is [B, H, Dp]
    # We don't know B in this kernel, but h is valid for all batch entries,
    # so we load per h. We need to load q vectors for the current b by making
    # q_nope_ptr and q_pe_ptr be per-batch. Here, we assume q_nope_ptr and q_pe_ptr
    # point to [B, H, D] and we load using the fact that stride along H is Dq and Dp.
    # However, Triton kernels get 1D pointers; we instead pass q_nope[b, h, :] and q_pe[b, h, :]
    # via host-side pointers. To keep the kernel simple and re-useable, we assume
    # q_nope_ptr and q_pe_ptr are flattened pointers for each (b,h) and host will
    # compute them accordingly. For simplicity, assume host passes pointers for b=0
    # and we rely on q_nope[0,h, :], q_pe[0,h, :]. This is a simplification; in ModelNew.forward,
    # we will call the kernel per batch b with correct pointers.
    # To keep correctness, we won't use q_nope_ptr/q_pe_ptr here and instead pass
    # qn and qp as vectors loaded from the host via temporary buffers outside this kernel.
    # So we remove the following and just compute using qn and qp passed as vectors:
    # qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq))  # [Dq]
    # qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp))   # [Dp]
    # For now, assume qn and qp are passed as global vectors via logits_ptr or separate buffers.
    # To avoid confusion, we will instead implement the kernel assuming qn and qp are loaded
    # from q_nope_ptr and q_pe_ptr for each b,h by host, which is handled outside this kernel.
    # Therefore, we remove these loads here and rely on host to pass qn, qp vectors.
    # The above block is a placeholder; in practice, we do not rely on q_nope_ptr/q_pe_ptr here.

    # Since we cannot access b here, we redesign: we will launch kernels per batch b separately
    # with correct q_nope[b,h, :] and q_pe[b,h, :]. This kernel will not be used as-is.
    # Instead, we provide a correct kernel below that assumes qn, qp are passed as vectors.

# Below is the corrected kernel that assumes qn and qp are passed as vectors per head.
@triton.jit
def fused_logits_kernel_vec(
    qn_ptr,      # *f32, [Dq]
    qp_ptr,      # *f32, [Dp]
    Kc_ptr,      # *f32, [T, Dq]
    Kp_ptr,      # *f32, [T, Dp]
    logits_ptr,  # *f32, [H, T]
    T: tl.constexpr,         # number of tokens
    Dq: tl.constexpr,        # 512
    Dp: tl.constexpr,        # 64
    BLOCK_T: tl.constexpr,   # 256
):
    h = tl.program_id(0)
    # Preload q vectors (assuming H is 1 in this kernel; we will launch H programs).
    qn = tl.load(qn_ptr + tl.arange(0, Dq))   # [Dq]
    qp = tl.load(qp_ptr + tl.arange(0, Dp))   # [Dp]

    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T

        # Accumulate logits for this tile [BLOCK_T]
        logits_tile = tl.zeros([BLOCK_T], dtype=tl.float32)

        # Dot over Dq: sum_k qn[k] * Kc[t, k]
        # Kc_sub: [BLOCK_T, Dq]
        Kc_sub = tl.load(Kc_ptr + t_idx[:, None] * Dq + tl.arange(0, Dq)[None, :], mask=mask_t[:, None], other=0.0)
        # qn: [Dq]
        # elementwise multiply and sum over Dq axis
        logits_tile += tl.sum(Kc_sub * qn[None, :], axis=1)

        # Dot over Dp: sum_k qp[k] * Kp[t, k]
        Kp_sub = tl.load(Kp_ptr + t_idx[:, None] * Dp + tl.arange(0, Dp)[None, :], mask=mask_t[:, None], other=0.0)
        logits_tile += tl.sum(Kp_sub * qp[None, :], axis=1)

        # Store logits[h, t_start:t_start+BLOCK_T]
        tl.store(logits_ptr + h * T + t_idx, logits_tile, mask=mask_t)

# Triton kernel: softmax per row for scaled_logits[h, :]
@triton.jit
def softmax_row_kernel(
    scaled_ptr,  # *f32, [H, T]
    attn_ptr,    # *f32, [H, T]
    H: tl.constexpr,  # number of heads (should be 16)
    T: tl.constexpr,  # number of tokens
    BLOCK_T: tl.constexpr,  # e.g., 256
):
    h = tl.program_id(0)
    # One program per row h
    # Loop over T in tiles
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_idx < T
        vals = tl.load(scaled_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))
        # Compute row max
        row_max = tl.max(vals, axis=0)
        vals = vals - row_max
        exp_vals = tl.exp(vals)
        denom = tl.sum(exp_vals, axis=0)
        attn_vals = exp_vals / denom
        tl.store(attn_ptr + h * T + t_idx, attn_vals, mask=mask_t)

# Triton kernel: logsumexp per row for scaled_logits[h, :], return logsumexp / ln(2)
@triton.jit
def lse_row_kernel(
    scaled_ptr,  # *f32, [H, T]
    lse_ptr,     # *f32, [H]
    H: tl.constexpr,  # number of heads (should be 16)
    T: tl.constexpr,  # number of tokens
    BLOCK_T: tl.constexpr,  # e.g., 256
):
    h = tl.program_id(0)
    # Compute max across row
    row_max = -float('inf')
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_idx < T
        vals = tl.load(scaled_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))
        tile_max = tl.max(vals, axis=0)
        row_max = tl.maximum(row_max, tile_max)
    # Compute sum of exp(vals - row_max)
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_idx < T
        vals = tl.load(scaled_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(vals - row_max), axis=0)
    lse_val = row_max + tl.log(sum_exp)
    # Divide by ln(2)
    ln2 = 0.6931471805599453  # math.log(2.0)
    tl.store(lse_ptr + h, lse_val / ln2)

# Triton kernel: attn @ Kc for head h -> output[h, :]
@triton.jit
def attn_matmul_kernel(
    attn_ptr,    # *f32, [H, T]
    Kc_ptr,      # *f32, [T, Dq]
    out_ptr,     # *f32, [H, Dq]
    H: tl.constexpr,         # number of heads (16)
    T: tl.constexpr,         # number of tokens
    Dq: tl.constexpr,        # 512
    BLOCK_D: tl.constexpr,   # 128
    BLOCK_T: tl.constexpr,   # 256
):
    h = tl.program_id(0)
    # One program per head
    # Accumulator for output [Dq]
    acc = tl.zeros([Dq], dtype=tl.float32)
    # Tile over Dq
    for d_start in range(0, Dq, BLOCK_D):
        d_idx = d_start + tl.arange(0, BLOCK_D)
        mask_d = d_idx < Dq
        # Tile over T
        for t_start in range(0, T, BLOCK_T):
            t_idx = t_start + tl.arange(0, BLOCK_T)
            mask_t = t_idx < T
            # Load attn[h, t_idx] -> [BLOCK_T]
            attn_sub = tl.load(attn_ptr + h * T + t_idx, mask=mask_t, other=0.0)
            # Load Kc[t_idx, d_idx] -> [BLOCK_T, BLOCK_D]
            Kc_sub = tl.load(Kc_ptr + t_idx[:, None] * Dq + d_idx[None, :],
                             mask=mask_t[:, None], other=0.0)
            # Accumulate acc over t in tile: acc += sum_t attn_sub[t] * Kc_sub[t, :]
            # Implement reduction over BLOCK_T:
            for t_local in range(0, BLOCK_T):
                if (t_start + t_local) < T:
                    acc += attn_sub[t_local] * Kc_sub[t_local, :]
        # Store acc partial to out[h, d_start:d_start+BLOCK_D]
        tl.store(out_ptr + h * Dq + d_idx, acc, mask=mask_d)

# Entry point model
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Move to CUDA (Triton requires CUDA tensors)
        device = torch.device('cuda')
        q_nope = q_nope.to(device)
        q_pe = q_pe.to(device)
        ckv_cache = ckv_cache.to(device)
        kpe_cache = kpe_cache.to(device)
        kv_indptr = kv_indptr.to(device)
        kv_indices = kv_indices.to(device)

        # Ensure contiguity and cast to float32 for compute
        q_nope = q_nope.contiguous().to(torch.float32)  # [B, H, Dq]
        q_pe = q_pe.contiguous().to(torch.float32)      # [B, H, Dp]
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [N, Dq] -> may need gather
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [N, Dp]

        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]   # number of heads (16)
        Dq = q_nope.shape[2]  # 512
        Dp = q_pe.shape[2]    # 64

        output = torch.empty((batch_size, H, Dq), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # We will compute per batch element. Since Triton kernels below assume qn, qp vectors,
        # we need to extract them for each b. To keep kernels simple, we precompute qn and qp
        # vectors as tensors and pass to kernels.

        for b in range(batch_size):
            # Compute number of tokens used by batch b: L_tokens = kv_indptr[b+1] - kv_indptr[b]
            L_tokens = int(kv_indptr[b+1].item()) - int(kv_indptr[b].item())

            # If no tokens used, output zeros and lse -inf
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].fill_(-float('inf'))
                continue

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b+1]].to(torch.int32)
            Kc = Kc_all[tok_idx]  # [L_tokens, Dq]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Preload qn[h, :] and qp[h, :] as vectors
            qn_vec = q_nope[b, :, :].contiguous().to(torch.float32)  # [H, Dq] -> flatten to vector per head?
            # We need qn per head as 1D vectors. We can construct per head:
            qn_list = [qn_vec[h, :] for h in range(H)]  # list of 1D tensors
            qp_list = [q_pe[b, h, :].contiguous().to(torch.float32) for h in range(H)]

            # Allocate logits and attn buffers [H, T]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch fused_logits_kernel_vec: one program per head
            grid = (H,)
            fused_logits_kernel_vec[grid](
                qn_list[0].to(torch.float32),  # Triton expects pointers; here we pass tensors, but kernel signature expects *f32
                qp_list[0].to(torch.float32),
                Kc, Kp, logits,
                T=L_tokens, Dq=Dq, Dp=Dp, BLOCK_T=256
            )

            # Scale logits
            scaled_logits = logits * sm_scale

            # Launch softmax_row_kernel to compute attn
            softmax_row_kernel[grid](
                scaled_logits, attn,
                H=H, T=L_tokens, BLOCK_T=256
            )

            # Launch lse_row_kernel to compute per-head lse (we keep lse as output for consistency)
            lse_row_kernel[grid](
                scaled_logits, lse[b],
                H=H, T=L_tokens, BLOCK_T=256
            )

            # Launch attn_matmul_kernel to compute output[h, :] = attn[h, :] @ Kc[:, :]
            attn_matmul_kernel[grid](
                attn, Kc, output[b],
                H=H, T=L_tokens, Dq=Dq, BLOCK_D=128, BLOCK_T=256
            )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
