# task: flashinfer/mla_paged_decode_h16_ckv512_kpe64_ps1
# batch: stts
# pass_at_1: 0.12
# final_geomean_speedup(A800, official re-eval): 0.215
# provenance: sample s6 reward=0.721 feedback_geomean=0.207 turn=36
import math
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_qn_KcT_fp32_kernel(
    qn_ptr,     # *fp32, [H, D], contiguous
    Kc_ptr,     # *fp32, [L_tokens, D], contiguous
    acc_ptr,    # *fp32, [H, L_tokens], contiguous
    H: tl.constexpr,         # number of heads (16)
    D: tl.constexpr,         # head_dim_ckv (512)
    L_tokens: tl.constexpr,  # number of tokens for this batch element (runtime, but we pass it as constexpr for simplicity)
):
    # 2D launch grid: (h, t)
    h = tl.program_id(0)
    t = tl.program_id(1)

    acc = 0.0
    for kk in range(0, D):
        q = tl.load(qn_ptr + h * D + kk)  # qn[h, kk]
        k = tl.load(Kc_ptr + t * D + kk)  # Kc[t, kk]
        acc += q * k
    tl.store(acc_ptr + h * L_tokens + t, acc)


@triton.jit
def matmul_qp_KpT_fp32_kernel(
    qp_ptr,     # *fp32, [H, Dp], contiguous
    Kp_ptr,     # *fp32, [L_tokens, Dp], contiguous
    acc_ptr,    # *fp32, [H, L_tokens], contiguous
    H: tl.constexpr,         # number of heads (16)
    Dp: tl.constexpr,        # head_dim_kpe (64)
    L_tokens: tl.constexpr,  # number of tokens for this batch element
):
    h = tl.program_id(0)
    t = tl.program_id(1)

    acc = 0.0
    for kk in range(0, Dp):
        q = tl.load(qp_ptr + h * Dp + kk)
        k = tl.load(Kp_ptr + t * Dp + kk)
        acc += q * k
    tl.store(acc_ptr + h * L_tokens + t, acc)


@triton.jit
def add_two_accs_fp32_kernel(
    acc1_ptr,    # *fp32, [H, L_tokens], contiguous
    acc2_ptr,    # *fp32, [H, L_tokens], contiguous
    sum_ptr,     # *fp32, [H, L_tokens], contiguous
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    t = tl.program_id(1)
    v1 = tl.load(acc1_ptr + h * L_tokens + t)
    v2 = tl.load(acc2_ptr + h * L_tokens + t)
    tl.store(sum_ptr + h * L_tokens + t, v1 + v2)


@triton.jit
def scale_logits_fp32_kernel(
    sum_ptr,      # *fp32, [H, L_tokens], contiguous
    scale,        # fp32 scalar
    out_ptr,      # *fp32, [H, L_tokens], contiguous
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    t = tl.program_id(1)
    v = tl.load(sum_ptr + h * L_tokens + t)
    v = v * scale
    tl.store(out_ptr + h * L_tokens + t, v)


@triton.jit
def rowwise_lse_fp32_kernel(
    logits_ptr,   # *fp32, [H, L_tokens], contiguous
    lse_ptr,      # *fp32, [H], contiguous
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    m = -float("inf")
    sum_exp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        m = tl.where(val > m, val, m)
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        exp_val = tl.exp(val - m)
        sum_exp += exp_val
    lse_val = m + tl.log(sum_exp)  # natural logsumexp
    ln2 = 0.6931471805599453
    lse_val = lse_val / ln2  # base-2 logsumexp
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def softmax_row_fp32_kernel(
    logits_ptr,   # *fp32, [H, L_tokens], contiguous
    sm_ptr,       # *fp32, [H, L_tokens], contiguous
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    m = -float("inf")
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        m = tl.where(val > m, val, m)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        exp_val = tl.exp(val - m)
        sum_exp += exp_val
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sm_val = tl.exp(val - m) / sum_exp
        tl.store(sm_ptr + h * L_tokens + t, sm_val)


@triton.jit
def output_matmul_fp32_kernel(
    sm_ptr,       # *fp32, [H, L_tokens], contiguous
    Kc_ptr,       # *fp32, [L_tokens, D], contiguous
    out_ptr,      # *fp32, [H, D], contiguous
    H: tl.constexpr,
    D: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # 2D grid over (h, d)
    h = tl.program_id(0)
    d = tl.program_id(1)
    acc = 0.0
    for t in range(0, L_tokens):
        s = tl.load(sm_ptr + h * L_tokens + t)
        k = tl.load(Kc_ptr + t * D + d)
        acc += s * k
    tl.store(out_ptr + h * D + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA device"

        # Cast q_nope, q_pe to fp32 for computation
        q_nope_fp32 = q_nope.to(torch.float32).contiguous()
        q_pe_fp32 = q_pe.to(torch.float32).contiguous()

        # Squeeze "num_pages" dimension (original code uses squeeze(1) -> num_pages=1)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_tokens, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_tokens, head_dim_kpe]

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64

        device = q_nope.device
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        H = num_qo_heads
        D = head_dim_ckv
        Dp = head_dim_kpe
        ln2 = 0.6931471805599453  # 1 / ln(2)

        for b in range(batch_size):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                lse[b] = -float("inf")
                output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
                continue

            # Gather token indices and corresponding keys
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.long).contiguous()  # [L_tokens]
            Kc = Kc_all[tok_idx].to(torch.float32).contiguous()  # [L_tokens, D]
            Kp = Kp_all[tok_idx].to(torch.float32).contiguous()  # [L_tokens, Dp]

            # Prepare qn and qp for this batch element
            qn = q_nope_fp32[b]  # [H, D]
            qp = q_pe_fp32[b]    # [H, Dp]

            # Allocate buffers
            acc1 = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # qn @ Kc.T
            acc2 = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # qp @ Kp.T
            sum_logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            logits_scaled = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            sm = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            out = torch.empty((H, D), dtype=torch.float32, device=device)

            # Launch Triton kernels
            # 1) qn @ Kc.T
            grid_qn = (H, L_tokens)
            matmul_qn_KcT_fp32_kernel[grid_qn](qn, Kc, acc1, H=H, D=D, L_tokens=L_tokens)

            # 2) qp @ Kp.T
            grid_qp = (H, L_tokens)
            matmul_qp_KpT_fp32_kernel[grid_qp](qp, Kp, acc2, H=H, Dp=Dp, L_tokens=L_tokens)

            # 3) sum acc1 + acc2
            grid_sum = (H, L_tokens)
            add_two_accs_fp32_kernel[grid_sum](acc1, acc2, sum_logits, H=H, L_tokens=L_tokens)

            # 4) scale by sm_scale
            grid_scale = (H, L_tokens)
            scale_logits_fp32_kernel[grid_scale](sum_logits, sm_scale, logits_scaled, H=H, L_tokens=L_tokens)

            # 5) rowwise lse (base-2)
            grid_lse = (H,)
            rowwise_lse_fp32_kernel[grid_lse](logits_scaled, lse[b], H=H, L_tokens=L_tokens)

            # 6) softmax over tokens for each head
            grid_sm = (H, L_tokens)
            softmax_row_fp32_kernel[grid_sm](logits_scaled, sm, H=H, L_tokens=L_tokens)

            # 7) output = softmax @ Kc
            grid_out = (H, D)
            output_matmul_fp32_kernel[grid_out](sm, Kc, out, H=H, D=D, L_tokens=L_tokens)

            # Store output for this batch element
            output[b] = out.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
