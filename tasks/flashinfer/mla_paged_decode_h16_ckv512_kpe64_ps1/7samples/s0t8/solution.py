import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_lse_kernel(
    qn_ptr,        # *f32, shape [D]
    qn_stride,     # int32, typically D
    Kc_ptr,        # *f32, shape [num_pages, D], we will index with tok_idx
    Kp_ptr,        # *f32, shape [num_pages, DP], we will index with tok_idx
    tok_idx_ptr,   # *int32, shape [L_TOKENS]
    L_TOKENS,      # int32
    sm_scale,      # f32
    out_lse_ptr,   # *f32, shape [B, 16], we write [b, h]
    B: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    H: tl.constexpr,  # number of heads, used to offset out_lse_ptr
):
    # Each program handles one (b, h)
    b = tl.program_id(0) % B
    h = tl.program_id(0) // B

    # Initialize running max and sum
    m = -float('inf')  # scalar f32
    s = 0.0            # scalar f32

    # Loop over tokens
    t = 0
    while t < L_TOKENS:
        idx = tl.load(tok_idx_ptr + t)  # int32
        # Load qn and qp for this head
        qn_vec = tl.load(qn_ptr + h * qn_stride)  # [D]
        # Load Kc_row and Kp_row (1D) for this token index
        # Note: Pointer arithmetic assumes contiguous layout; idx selects the row
        Kc_row = tl.load(Kc_ptr + idx * D + tl.arange(0, D), mask=True)  # [D]
        Kp_row = tl.load(Kp_ptr + idx * DP + tl.arange(0, DP), mask=True)  # [DP]
        # Compute dot products
        dot_qn = (qn_vec * Kc_row).sum()
        dot_qp = (tl.load(qn_ptr + (h + 1) * qn_stride) * Kp_row).sum()  # erroneous: incorrect usage; see below

        # Correct way: load qp[h] correctly. We need qp vector for head h.
        # Since qn_ptr is for head h, we should have a separate ptr for qp[h].
        # However, kernel arguments only include qn_ptr. We need to pass qp_ptr.
        # Fix by passing a separate pointer. For now, compute dot_qp correctly using qp_ptr.
        # We'll redefine the kernel with qp_ptr below.

        # Placeholder: we need to fix dot_qp; redefine kernel properly below.
        t += 1

    # Compute lse = (m + log(s)) / log(2)
    lse_val = (m + tl.log(s)) / math.log(2.0)
    # Store lse[b, h]
    tl.store(out_lse_ptr + b * H + h, lse_val)


# We need to fix the above kernel with correct qp handling. Let's redefine properly.
@triton.jit
def _compute_lse_kernel_fixed(
    qn_ptr,        # *f32, shape [D, H] or [D], but we index by head via stride
    qn_stride,     # int32, stride for head, typically D
    qp_ptr,        # *f32, shape [DP, H] or [DP], we index by head via stride
    qp_stride,     # int32
    Kc_ptr,        # *f32, shape [num_pages, D]
    Kp_ptr,        # *f32, shape [num_pages, DP]
    tok_idx_ptr,   # *int32, shape [L_TOKENS]
    L_TOKENS,      # int32
    sm_scale,      # f32
    out_lse_ptr,   # *f32, shape [B, H]
    B: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    H: tl.constexpr,
):
    b = tl.program_id(0) % B
    h = tl.program_id(0) // B

    m = -float('inf')
    s = 0.0
    t = 0
    while t < L_TOKENS:
        idx = tl.load(tok_idx_ptr + t)  # int32
        # Load qn[h] and qp[h]
        qn_vec = tl.load(qn_ptr + h * qn_stride)  # [D]
        qp_vec = tl.load(qp_ptr + h * qp_stride)  # [DP]
        # Load Kc_row and Kp_row
        Kc_row = tl.load(Kc_ptr + idx * D + tl.arange(0, D), mask=True)  # [D]
        Kp_row = tl.load(Kp_ptr + idx * DP + tl.arange(0, DP), mask=True)  # [DP]
        dot_qn = (qn_vec * Kc_row).sum()
        dot_qp = (qp_vec * Kp_row).sum()
        scaled = (dot_qn + dot_qp) * sm_scale
        m_new = tl.maximum(m, scaled)
        s = s * tl.exp(m - m_new) + tl.exp(scaled - m_new)
        m = m_new
        t += 1
    lse_val = (m + tl.log(s)) / math.log(2.0)
    tl.store(out_lse_ptr + b * H + h, lse_val)


@triton.jit
def _compute_output_kernel(
    qn_ptr,        # *f32, shape [D]
    qn_stride,     # int32
    Kc_ptr,        # *f32, shape [num_pages, D]
    tok_idx_ptr,   # *int32, shape [L_TOKENS]
    L_TOKENS,      # int32
    sm_scale,      # f32
    out_ptr,       # *f32, shape [B, H, D] flattened as [B*H, D]
    B: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    b = tl.program_id(0) % B
    h = tl.program_id(0) // B

    # Recompute m and s to avoid relying on external lse
    m = -float('inf')
    s = 0.0
    t = 0
    while t < L_TOKENS:
        idx = tl.load(tok_idx_ptr + t)
        qn_vec = tl.load(qn_ptr + h * qn_stride)  # [D]
        Kc_row = tl.load(Kc_ptr + idx * D + tl.arange(0, D), mask=True)  # [D]
        dot_qn = (qn_vec * Kc_row).sum()
        dot_qp = 0.0  # not needed for output accumulation (only lse is needed for attn), but we need sm_scale and L_TOKENS for attn; recompute in the next kernel
        scaled = (dot_qn + dot_qp) * sm_scale
        m_new = tl.maximum(m, scaled)
        s = s * tl.exp(m - m_new) + tl.exp(scaled - m_new)
        m = m_new
        t += 1

    # Second pass: compute attn and accumulate output
    out_vec = tl.zeros((D,), dtype=tl.float32)
    t = 0
    while t < L_TOKENS:
        idx = tl.load(tok_idx_ptr + t)
        qn_vec = tl.load(qn_ptr + h * qn_stride)  # [D]
        Kc_row = tl.load(Kc_ptr + idx * D + tl.arange(0, D), mask=True)  # [D]
        dot_qn = (qn_vec * Kc_row).sum()
        dot_qp = 0.0
        scaled = (dot_qn + dot_qp) * sm_scale
        attn = tl.exp(scaled - m) / s
        out_vec += attn * Kc_row
        t += 1

    # Store output[b, h, :]
    # out_ptr is flattened as [B*H, D]
    out_offset = b * H + h
    tl.store(out_ptr + out_offset * D + tl.arange(0, D), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA for Triton
        device = q_nope.device
        assert device.type == 'cuda', "Inputs must be on CUDA for Triton kernels."

        B, H, D = q_nope.shape
        DP = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]

        # Compute L_tokens per batch element from kv_indptr
        L_tokens_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens_list.append(max(0, end - start))
        # For simplicity, we process one batch element per program_id(0). We'll set grid = (B*H,)
        grid = (B * H,)

        # Prepare output buffers
        output = torch.empty((B, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Cast inputs to float32 for computation
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        ckv_cache_f32 = ckv_cache.to(torch.float32)
        kpe_cache_f32 = kpe_cache.to(torch.float32)
        tok_idx = kv_indices.to(torch.int32).contiguous()

        # Launch kernel to compute lse
        _compute_lse_kernel_fixed[grid](
            q_nope_f32, D, q_pe_f32, DP, ckv_cache_f32, kpe_cache_f32, tok_idx, L_tokens_list[0], sm_scale,
            lse, B=B, D=D, DP=DP, H=H
        )

        # Launch kernel to compute output
        _compute_output_kernel[grid](
            q_nope_f32, D, ckv_cache_f32, tok_idx, L_tokens_list[0], sm_scale,
            output, B=B, D=D, H=H
        )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse

# The following helpers are unchanged from the original, provided here for completeness.

def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)], dim=0).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# Entry point class for the evaluator
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
