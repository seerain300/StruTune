import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits[h, t] = dot(qn[h, :], Kc[t, :]) + dot(qp[h, :], Kp[t, :])
# Inputs:
#   q_nope_ptr: [H, Dq], float32
#   q_pe_ptr:   [H, Dp], float32
#   Kc_ptr:     [T, Dq], float32
#   Kp_ptr:     [T, Dp], float32
#   logits_ptr: [H, T], float32 (output)
@triton.jit
def fused_logits_kernel(q_nope_ptr, q_pe_ptr, Kc_ptr, Kp_ptr, logits_ptr,
                         H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr, Dp: tl.constexpr,
                         BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head

    # Preload q vectors for this head (Dq/Dp are constexpr)
    qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq))  # [Dq]
    qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp))   # [Dp]

    offs_t = tl.arange(0, BLOCK_T)

    # Loop over tokens in tiles
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T

        # Load Kc and Kp tiles
        Kc_sub = tl.load(Kc_ptr + t_idx * Dq + tl.arange(0, Dq), mask=mask_t, other=0.0)  # [BLOCK_T, Dq]
        Kp_sub = tl.load(Kp_ptr + t_idx * Dp + tl.arange(0, Dp), mask=mask_t, other=0.0)  # [BLOCK_T, Dp]

        # Dot-products: sum over feature axes
        # qn: [Dq], Kc_sub[:, d] via broadcasting -> [BLOCK_T]
        dot_qn = tl.sum(qn * Kc_sub, axis=0)  # [BLOCK_T]
        dot_qp = tl.sum(qp * Kp_sub, axis=0)  # [BLOCK_T]

        # Store to logits[h, t_start : t_start+BLOCK_T]
        tl.store(logits_ptr + h * T + t_idx, dot_qn, mask=mask_t)


# Triton kernel: compute lse[h] = logsumexp(logits[h, :]) * sm_scale / ln(2)
@triton.jit
def lse_row_kernel(logits_ptr, lse_ptr,
                   H: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr, sm_scale: tl.float32):
    h = tl.program_id(0)
    row = logits_ptr + h * T
    # Initialize max and sum in fp32
    max_val = -float('inf')
    sum_exp = 0.0

    offs_t = tl.arange(0, BLOCK_T)

    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        vals = tl.load(row + t_idx, mask=mask_t, other=-float('inf'))
        # Reduce to scalar
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: compute sum exp(vals - max)
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        vals = tl.load(row + t_idx, mask=mask_t, other=-float('inf'))
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)

    lse_val = tl.log(sum_exp) + max_val  # logsumexp
    lse_val = lse_val * sm_scale / math.log(2.0)
    tl.store(lse_ptr + h, lse_val)


# Triton kernel: softmax per row (head): attn[h, t] = exp(logits[h, t] - max) / sum
@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr,
                        H: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr, sm_scale: tl.float32):
    h = tl.program_id(0)
    row = logits_ptr + h * T
    attn_row = attn_ptr + h * T

    # First pass: row max
    max_val = -float('inf')
    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        vals = tl.load(row + t_idx, mask=mask_t, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: sum of exp(vals - max)
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        vals = tl.load(row + t_idx, mask=mask_t, other=-float('inf'))
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)

    # Third pass: write normalized attn
    inv_sum = 1.0 / sum_exp
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        vals = tl.load(row + t_idx, mask=mask_t, other=-float('inf'))
        attn_vals = tl.exp(vals - max_val) * inv_sum
        tl.store(attn_row + t_idx, attn_vals, mask=mask_t)


# Triton kernel: per-row matmul out[h, :] = attn[h, :] @ Kc[:, :]
@triton.jit
def matmul_row_kernel(attn_ptr, Kc_ptr, out_ptr,
                      H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr,
                      BLOCK_D: tl.constexpr, BLOCK_T2: tl.constexpr):
    h = tl.program_id(0)

    # out[h, :] initialization
    out = tl.zeros((Dq,), dtype=tl.float32)

    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_T2)

    # Loop over tokens for reduction
    for t_start in range(0, T, BLOCK_T2):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        attn_sub = tl.load(attn_ptr + h * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T2]

        # Load Kc tiles [BLOCK_T2, Dq]
        Kc_sub = tl.load(Kc_ptr + t_idx * Dq + offs_d, mask=mask_t[:, None], other=0.0)  # [BLOCK_T2, Dq]

        # Accumulate dot for each d
        # attn_sub[j] * Kc_sub[j, d] -> sum over j
        for d_chunk_start in range(0, Dq, BLOCK_D):
            d_idx = d_chunk_start + offs_d
            mask_d = d_idx < Dq
            # Kc_sub for this chunk: [BLOCK_T2, BLOCK_D]
            Kc_sub_chunk = Kc_sub[:, d_chunk_start:d_chunk_start+BLOCK_D]  # [BLOCK_T2, BLOCK_D]
            # attn_sub_chunk: [BLOCK_T2]
            attn_sub_chunk = attn_sub  # [BLOCK_T2]
            # Multiply and sum over j: result [BLOCK_D]
            acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            # For each j in the tile, accumulate
            for j in range(0, BLOCK_T2):
                # vectorized j: each row of Kc_sub_chunk[j, :] * attn_sub_chunk[j]
                # We need to multiply each row j by its corresponding attn_sub_chunk[j] and accumulate
                # Implement as elementwise product and reduction:
                # Extract scalar attn[j]
                a_j = attn_sub_chunk[j]
                # Multiply Kc_sub_chunk[j, :] by a_j
                row_j = Kc_sub_chunk[j, :]  # [BLOCK_D]
                acc += a_j * row_j
            # Add to out for valid d
            out[d_chunk_start:d_chunk_start+BLOCK_D] += acc * mask_d

    # Store final out[h, :]
    tl.store(out_ptr + h * Dq + offs_d, out, mask=tl.arange(0, Dq) < Dq)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Fallback if Triton not available or tensors not on CUDA
        if (not TRITON_AVAILABLE) or (q_nope.device.type != 'cuda'):
            # Fallback to PyTorch implementation (not recommended in evaluation but kept for robustness)
            batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
            head_dim_kpe = q_pe.shape[-1]
            assert num_qo_heads == 16, "num_qo_heads must be 16"
            assert head_dim_ckv == 512, "head_dim_ckv must be 512"
            assert head_dim_kpe == 64, "head_dim_kpe must be 64"
            device = q_nope.device
            output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
            lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

            for b in range(batch_size):
                L = int((kv_indptr[b + 1] - kv_indptr[b]).item())
                if L <= 0:
                    lse[b] = 0.0
                    output[b] = 0
                    continue

                tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.long)
                Kc_b = ckv_cache[tok_idx].to(torch.float32)  # [L, 512]
                Kp_b = kpe_cache[tok_idx].to(torch.float32)  # [L, 64]

                qn = q_nope[b].to(torch.float32)  # [16, 512]
                qp = q_pe[b].to(torch.float32)    # [16, 64]

                logits = (qn @ Kc_b.transpose(0, 1)) + (qp @ Kp_b.transpose(0, 1))  # [16, L]
                logits_scaled = logits * float(sm_scale)

                # lse
                lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                attn = torch.softmax(logits_scaled, dim=-1)  # [16, L]
                out = attn @ Kc_b  # [16, 512]
                output[b] = out.to(torch.bfloat16)

            return output, lse

        # Triton path
        device = q_nope.device
        # Constants
        H = 16
        Dq = 512
        Dp = 64
        batch_size = q_nope.shape[0]

        output = torch.empty((batch_size, H, Dq), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Compute lengths and indices
            L = int((kv_indptr[b + 1] - kv_indptr[b]).item())
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.int64)
            # Load Kc and Kp for this batch element
            Kc_b = ckv_cache[tok_idx].to(torch.float32).contiguous()  # [L, 512]
            Kp_b = kpe_cache[tok_idx].to(torch.float32).contiguous()  # [L, 64]

            # q vectors for this batch
            qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

            # Allocate logits [H, L]
            logits = torch.empty((H, L), dtype=torch.float32, device=device)

            # Launch fused_logits_kernel: one program per head
            grid_logits = (H,)
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, T=L, Dq=Dq, Dp=Dp,
                BLOCK_T=256,
                num_warps=4, num_stages=2
            )

            # Compute lse per head
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                logits, lse[b],
                H=H, T=L, BLOCK_T=256, sm_scale=float(sm_scale),
                num_warps=4, num_stages=2
            )

            # Softmax per head
            attn = torch.empty((H, L), dtype=torch.float32, device=device)
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                logits, attn,
                H=H, T=L, BLOCK_T=256, sm_scale=float(sm_scale),
                num_warps=4, num_stages=2
            )

            # Per-head matmul out[h, :] = attn[h, :] @ Kc_b[:, :]
            out_b = torch.empty((H, Dq), dtype=torch.float32, device=device)
            grid_mm = (H,)
            matmul_row_kernel[grid_mm](
                attn, Kc_b, out_b,
                H=H, T=L, Dq=Dq,
                BLOCK_D=128, BLOCK_T2=256,
                num_warps=4, num_stages=2
            )

            # Store into output tensor
            output[b] = out_b

        # Cast output to bfloat16 as in original
        output_bf16 = output.to(torch.bfloat16)

        return output_bf16, lse


# The rest (get_inputs, fused_operator) remain unchanged
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

# Entry point Model
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
