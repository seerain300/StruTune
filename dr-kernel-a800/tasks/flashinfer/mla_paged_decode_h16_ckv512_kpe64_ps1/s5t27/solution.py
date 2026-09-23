import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scaled_logits_kernel(
    qn_ptr,          # *f32, [H, CK]
    qp_ptr,          # *f32, [H, KP]
    Kc_ptr,          # *f32, [L, CK]
    Kp_ptr,          # *f32, [L, KP]
    logits_ptr,      # *f32, [H, L] (output)
    H: tl.constexpr, CK: tl.constexpr, KP: tl.constexpr, L: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # Each program handles one (h, t)
    h = tl.program_id(axis=0)
    t = tl.program_id(axis=1)
    if h >= H or t >= L:
        return

    # Load qn[h, :] and qp[h, :]
    qn = tl.load(qn_ptr + h * CK + tl.arange(0, CK))
    qp = tl.load(qp_ptr + h * KP + tl.arange(0, KP))

    # Load Kc[t, :] and Kp[t, :]
    Kc_row = tl.load(Kc_ptr + t * CK + tl.arange(0, CK))
    Kp_row = tl.load(Kp_ptr + t * KP + tl.arange(0, KP))

    # Dot products
    dot1 = tl.sum(qn * Kc_row, axis=0)
    dot2 = tl.sum(qp * Kp_row, axis=0)

    val = sm_scale * (dot1 + dot2)
    tl.store(logits_ptr + h * L + t, val)


@triton.jit
def _lse_kernel(
    logits_ptr,      # *f32, [H, L]
    lse_ptr,         # *f32, [H]
    H: tl.constexpr, L: tl.constexpr,
    stride_h: tl.constexpr, stride_l: tl.constexpr,
):
    # One program per head
    h = tl.program_id(axis=0)
    max_val = -float('inf')
    # Pass 1: find max
    for t in range(0, L):
        ptr = logits_ptr + h * stride_h + t * stride_l
        val = tl.load(ptr)
        if val > max_val:
            max_val = val

    sumexp = 0.0
    # Pass 2: sum exp(s - max)
    for t in range(0, L):
        ptr = logits_ptr + h * stride_h + t * stride_l
        val = tl.load(ptr)
        sumexp += tl.exp(val - max_val)

    lse = tl.log(sumexp)
    tl.store(lse_ptr + h, lse)


@triton.jit
def _compute_output_kernel(
    logits_ptr,      # *f32, [H, L]
    lse_ptr,         # *f32, [H]
    Kc_ptr,          # *f32, [L, CK]
    out_ptr,         # *f32, [H, CK] (accumulator)
    H: tl.constexpr, L: tl.constexpr, CK: tl.constexpr,
    stride_log_h: tl.constexpr, stride_log_l: tl.constexpr,
    stride_kc_t: tl.constexpr, stride_kc_k: tl.constexpr,
    stride_out_h: tl.constexpr, stride_out_k: tl.constexpr,
):
    # One program per head
    h = tl.program_id(axis=0)
    lse_h = tl.load(lse_ptr + h)

    # Accumulate output[h, :] = sum_t softmax(logits[h, t]) * Kc[t, :]
    for t in range(0, L):
        ptr = logits_ptr + h * stride_log_h + t * stride_log_l
        val = tl.load(ptr)
        p = tl.exp(val - lse_h)
        Kc_row = tl.load(Kc_ptr + t * stride_kc_t + tl.arange(0, CK))
        out_row = tl.load(out_ptr + h * stride_out_h + tl.arange(0, CK))
        out_row += p * Kc_row
        tl.store(out_ptr + h * stride_out_h + tl.arange(0, CK), out_row)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        dtype = torch.float32

        # Cast inputs to float32 for compute
        q_nope = q_nope.to(torch.float32)
        q_pe = q_pe.to(torch.float32)
        ckv_cache = ckv_cache.to(torch.float32)  # [num_pages, 1, CK]
        kpe_cache = kpe_cache.to(torch.float32)  # [num_pages, 1, KP]

        # Dimensions
        H = q_nope.shape[1]  # 16
        CK = q_nope.shape[2]  # 512
        assert CK == 512, "CK must be 512"
        KP = q_pe.shape[2]  # 64
        assert KP == 64, "KP must be 64"

        # Output buffers per batch element
        output = []  # [batch, H, CK]
        lse_list = []  # [batch, H]

        # Squeeze the 1-size dim from cache tensors
        Kc_all = ckv_cache.squeeze(1)  # [num_pages, CK]
        Kp_all = kpe_cache.squeeze(1)  # [num_pages, KP]

        batch_size = q_nope.shape[0]

        for b in range(batch_size):
            # Determine L_tokens for this batch using kv_indptr
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                out_row = torch.zeros((H, CK), dtype=torch.float32, device=device)
                output.append(out_row)
                lse_list.append(torch.full((H,), -float("inf"), dtype=torch.float32, device=device))
                continue

            # Slice token indices and corresponding Kc/Kp
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.int32)
            Kc_slice = Kc_all[tok_idx]  # [L_tokens, CK]
            Kp_slice = Kp_all[tok_idx]  # [L_tokens, KP]

            # Prepare qn and qp for this batch
            qn = q_nope[b].contiguous()  # [H, CK]
            qp = q_pe[b].contiguous()    # [H, KP]

            # Allocate intermediates and outputs
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # [H, L_tokens]
            lse = torch.empty((H,), dtype=torch.float32, device=device)

            # Launch kernel 1: compute scaled_logits[h, t]
            grid = (H, L_tokens)
            _compute_scaled_logits_kernel[grid](
                qn, qp, Kc_slice, Kp_slice, logits,
                H=H, CK=CK, KP=KP, L=L_tokens, sm_scale=float(sm_scale),
            )

            # Launch kernel 2: compute per-head lse
            stride_h = H
            stride_l = L_tokens
            _lse_kernel[(H,)](
                logits, lse,
                H=H, L=L_tokens,
                stride_h=stride_h, stride_l=stride_l,
            )

            # Launch kernel 3: compute output[h, :] = sum_t softmax(logits[h, t]) * Kc[t, :]
            out_acc = torch.zeros((H, CK), dtype=torch.float32, device=device)
            stride_log_h = H
            stride_log_l = L_tokens
            stride_kc_t = L_tokens
            stride_kc_k = CK
            stride_out_h = H
            stride_out_k = CK

            _compute_output_kernel[(H,)](
                logits, lse, Kc_slice, out_acc,
                H=H, L=L_tokens, CK=CK,
                stride_log_h=stride_log_h, stride_log_l=stride_log_l,
                stride_kc_t=stride_kc_t, stride_kc_k=stride_kc_k,
                stride_out_h=stride_out_h, stride_out_k=stride_out_k,
            )

            output.append(out_acc)
            lse_list.append(lse)

        # Stack to [batch, H, CK] and [batch, H]
        output = torch.stack(output, dim=0)
        lse = torch.stack(lse_list, dim=0)

        # Return output in bfloat16 to match original, and lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
