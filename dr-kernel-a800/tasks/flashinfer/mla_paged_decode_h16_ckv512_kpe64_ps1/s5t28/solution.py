import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scaled_logits_kernel(
    A_qn, A_qp, B_qc, B_qp, Out,
    H: tl.constexpr, CK: tl.constexpr, KP: tl.constexpr, L: tl.constexpr, sm_scale: tl.constexpr
):
    # Grid is (H, L): each program handles one (h, t) pair
    pid_h = tl.program_id(axis=0)
    pid_t = tl.program_id(axis=1)
    # Compute dot products: qn[pid_h] · Kc[pid_t] and qp[pid_h] · Kp[pid_t]
    sum1 = 0.0
    sum2 = 0.0
    # qn[pid_h] is length CK, Kc[pid_t] is length CK
    for k in range(0, CK):
        a = tl.load(A_qn + pid_h * CK + k)
        b = tl.load(B_qc + pid_t * CK + k)
        sum1 += a * b
    for k in range(0, KP):
        a = tl.load(A_qp + pid_h * KP + k)
        b = tl.load(B_qp + pid_t * KP + k)
        sum2 += a * b
    val = sm_scale * (sum1 + sum2)
    tl.store(Out + pid_h * L + pid_t, val)


@triton.jit
def _lse_kernel(
    Logits, Lse,
    H: tl.constexpr, L: tl.constexpr,
    stride_log_h, stride_log_l
):
    # One program per head; compute natural logsumexp over tokens
    h = tl.program_id(axis=0)
    # Pass 1: find max
    max_val = -float('inf')
    for t in range(0, L):
        ptr = Logits + h * stride_log_h + t * stride_log_l
        val = tl.load(ptr)
        if val > max_val:
            max_val = val
    # Pass 2: sumexp
    sumexp = 0.0
    for t in range(0, L):
        ptr = Logits + h * stride_log_h + t * stride_log_l
        val = tl.load(ptr)
        sumexp += tl.exp(val - max_val)
    lse = tl.log(sumexp)
    tl.store(Lse + h, lse)


@triton.jit
def _compute_output_kernel(
    Logits, Lse, B_qc, Out,
    H: tl.constexpr, CK: tl.constexpr, L: tl.constexpr,
    stride_log_h, stride_log_l,
    stride_bc_t, stride_bc_k,
    stride_out_h, stride_out_k
):
    # One program per head; accumulate output
    h = tl.program_id(axis=0)
    lse_h = tl.load(Lse + h)
    out_acc = tl.zeros((CK,), dtype=tl.float32)
    for t in range(0, L):
        ptr = Logits + h * stride_log_h + t * stride_log_l
        val = tl.load(ptr)  # scaled logits for this head and token
        soft = tl.exp(val - lse_h)
        # Kc[t, :] row
        for k in range(0, CK):
            b = tl.load(B_qc + t * CK + k)
            out_acc[k] += soft * b
    # Store accumulated output
    out_ptr = Out + h * CK
    for k in range(0, CK):
        tl.store(out_ptr + k, out_acc[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and float32 for compute
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be CUDA"
        device = q_nope.device

        # Cast inputs to float32 for Triton compute
        q_nope = q_nope.contiguous().to(torch.float32)   # [B, H, CK]
        q_pe = q_pe.contiguous().to(torch.float32)      # [B, H, KP]
        ckv_cache = ckv_cache.contiguous().to(torch.float32)  # [num_pages, 1, CK]
        kpe_cache = kpe_cache.contiguous().to(torch.float32)  # [num_pages, 1, KP]

        # Dimensions
        B, H, CK = q_nope.shape
        _, _, KP = q_pe.shape
        # We assume single KV slice per batch element using kv_indptr (as in original code)
        # Allocate output and lse buffers
        output = torch.empty((B, H, CK), dtype=torch.float32, device=device)  # will be cast to bfloat16
        lse = torch.empty((B, H), dtype=torch.float32, device=device)         # per-(b,h) logsumexp (natural log)

        # For each batch b, determine L_tokens and token indices
        for b in range(B):
            # L_tokens is distance between kv_indptr[b] and kv_indptr[b+1]
            # Note: kv_indptr shape is [len_indptr], and len_indptr == B + 1
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No valid tokens for this batch element; output zero and lse -inf
                output[b] = torch.zeros((H, CK), dtype=torch.float32, device=device)
                lse[b] = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32).contiguous()  # [L_tokens]

            # Prepare A_qn (qn), B_qc (Kc), A_qp (qp), B_qp (Kp)
            # q_nope[b] shape [H, CK], q_pe[b] shape [H, KP]
            A_qn = q_nope[b].contiguous().view(H, CK)           # [H, CK]
            A_qp = q_pe[b].contiguous().view(H, KP)            # [H, KP]
            B_qc = ckv_cache[tok_idx].contiguous().view(L_tokens, CK)  # [L_tokens, CK]
            B_qp = kpe_cache[tok_idx].contiguous().view(L_tokens, KP)  # [L_tokens, KP]

            # Allocate logits buffer [H, L_tokens]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch kernel 1: compute scaled logits
            grid1 = (H, L_tokens)
            _compute_scaled_logits_kernel[grid1](
                A_qn, A_qp, B_qc, B_qp, logits,
                H=H, CK=CK, KP_KP=KP, L=L_tokens, sm_scale=float(sm_scale)
            )

            # Launch kernel 2: compute lse per (b,h)
            lse[b] = torch.empty((H,), dtype=torch.float32, device=device)
            _lse_kernel[(H,)](
                logits, lse[b],
                H=H, L=L_tokens,
                stride_log_h=H, stride_log_l=1  # simple row-major addressing: row stride = H, col stride = 1
            )

            # Launch kernel 3: compute output per (b,h)
            out_buf = torch.empty((H, CK), dtype=torch.float32, device=device)
            _compute_output_kernel[(H,)](
                logits, lse[b], B_qc, out_buf,
                H=H, CK=CK, L=L_tokens,
                stride_log_h=H, stride_log_l=1,
                stride_bc_t=CK, stride_bc_k=1,
                stride_out_h=CK, stride_out_k=1
            )

            output[b] = out_buf

        # Return output as bfloat16 and lse as float32, matching original signature (output, lse)
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
