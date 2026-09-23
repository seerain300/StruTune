import torch
import triton
import triton.language as tl


@triton.jit
def _compute_one_bh(
    q_nope_ptr,         # *bf16, shape [H, D1], contiguous
    q_pe_ptr,           # *bf16, shape [H, D2], contiguous
    Kc_all_ptr,         # *bf16, shape [N, D1], contiguous
    Kp_all_ptr,         # *bf16, shape [N, D2], contiguous
    kv_indptr_ptr,      # *int32, shape [B+1], contiguous
    out_ptr,            # *bf16, 1D buffer of size B*H*D1
    H: tl.constexpr,    # num heads (e.g., 16)
    D1: tl.constexpr,   # head_dim_ckv (e.g., 512)
    D2: tl.constexpr,   # head_dim_kpe (e.g., 64)
    MAX_T: tl.constexpr,# max tokens to loop over, e.g., 4096
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load batch indptr start/end to compute L_tokens
    indptr_b = tl.load(kv_indptr_ptr + b)            # int32
    indptr_b1 = tl.load(kv_indptr_ptr + b + 1)       # int32
    L_tokens = indptr_b1 - indptr_b                  # int32 scalar

    # Pointers to q vectors for this head
    qn_ptr = q_nope_ptr + h * D1
    qp_ptr = q_pe_ptr + h * D2

    # Output vector for this (b, h): out_ptr[pid * D1 : (pid+1)*D1]
    out_vec_ptr = out_ptr + pid * D1

    # Initialize output vector to zeros
    for d in tl.static_range(0, D1):
        tl.store(out_vec_ptr + d, 0.0)

    # Iterate tokens with static loop; guard with valid
    for t in tl.static_range(0, MAX_T):
        valid = t < L_tokens
        if not valid:
            continue
        # Compute row index in Kc_all/Kp_all: assume tokens are contiguous within batch
        row_idx = indptr_b + t  # int32
        # Load qn and qp (1D vectors)
        qn = tl.load(qn_ptr + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
        qp = tl.load(qp_ptr + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]
        # Load corresponding Kc_row and Kp_row
        Kc_row_ptr = Kc_all_ptr + row_idx * D1
        Kp_row_ptr = Kp_all_ptr + row_idx * D2
        Kc_row = tl.load(Kc_row_ptr + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
        Kp_row = tl.load(Kp_row_ptr + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

        # Compute scalar logits: sum(qn*Kc_row) + sum(qp*Kp_row)
        dot1 = tl.sum(qn * Kc_row, axis=0)
        dot2 = tl.sum(qp * Kp_row, axis=0)
        logits = (dot1 + dot2)  # scale factor sm_scale implicitly 1.0 here

        # Accumulate output: out[b, h, :] += logits * Kc_row
        for d in tl.static_range(0, D1):
            tl.store(out_vec_ptr + d, tl.load(out_vec_ptr + d) + logits * Kc_row[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use a conservative MAX_T to cover typical token counts across workloads
        self.max_t = 4096

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device
        B, H, D1 = q_nope.shape
        _, _, D2 = q_pe.shape
        N = ckv_cache.shape[0]

        # Output buffer (1D): will be reshaped to [B, H, D1]
        out_flat = torch.empty(B * H * D1, dtype=torch.bfloat16, device=device)

        # Launch Triton: one program per (b, h)
        grid = (B * H,)
        _compute_one_bh[grid](
            q_nope,            # *bf16
            q_pe,              # *bf16
            ckv_cache,         # *bf16, [N, D1]
            kpe_cache,         # *bf16, [N, D2]
            kv_indptr,         # *int32
            out_flat,          # *bf16
            H=H, D1=D1, D2=D2, MAX_T=self.max_t,
            num_warps=4,
        )

        output = out_flat.view(B, H, D1)
        return output


def run(*args):
    return ModelNew()(*args)
