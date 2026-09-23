import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_kernel(source_ptr, indices_ptr, out_ptr, length: tl.int32, row_stride: tl.int32):
    # Each program instance handles one output row index in [0, length)
    pid = tl.program_id(0)
    if pid >= length:
        return
    # Load the source row index
    idx = tl.load(indices_ptr + pid)
    # Compute source address and write to out_ptr[pid]
    val = tl.load(source_ptr + idx * row_stride)
    tl.store(out_ptr + pid, val)


@triton.jit
def per_token_attention_kernel(
    qn_ptr,           # [512], float32
    qp_ptr,           # [64],  float32
    Kc_flat_ptr,      # [L*512], float32
    Kp_flat_ptr,      # [L*64],  float32
    out_ptr,          # [512],    float32
    lse_ptr,          # [1],      float32
    valid_length: tl.int32,
    sm_scale: tl.float32,
    head_dim_kc: tl.int32,    # 512
    head_dim_kp: tl.int32,    # 64
):
    # Element-wise compute of attention per head
    # Initialize logits vector
    logits = tl.zeros([valid_length], dtype=tl.float32)

    # Accumulate logits = qn @ Kc.T + qp @ Kp.T
    # Iterate over keys element-wise
    for l in range(0, valid_length):
        # Compute qn dot with Kc[l, :]
        dot_qn = tl.zeros((), dtype=tl.float32)
        for j in range(0, head_dim_kc):
            qj = tl.load(qn_ptr + j)
            kc_j = tl.load(Kc_flat_ptr + l * head_dim_kc + j)
            dot_qn += qj * kc_j

        # Compute qp dot with Kp[l, :]
        dot_qp = tl.zeros((), dtype=tl.float32)
        for j in range(0, head_dim_kp):
            qj = tl.load(qp_ptr + j)
            kp_j = tl.load(Kp_flat_ptr + l * head_dim_kp + j)
            dot_qp += qj * kp_j

        logits[l] = dot_qn + dot_qp

    # Scale logits
    logits_scaled = logits * sm_scale

    # Compute LSE per head (base-2 log) using max trick
    max_val = -float('inf')
    for l in range(0, valid_length):
        if logits_scaled[l] > max_val:
            max_val = logits_scaled[l]

    sum_exp = tl.zeros((), dtype=tl.float32)
    for l in range(0, valid_length):
        sum_exp += tl.exp(logits_scaled[l] - max_val)

    lse_val = tl.log(sum_exp) + max_val  # ln of sum
    lse_val = lse_val / math.log(2.0)    # ln -> log2
    tl.store(lse_ptr, lse_val)

    # Compute softmax of logits_scaled across valid_length
    sum_exp_all = tl.zeros((), dtype=tl.float32)
    for l in range(0, valid_length):
        sum_exp_all += tl.exp(logits_scaled[l] - max_val)
    attn = tl.zeros([valid_length], dtype=tl.float32)
    for l in range(0, valid_length):
        attn[l] = tl.exp(logits_scaled[l] - max_val) / sum_exp_all

    # Compute output: out = sum_l attn[l] * Kc[l, :]
    out_vec = tl.zeros([head_dim_kc], dtype=tl.float32)
    for l in range(0, valid_length):
        for j in range(0, head_dim_kc):
            kc_j = tl.load(Kc_flat_ptr + l * head_dim_kc + j)
            out_vec[j] += attn[l] * kc_j

    # Store output
    for j in range(0, head_dim_kc):
        tl.store(out_ptr + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0):
        super().__init__()
        self.sm_scale = sm_scale
        # Fixed dimensions from original code
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.topk = 2048

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Ensure all tensors are on CUDA device
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda, "All tensors must be on CUDA device"

        # Flatten and cast paged KV caches to float32
        Kc_all = ckv_cache.reshape(-1, self.head_dim_ckv).to(torch.float32)  # [num_pages*64, 512]
        Kp_all = kpe_cache.reshape(-1, self.head_dim_kpe).to(torch.float32)  # [num_pages*64, 64]

        num_tokens, _, _ = q_nope.shape

        # Prepare outputs (float32 for compute)
        output = torch.empty((num_tokens, self.num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, self.num_qo_heads), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            # Get indices for this token: shape [topk]
            indices = sparse_indices[t]  # int32 tensor
            # Build tok_idx: filter out -1 if present. The provided inputs don't use -1; keep valid indices.
            # If your data may have -1, uncomment the line below to filter.
            tok_idx = indices[indices != -1].to(torch.int32)
            valid_length = tok_idx.numel()

            # Allocate Kc_t and Kp_t for this token: [valid_length, 512] and [valid_length, 64]
            Kc_t = torch.empty((valid_length, self.head_dim_ckv), dtype=torch.float32, device=device)
            Kp_t = torch.empty((valid_length, self.head_dim_kpe), dtype=torch.float32, device=device)

            # Launch Triton gather kernel to fill Kc_t (flat) and Kp_t (flat)
            Kc_t_flat = Kc_t.view(-1)  # [valid_length * 512]
            Kp_t_flat = Kp_t.view(-1)  # [valid_length * 64]

            # Cast Kc_all/Kp_all to flat contiguous buffers for Triton
            Kc_all_flat = Kc_all.reshape(-1)  # [(num_pages*64) * 512]
            Kp_all_flat = Kp_all.reshape(-1)  # [(num_pages*64) * 64]

            # For Kc gather: grid=(valid_length,)
            gather_rows_kernel[(valid_length,)](Kc_all_flat, tok_idx, Kc_t_flat, valid_length, 512)
            # For Kp gather: grid=(valid_length,)
            gather_rows_kernel[(valid_length,)](Kp_all_flat, tok_idx, Kp_t_flat, valid_length, 64)

            # Reshape gathered tensors back
            Kc_t = Kc_t_flat.reshape(valid_length, self.head_dim_ckv)  # [valid_length, 512]
            Kp_t = Kp_t_flat.reshape(valid_length, self.head_dim_kpe)  # [valid_length, 64]

            # Prepare qn and qp for this token: per head
            qn_t = q_nope[t].to(torch.float32).contiguous()  # [16, 512]
            qp_t = q_pe[t].to(torch.float32).contiguous()    # [16, 64]

            # Compute per-head attention using Triton kernel
            for h in range(self.num_qo_heads):
                qn_row = qn_t[h, :].contiguous()  # [512]
                qp_row = qp_t[h, :].contiguous()  # [64]

                out_row = torch.empty(self.head_dim_ckv, dtype=torch.float32, device=device)  # [512]
                lse_row = torch.empty(1, dtype=torch.float32, device=device)                 # [1]

                # Launch Triton kernel for this head on Kc_t and Kp_t
                per_token_attention_kernel[(1,)](
                    qn_row, qp_row,
                    Kc_t.reshape(-1), Kp_t.reshape(-1),
                    out_row, lse_row,
                    valid_length, float(self.sm_scale),  # ensure float32
                    self.head_dim_ckv, self.head_dim_kpe,
                )

                # Store results
                output[t, h, :] = out_row
                lse[t, h] = lse_row[0]

        # Convert output to bfloat16 to match original function's output type
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
