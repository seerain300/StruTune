import math
import torch
import triton
import triton.language as tl


@triton.jit
def count_valid_indices(indices_ptr, out_ptr, N: tl.int32):
    # Each program instance processes one index
    pid = tl.program_id(0)
    if pid >= N:
        return
    val = tl.load(indices_ptr + pid)
    is_valid = val != -1  # -1 indicates padding
    # Write 1 if valid else 0
    one = tl.full((), 1, tl.int32)
    zero = tl.full((), 0, tl.int32)
    count = tl.where(is_valid, one, zero)
    tl.store(out_ptr + pid, count)


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
    sm_scale: tl.constexpr,    # e.g., 1.0
    L: tl.constexpr,           # number of keys (valid_length)
    BLOCK_K: tl.constexpr,     # chunk size along K dimension
):
    # Accumulator for logits across keys (before scaling), length L
    logits = tl.zeros([L], dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K and accumulate dot products
    for k0 in range(0, L, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_ids < L

        # Load qn row vector (512)
        q_row = tl.load(qn_ptr + tl.arange(0, 512))
        # Load Kc chunk: shape [BLOCK_K, 512] from flat Kc_flat_ptr
        Kc_chunk = tl.load(Kc_flat_ptr + k_ids[:, None] * 512 + tl.arange(0, 512)[None, :], mask=mask_k[:, None])
        # Compute dot(q_row, Kc_chunk) -> [BLOCK_K]
        dot_qn = tl.sum(q_row[None, :] * Kc_chunk, axis=1)
        logits += dot_qn

        # Also compute contribution from q_pe: dot(qp, Kp_chunk)
        qp_row = tl.load(qp_ptr + tl.arange(0, 64))
        Kp_chunk = tl.load(Kp_flat_ptr + k_ids[:, None] * 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None])
        dot_qp = tl.sum(qp_row[None, :] * Kp_chunk, axis=1)
        logits += dot_qp  # add to logits

    # Scale logits
    logits_scaled = logits * sm_scale

    # Compute LSE per head (base-2 log) using max trick
    max_val = tl.max(logits_scaled, axis=0)
    sum_exp = tl.sum(tl.exp(logits_scaled - max_val), axis=0)
    lse_val = tl.log(sum_exp) + max_val  # ln of sum
    lse_val = lse_val / math.log(2.0)    # ln -> log2
    tl.store(lse_ptr, lse_val)

    # Compute softmax of logits_scaled across L
    sum_exp_all = tl.sum(tl.exp(logits_scaled - max_val), axis=0)
    attn = tl.exp(logits_scaled - max_val) / sum_exp_all  # shape [L]

    # Compute output: out = sum_l attn[l] * Kc[l, :]
    out_vec = tl.zeros([512], dtype=tl.float32)
    for l_idx in range(0, L):
        Kc_row = tl.load(Kc_flat_ptr + l_idx * 512 + tl.arange(0, 512))
        out_vec += attn[l_idx] * Kc_row

    # Store output
    tl.store(out_ptr + tl.arange(0, 512), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0, block_k=128):
        super().__init__()
        self.sm_scale = sm_scale
        self.block_k = block_k
        # Fixed dimensions from original code
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.topk = 2048  # max possible, actual L can be less with padding

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

            # Compute valid_length via Triton kernel
            N = indices.numel()
            valid_counts = torch.empty(N, dtype=torch.int32, device=device)
            count_valid_indices[(N,)](indices, valid_counts, N)
            valid_length = int(valid_counts.sum().item())

            # Build tok_idx = indices[valid_mask] using PyTorch (metadata selection)
            tok_idx = indices[indices != -1].to(torch.int32)

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
                    sm_scale=self.sm_scale,
                    L=valid_length,
                    BLOCK_K=self.block_k,
                )

                # Store results
                output[t, h, :] = out_row
                lse[t, h] = lse_row[0]

        # Convert output to bfloat16 to match original function's output type
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
