import math
import torch
import triton
import triton.language as tl


# Triton kernel: row-wise matmul C = A_row @ B, where:
# A_row is a 1D vector of length M (head_dim_ckv = 512).
# B is a 2D matrix of shape (TOPK, M), contiguous in the last dim (i).
# Output C is a 1D vector of length TOPK (logits).
@triton.jit
def matmul_row(A_ptr, B_ptr, C_ptr,
               M: tl.constexpr, TOPK: tl.constexpr):
    # Load A_row (vector of length M)
    A = tl.load(A_ptr + tl.arange(0, M))  # [M]
    # Accumulate logits for each j in [0, TOPK)
    for j in range(0, TOPK):
        # Load B[j, :] vector (length M)
        B_j = tl.load(B_ptr + j * M + tl.arange(0, M))  # [M]
        # Dot product: sum_i A[i] * B_j[i]
        s = tl.sum(A * B_j, axis=0)
        # Store
        tl.store(C_ptr + j, s)


# Triton kernel: softmax over TOPK entries and compute base-2 logsumexp per token/head.
# Inputs:
#   - X_ptr: logits vector (length TOPK), float32
#   - LSE_ptr: output scalar per token/head, float32 (base-2 logsumexp)
#   - Out_ptr: output probabilities vector (length TOPK), float32
@triton.jit
def softmax_logsumexp2_row(X_ptr, LSE_ptr, Out_ptr,
                            TOPK: tl.constexpr):
    # Compute stable logsumexp (base e), then convert to base-2 at the end.
    m = -float("inf")
    for j in range(0, TOPK):
        v = tl.load(X_ptr + j)
        m = tl.maximum(m, v)
    # Compute sum of exp(v - m)
    s = 0.0
    for j in range(0, TOPK):
        v = tl.load(X_ptr + j)
        s += tl.exp(v - m)
    # lse (base 2) = m + log(s) / ln(2)
    lse_val = m + tl.log(s) / 1.4426950408889634  # 1 / ln(2)
    tl.store(LSE_ptr, lse_val)
    # Write probabilities for all j
    for j in range(0, TOPK):
        v = tl.load(X_ptr + j)
        prob = tl.exp(v - m) / s
        tl.store(Out_ptr + j, prob)


# Triton kernel: reduction out = Attn @ Kc, where:
# Attn is a 1D vector of length NUM_VALID (compile-time; pass NUM_VALID=TOPK).
# Kc is a flattened vector of length NUM_VALID * OUT (e.g., 2048 * 512).
# OUT is tl.constexpr (e.g., 512). Produces a 1D Out vector of length OUT.
@triton.jit
def reduction_row(Attn_ptr, Kc_ptr, Out_ptr,
                  NUM_VALID: tl.constexpr, OUT: tl.constexpr):
    acc = tl.zeros([OUT], dtype=tl.float32)
    for o in range(0, OUT):
        sum_val = 0.0
        for j in range(0, NUM_VALID):
            sum_val += tl.load(Attn_ptr + j) * tl.load(Kc_ptr + j * OUT + o)
        acc[o] = sum_val
    tl.store(Out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed dimensions per the original code
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.topk = 2048  # candidates per token
        self.out_dim = self.head_dim_ckv  # 512

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA for Triton kernels"
        device = q_nope.device

        # Extract shapes
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head_dim_ckv
        _, _, head_dim_kpe = q_pe.shape
        assert head_dim_kpe == 64  # consistent with original

        # Flatten KV caches to [num_pages * 64, dim], contiguous
        num_pages, page_size, _ = ckv_cache.shape
        assert page_size == 64
        Kc_all = ckv_cache.reshape(-1, self.head_dim_ckv).to(torch.float32).contiguous()  # [num_pages * 64, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32).contiguous()      # [num_pages * 64, 64]

        # Prepare outputs
        output = torch.empty(
            (num_tokens, num_qo_heads, self.head_dim_ckv),
            dtype=torch.bfloat16,
            device=device
        )
        lse = torch.empty(
            (num_tokens, num_qo_heads),
            dtype=torch.float32,
            device=device
        )

        # Process each token; Triton kernels parallelize across heads inside the loop.
        for t in range(num_tokens):
            # Load sparse indices for this token: shape [topk]
            indices = sparse_indices[t]  # int32 of shape [topk]

            # Prepare A_row pointers for q_nope and q_pe
            q_row = q_nope[t, :].to(torch.float32).contiguous()  # [16, 512]
            qpe_row = q_pe[t, :].to(torch.float32).contiguous()  # [16, 64]

            # Allocate temporaries for logits.
            logits_qn = torch.empty(self.topk, dtype=torch.float32, device=device)  # [TOPK]
            logits_qp = torch.empty(self.topk, dtype=torch.float32, device=device)  # [TOPK]

            # Compute logits_qn = q_nope[t, :] @ Kc_all[indices, :]
            # For Triton matmul_row, we pass A_ptr as a flattened q_row vector and B_ptr as a contiguous
            # 2D block of selected rows from Kc_all. Here, B is 2048 x 512.
            # Construct B_mat by selecting rows: we'll build a contiguous [TOPK, M] tensor in PyTorch,
            # then pass its pointer to Triton. This is acceptable and ensures Triton performs the heavy dot.
            # Build B_mat for Kc
            B_mat_qn = torch.empty((self.topk, self.head_dim_ckv), dtype=torch.float32, device=device)
            for j in range(0, self.topk):
                idx = int(indices[j].item())
                B_mat_qn[j] = Kc_all[idx]  # [512]
            # Launch Triton kernel to compute logits_qn
            matmul_row[(1,)](
                q_row.view(-1),  # A_ptr length M
                B_mat_qn.view(-1),  # B_ptr length TOPK*M
                logits_qn,
                M=self.head_dim_ckv,
                TOPK=self.topk
            )

            # Compute logits_qp = q_pe[t, :] @ Kp_all[indices, :]
            B_mat_qp = torch.empty((self.topk, head_dim_kpe), dtype=torch.float32, device=device)
            for j in range(0, self.topk):
                idx = int(indices[j].item())
                B_mat_qp[j] = Kp_all[idx]  # [64]
            # Launch Triton kernel to compute logits_qp
            matmul_row[(1,)](
                qpe_row.view(-1),  # A_ptr length M=64
                B_mat_qp.view(-1),  # B_ptr length TOPK*M=2048*64
                logits_qp,
                M=head_dim_kpe,
                TOPK=self.topk
            )

            # Precompute scaled logits: (logits_qn + logits_qp) * sm_scale
            scaled_logits = logits_qn + logits_qp  # [TOPK]
            scaled_logits = scaled_logits * sm_scale

            # Launch softmax_logsumexp2_row kernel to compute softmax and lse per head
            for h in range(0, num_qo_heads):
                prob_out = torch.empty(self.topk, dtype=torch.float32, device=device)
                lse_out = torch.empty((), dtype=torch.float32, device=device)
                softmax_logsumexp2_row[(1,)](
                    scaled_logits, lse_out, prob_out,
                    TOPK=self.topk
                )
                attn = prob_out  # [TOPK], float32

                # Reduction: out = attn @ Kc_all[indices, :]
                out_vec = torch.empty(self.out_dim, dtype=torch.float32, device=device)
                # Build Kc_ptr for this token: [NUM_VALID * OUT] contiguous vector
                # We don't have a prebuilt contiguous Kc for all j; instead, construct it on the fly.
                # For Triton reduction_row, we need a contiguous Kc_ptr of length TOPK * OUT (2048 * 512).
                # Build Kc_sel by gathering selected rows from Kc_all according to indices, then flatten.
                # Note: This gather is done in PyTorch to ensure correctness and simplicity. Triton can't
                # accept dynamic lists of pointers, so we reconstruct the contiguous Kc segment here.
                Kc_sel = torch.stack([Kc_all[int(indices[j].item())] for j in range(self.topk)], dim=0)  # [TOPK, 512]
                Kc_ptr = Kc_sel.contiguous().view(-1)  # length TOPK * 512
                # Launch Triton kernel to compute out_vec
                reduction_row[(1,)](
                    attn.contiguous(), Kc_ptr, out_vec,
                    NUM_VALID=self.topk, OUT=self.out_dim
                )
                # Store to output in bfloat16
                output[t, h, :] = out_vec.to(torch.bfloat16)
                # Store lse for this head
                lse[t, h] = lse_out.item()

        return output, lse


def run(*args):
    return ModelNew()(*args)
