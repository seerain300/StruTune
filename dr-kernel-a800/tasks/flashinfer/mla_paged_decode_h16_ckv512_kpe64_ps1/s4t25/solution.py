import torch
import math
import triton
import triton.language as tl


# Triton kernel: compute logits vector C[i] = dot(q_row, B[i, :]) for i in 0..M_CONST-1
# A_ptr points to a row vector of length K. B_ptr points to [M_CONST, K]. C_ptr points to [M_CONST].
@triton.jit
def matvec_row_kernel(A_ptr, B_ptr, C_ptr,
                      K: tl.constexpr,            # K dimension (e.g., 512 or 64)
                      M_CONST: tl.constexpr,      # compile-time maximum number of rows (grid size)
                      BLOCK_K: tl.constexpr):
    i = tl.program_id(0)  # output index
    acc = 0.0
    # Loop over K in tiles
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # vectorized offsets along K
        mask_k = offs_k < K
        # Load a chunk of A (row vector)
        a = tl.load(A_ptr + offs_k, mask=mask_k, other=0.0)
        # Load corresponding chunk from B[i, :]
        # Note: B is laid out as [M_CONST, K], row i has offset i*K + offs_k
        b = tl.load(B_ptr + i * K + offs_k, mask=mask_k, other=0.0)
        # Accumulate dot product for this tile
        acc += tl.sum(a * b, axis=0)
    # Store result to C[i]; mask store if i >= M
    tl.store(C_ptr + i, acc)


# Triton kernel: compute per-head lse and attention vector from x (logits_scaled) of length M_CONST
@triton.jit
def softmax_lse_kernel(x_ptr, out_lse_ptr, attn_ptr,
                       M_CONST: tl.constexpr, sm_scale: tl.constexpr):
    idx = tl.arange(0, M_CONST)
    x = tl.load(x_ptr + idx)  # vector of size M_CONST (runtime x_ptr has zeros for i >= M)
    # Compute max for numerical stability
    max_val = tl.max(x)
    x = x - max_val
    # Compute sum of exp
    sum_exp = tl.sum(tl.exp(x))
    # lse = log(sum_exp) * (1 / ln(2))
    inv_ln2 = 1.0 / math.log(2.0)
    lse = tl.log(sum_exp) * inv_ln2
    # Store lse
    tl.store(out_lse_ptr, lse)
    # attention vector
    attn = tl.exp(x) / sum_exp
    # Store attn vector
    tl.store(attn_ptr + idx, attn)


# Triton kernel: compute out[d] = sum_m attn[m] * Kc[m, d] for d in 0..Kc_dim-1
@triton.jit
def matvec_attn_kernel(attn_ptr, Kc_ptr, out_ptr,
                       Kc_dim: tl.constexpr,   # output feature dimension (e.g., 512)
                       M_CONST: tl.constexpr,   # number of tokens (constexpr for grid=M_CONST)
                       BLOCK_M: tl.constexpr):
    d = tl.program_id(0)  # output feature index
    acc = 0.0
    for m0 in tl.static_range(0, M_CONST, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M_CONST
        attn_chunk = tl.load(attn_ptr + offs_m, mask=mask_m, other=0.0)
        Kc_chunk = tl.load(Kc_ptr + offs_m * Kc_dim + d, mask=mask_m, other=0.0)
        acc += tl.sum(attn_chunk * Kc_chunk, axis=0)
    tl.store(out_ptr + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        # Triton tuning parameters
        self.BLOCK_K_QN = 128  # for K=512
        self.BLOCK_K_QP = 64   # for K=64
        self.BLOCK_M = 128     # for attn over M
        # Choose a compile-time M_CONST (power of two up to 1024)
        self.M_CONST = 1024

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Ensure inputs are on the same device and contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [N, 64]
        batch_size = q_nope.shape[0]
        H = self.num_qo_heads
        Kc_dim = self.head_dim_ckv

        output = torch.empty((batch_size, H, Kc_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Determine valid token range
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            M = max(0, page_end - page_beg)

            if M == 0:
                lse[b].zero_()
                continue

            # Gather tokens
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()  # [M]
            # Gather Kc and Kp rows
            Kc = Kc_all[tok_idx]  # [M, 512], float32
            Kp = Kp_all[tok_idx]  # [M, 64], float32

            # Create row-major Kc_row and Kp_row of shape [M_CONST, K] with zeros beyond M
            Kc_row = torch.zeros((self.M_CONST, Kc_dim), dtype=torch.float32, device=device)
            Kp_row = torch.zeros((self.M_CONST, self.head_dim_kpe), dtype=torch.float32, device=device)
            Kc_row[:M, :] = Kc
            Kp_row[:M, :] = Kp

            # Prepare qn and qp as float32 row vectors (1xK)
            for h in range(H):
                qn_row = q_nope[b, h].to(torch.float32).contiguous()  # [512]
                qp_row = q_pe[b, h].to(torch.float32).contiguous()    # [64]

                # Compute logits = qn_row @ Kc.T  -> [M_CONST], but only first M elements are valid
                logits_qn = torch.empty((self.M_CONST,), dtype=torch.float32, device=device)
                matvec_row_kernel[(self.M_CONST,)](
                    qn_row, Kc_row, logits_qn,
                    K=self.head_dim_ckv,
                    M_CONST=self.M_CONST,
                    BLOCK_K=self.BLOCK_K_QN,
                )

                # Compute logits_qp = qp_row @ Kp.T  -> [M_CONST]
                logits_qp = torch.empty((self.M_CONST,), dtype=torch.float32, device=device)
                matvec_row_kernel[(self.M_CONST,)](
                    qp_row, Kp_row, logits_qp,
                    K=self.head_dim_kpe,
                    M_CONST=self.M_CONST,
                    BLOCK_K=self.BLOCK_K_QP,
                )

                # Sum and scale
                logits = logits_qn + logits_qp  # [M_CONST]; entries beyond M are sum of zeros -> fine
                logits_scaled = logits * sm_scale  # [M_CONST]

                # Compute lse and attention vector in Triton
                attn = torch.empty((self.M_CONST,), dtype=torch.float32, device=device)
                lse_sub = torch.empty((1,), dtype=torch.float32, device=device)
                softmax_lse_kernel[(1,)](
                    logits_scaled, lse_sub, attn,
                    M_CONST=self.M_CONST,  # constexpr
                    sm_scale=sm_scale,     # constexpr
                )
                # Store lse[b, h]
                lse[b, h] = lse_sub[0]

                # Compute out[h, :] = attn @ Kc  -> [512]
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                matvec_attn_kernel[(Kc_dim,)](
                    attn, Kc_row, out_vec,
                    Kc_dim=Kc_dim,
                    M_CONST=self.M_CONST,  # constexpr
                    BLOCK_M=self.BLOCK_M,
                )
                output[b, h] = out_vec

        # Cast outputs to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


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
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
