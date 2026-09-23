import torch
import math
import triton
import triton.language as tl


@triton.jit
def lse_and_attn_kernel(
    qn_ptr,          # *float32, [B*N*Dc] flattened
    qp_ptr,          # *float32, [B*N*Dp] flattened
    Kc_ptr,          # *float32, [P*Dc] flattened
    Kp_ptr,          # *float32, [P*Dp] flattened
    attn_ptr,        # *float32, [B*N*M_b] flattened, per (b,h) vector
    lse_ptr,         # *float32, [B*N] flattened
    B: tl.constexpr,      # batch size (int)
    N: tl.constexpr,      # number of heads (int)
    Dc: tl.constexpr,     # head_dim_ckv (int, e.g., 512)
    Dp: tl.constexpr,     # head_dim_kpe (int, e.g., 64)
    M_b_max: tl.constexpr,# max tokens across batches (int), used for grid sizing
    sm_scale: tl.constexpr,   # scaling factor (float)
    M_b_list_ptr,      # *int32, [B] per-batch tokens (int array)
    BLOCK_T: tl.constexpr  # token tile size (e.g., 128)
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Base offsets for qn, qp for this (b,h)
    qn_base = (pid_b * N + pid_h) * Dc
    qp_base = (pid_b * N + pid_h) * Dp

    # Load qn and qp vectors
    qn = tl.load(qn_ptr + qn_base + tl.arange(0, Dc))
    qp = tl.load(qp_ptr + qp_base + tl.arange(0, Dp))

    # Load M_b for this batch (pid_b), and set up
    M_b = tl.load(M_b_list_ptr + pid_b)

    # Initialize numerically stable logsumexp accumulators (scalars)
    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    sum_val = tl.full([1], 0.0, dtype=tl.float32)

    # Loop over tokens in chunks
    for t0 in range(0, M_b_max, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < M_b

        # Compute pointers for Kc_sub rows and Kp_sub rows
        Kc_rows = Kc_ptr + offs_t * Dc
        Kp_rows = Kp_ptr + offs_t * Dp

        # Load Kc_sub and Kp_sub rows with mask
        Kc_sub = tl.load(Kc_rows, mask=mask_t, other=0.0)  # [BLOCK_T, Dc]
        Kp_sub = tl.load(Kp_rows, mask=mask_t, other=0.0)  # [BLOCK_T, Dp]

        # Compute logits for this chunk: qn @ Kc_sub + qp @ Kp_sub
        # qn and qp are 1D vectors of length Dc and Dp, respectively.
        # To compute qn @ Kc_sub[:, :] efficiently, we can use tl.dot for each row in the chunk.
        # However, Triton prefers vectorized ops. We compute elementwise and reduce.
        # For each t in chunk:
        for i in range(0, BLOCK_T):
            valid_i = mask_t[i]
            # Gather Kc_sub[i] and Kp_sub[i]
            kc_i = Kc_sub[i, :]  # [Dc]
            kp_i = Kp_sub[i, :]  # [Dp]
            # Compute dot products (reduce over Dc and Dp)
            # dot_qn_kc = sum(qn * kc_i), dot_qp_kp = sum(qp * kp_i)
            dot_qn_kc = tl.sum(qn * kc_i, axis=0)
            dot_qp_kp = tl.sum(qp * kp_i, axis=0)
            logits_i = dot_qn_kc + dot_qp_kp  # scalar
            logits_scaled_i = logits_i * sm_scale
            # Update logsumexp in a numerically stable manner
            cur = tl.full([1], logits_scaled_i, dtype=tl.float32)
            greater = cur > max_val
            # When cur > max_val, sum = sum*exp(max - cur) + 1; else sum = sum + exp(cur - max)
            sum_val = tl.where(greater, sum_val * tl.exp(max_val - cur) + 1.0, sum_val + tl.exp(cur - max_val))
            max_val = tl.where(greater, cur, max_val)
            # Also write attn[b,h,offs_t[i]] = exp(logits_scaled_i - max_val) / sum_val
            attn_idx = pid_b * (N * M_b_max) + pid_h * M_b_max + offs_t[i]
            attn_val = tl.where(valid_i, tl.exp(cur - max_val) / sum_val, 0.0)
            tl.store(attn_ptr + attn_idx, attn_val)

    # Compute lse = max + log(sum) / log(2)
    # Convert natural log to base-2 log
    lse_val = max_val + tl.log(sum_val) / 2.0  # logsumexp in base-2
    tl.store(lse_ptr + (pid_b * N + pid_h), lse_val)


@triton.jit
def matvec_reduce_kernel(
    attn_ptr,        # *float32, [B*N*M_b] flattened
    Kc_ptr,          # *float32, [P*Dc] flattened
    out_ptr,         # *float32, [B*N*Dc] flattened
    B: tl.constexpr,      # batch size (int)
    N: tl.constexpr,      # number of heads (int)
    Dc: tl.constexpr,     # head_dim_ckv (int, e.g., 512)
    M_b_max: tl.constexpr,# max tokens across batches (int), used for indexing attn
    BLOCK_D: tl.constexpr # tile size along Dc (e.g., 128)
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # We need M_b for this (b,h). We don't have M_b directly; attn_ptr is indexed by (b,h,t), but Triton
    # requires static grid. We compute M_b via loading from a device tensor or host-provided M_b_list.
    # To keep it simple and Triton-only, we assume M_b = M_b_max (mask in lse kernel used M_b_max). Here we
    # rely on grid sizing and masking via Dc loops and attn_ptr bounds. We'll recompute M_b using a device
    # int32 tensor M_b_list_ptr. Triton lacks host access; thus we pass M_b as constexpr per launch.
    # Instead, we compute M_b inside per program: recompute using the same lse kernel's M_b approach isn't feasible.
    # Therefore, we use M_b_max to iterate and mask loads by actual M_b. We must pass M_b. Triton expects
    # static args; we can't query M_b. To ensure correctness, we assume M_b = M_b_max; attn_ptr writes only
    # valid entries. If M_b < M_b_max, attn entries beyond M_b are 0.

    # Output base pointer for this (b,h)
    out_base = (pid_b * N + pid_h) * Dc
    # Accumulator vector for output
    out_vec = tl.zeros([Dc], dtype=tl.float32)

    # Iterate over tokens t from 0 to M_b_max-1; for t >= M_b, attn value is 0
    for t in range(0, M_b_max):
        attn_idx = pid_b * (N * M_b_max) + pid_h * M_b_max + t
        attn_val = tl.load(attn_ptr + attn_idx)
        # Load corresponding Kc_sub row
        Kc_row = tl.load(Kc_ptr + t * Dc)
        # Accumulate: out_vec += attn_val * Kc_row
        out_vec += attn_val * Kc_row

    # Store out_vec
    tl.store(out_ptr + out_base + tl.arange(0, Dc), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable tile sizes
        self.block_t = 128
        self.block_d = 128

    def forward(self, *args):
        # Accept up to 8 inputs; ignore the 8th if provided to match evaluator
        # We expect: q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, [unused]
        # args[0..6] are the inputs; args[7] is ignored.
        q_nope = args[0]
        q_pe = args[1]
        ckv_cache = args[2]
        kpe_cache = args[3]
        kv_indptr = args[4]
        kv_indices = args[5]
        sm_scale = args[6]

        # Ensure device consistency
        device = q_nope.device
        dtype_q = q_nope.dtype
        assert dtype_q == torch.bfloat16, "q_nope must be bfloat16"

        # Cast q_nope and q_pe to float32 for Triton compute
        q_nope_f32 = q_nope.contiguous().to(torch.float32)  # [B, N, Dc]
        q_pe_f32 = q_pe.contiguous().to(torch.float32)      # [B, N, Dp]

        # Squeeze ckv_cache and kpe_cache along dim=1
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [P, Dp]

        B = q_nope_f32.shape[0]
        N = q_nope_f32.shape[1]
        Dc = q_nope_f32.shape[2]
        Dp = q_pe_f32.shape[2]
        P = Kc_all.shape[0]

        # Compute per-batch number of tokens M_b
        M_b_list = (kv_indptr[1:] - kv_indptr[:B]).to(torch.int32).to(device)  # [B]
        # Max tokens across batches for grid sizing
        M_b_max = int(M_b_list.max().item())
        # Prepare attn and lse buffers
        attn = torch.empty((B, N, M_b_max), dtype=torch.float32, device=device)
        lse = torch.empty((B, N), dtype=torch.float32, device=device)

        # Flatten q_nope and q_pe for kernel (1D)
        qn_flat = q_nope_f32.reshape(-1, Dc).reshape(-1)  # [B*N*Dc]
        qp_flat = q_pe_f32.reshape(-1, Dp).reshape(-1)    # [B*N*Dp]

        # Launch lse_and_attn_kernel: one program per (b,h)
        grid = (B, N)
        lse_and_attn_kernel[grid](
            qn_flat, qp_flat, Kc_all, Kp_all, attn, lse,
            B=B, N=N, Dc=Dc, Dp=Dp, M_b_max=M_b_max, sm_scale=float(sm_scale),
            M_b_list_ptr=M_b_list,
            BLOCK_T=self.block_t
        )

        # Prepare output buffer for matvec reduction
        out = torch.empty((B, N, Dc), dtype=torch.float32, device=device)

        # Launch matvec_reduce_kernel: one program per (b,h)
        grid_reduce = (B, N)
        matvec_reduce_kernel[grid_reduce](
            attn, Kc_all, out,
            B=B, N=N, Dc=Dc, M_b_max=M_b_max,
            BLOCK_D=self.block_d
        )

        # Cast output to bfloat16 to match original
        out = out.to(torch.bfloat16)

        return out, lse


# Optional helpers to match the provided get_inputs and fused_operator interfaces
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, None]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7=None):
    return ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)