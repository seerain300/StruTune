import math
import torch
import triton
import triton.language as tl


# Kernel 1: Generate hidden_states (num_tokens x hidden_size) in bfloat16
@triton.jit
def _init_hidden_states_kernel(
    out_ptr,               # *bf16
    num_tokens: tl.int32,
    hidden_size: tl.int32,
    seed: tl.int32,
):
    pid = tl.program_id(axis=0)
    # Compute row and column indices
    row = pid // hidden_size
    col = pid % hidden_size
    if row >= num_tokens:
        return
    # Initialize random value
    val = tl.rand(seed, pid)
    # Store as bf16
    # Triton does not have a direct tl.float16 type to store; emulate via float32 pointer and cast later in host
    tl.store(out_ptr + row * hidden_size + col, val.to(tl.float32))


# Kernel 2: Generate selected_experts (num_tokens x num_experts_per_tok) int64
@triton.jit
def _init_selected_experts_kernel(
    out_ptr,               # *int64
    num_tokens: tl.int32,
    num_experts_per_tok: tl.int32,
    num_experts: tl.int32,
    seed: tl.int32,
):
    t = tl.program_id(axis=0)
    k = tl.program_id(axis=1)
    if t >= num_tokens:
        return
    # Pick a random expert index without replacement
    idx = tl.rand(seed, t * num_experts_per_tok + k) * num_experts
    # Cast to int64
    idx = idx.to(tl.int64)
    tl.store(out_ptr + t * num_experts_per_tok + k, idx)


# Kernel 3: Generate routing_logits (num_tokens x num_experts_per_tok) bfloat16, then softmax in Triton
@triton.jit
def _init_routing_logits_softmax_kernel(
    logits_ptr,            # *bf16
    weights_ptr,           # *bf16
    num_tokens: tl.int32,
    K: tl.int32,
    seed: tl.int32,
):
    t = tl.program_id(axis=0)
    k = tl.program_id(axis=1)
    if t >= num_tokens:
        return
    val = tl.rand(seed, t * K + k)
    # Store as bf16
    tl.store(logits_ptr + t * K + k, val.to(tl.float32))  # host will cast to bf16
    # Softmax: compute exp, sum, divide
    # First pass: sum
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(K):
        v = tl.load(logits_ptr + t * K + j, eviction_policy="evict_first")
        exp_v = tl.exp(v)
        sum_exp += exp_v
    # Second pass: normalize
    for j in range(K):
        v = tl.load(logits_ptr + t * K + j)
        w = tl.exp(v) / sum_exp
        tl.store(weights_ptr + t * K + j, w.to(tl.float32))  # host will cast to bf16


# Kernel 4: Per-row matmul: C[hidden_size] = A_row[hidden_size] @ B[hidden_size, hidden_size]
@triton.jit
def _per_row_matmul_kernel(
    C_ptr,                 # *bf16
    A_ptr,                 # *bf16 (row vector)
    B_ptr,                 # *bf16 (matrix)
    H: tl.constexpr,       # hidden_size (runtime int32)
    stride_b_row: tl.int32,
    stride_b_col: tl.int32,
):
    # One program computes one output element c[col]
    col = tl.program_id(axis=0)
    if col >= H:
        return
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K dimension (here K=H)
    for k in range(H):
        a_k = tl.load(A_ptr + k)  # row vector element
        b_k = tl.load(B_ptr + k * stride_b_row + col * stride_b_col)
        acc += a_k * b_k
    tl.store(C_ptr + col, acc.to(tl.float32))


# Kernel 5: Elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def _silu_kernel(
    out_ptr,               # *bf16
    in_ptr,                # *bf16
    N: tl.int32,
):
    i = tl.program_id(axis=0)
    if i >= N:
        return
    x = tl.load(in_ptr + i)
    # sigmoid(x) = 1 / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + i, y.to(tl.float32))


# Kernel 6: Elementwise multiply: out[i] = a[i] * b[i]
@triton.jit
def _mul_kernel(
    out_ptr,               # *bf16
    a_ptr,                 # *bf16
    b_ptr,                 # *bf128
    N: tl.int32,
):
    i = tl.program_id(axis=0)
    if i >= N:
        return
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, (a * b).to(tl.float32))


# Kernel 7: Atomic add weighted vector into output row: out[t * H + col] += weight * val
@triton.jit
def _atomic_add_weighted_kernel(
    out_ptr,               # *bf16
    val_ptr,               # *bf16
    weights_ptr,           # *bf16
    num_tokens: tl.int32,
    H: tl.int32,
    t: tl.int32,           # which token
):
    col = tl.program_id(axis=0)
    if col >= H:
        return
    weight = tl.load(weights_ptr + t * H + col)
    val = tl.load(val_ptr + col)
    old = tl.atomic_add(out_ptr + t * H + col, val * weight)
    # old is not used, but keeps the atomic op valid


# Helper to launch per-row matmul: computes gate_out, up_out, and expert_outputs
# We will call _per_row_matmul_kernel with grid=(H,) and appropriate pointers
def _launch_per_row_matmul(C_ptr, A_ptr, B_ptr, H: int, stride_b_row: int, stride_b_col: int):
    grid = (H,)
    _per_row_matmul_kernel[grid](C_ptr, A_ptr, B_ptr, H=H, stride_b_row=stride_b_row, stride_b_col=stride_b_col)
    return C_ptr


# Host-side forward of ModelNew: use Triton for all numerical compute
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Seed for Triton RNG
        self.seed = torch.randint(0, 2**31 - 1, ()).item()

    def forward(self, axes_and_scalars: dict, device: torch.device):
        # Extract axes
        num_tokens = int(axes_and_scalars["num_tokens"])
        hidden_size = int(axes_and_scalars["hidden_size"])
        moe_intermediate_size = int(axes_and_scalars["moe_intermediate_size"])
        num_experts = int(axes_and_scalars["num_experts"])
        num_experts_per_tok = int(axes_and_scalars["num_experts_per_tok"])

        # 1) Generate hidden_states (bf16) using Triton kernel (host creates float32, kernel casts)
        hidden = torch.empty((num_tokens, hidden_size), dtype=torch.float32, device=device)
        grid_hs = (num_tokens * hidden_size,)
        _init_hidden_states_kernel[grid_hs](hidden, num_tokens, hidden_size, self.seed)

        # 2) Generate selected_experts (int64) using Triton kernel
        selected_experts = torch.empty((num_tokens, num_experts_per_tok), dtype=torch.int64, device=device)
        grid_se = (num_tokens, num_experts_per_tok)
        _init_selected_experts_kernel[grid_se](selected_experts, num_tokens, num_experts_per_tok, num_experts, self.seed)

        # 3) Generate routing_logits and routing_weights (bf16) via Triton softmax kernel
        routing_logits = torch.empty((num_tokens, num_experts_per_tok), dtype=torch.float32, device=device)
        routing_weights = torch.empty((num_tokens, num_experts_per_tok), dtype=torch.float32, device=device)
        grid_rw = (num_tokens, num_experts_per_tok)
        _init_routing_logits_softmax_kernel[grid_rw](routing_logits, routing_weights, num_tokens, num_experts_per_tok, self.seed)

        # 4) Generate expert weights (bf16) with random and scaling
        # gate/up weights: [num_experts, hidden_size, moe_intermediate_size]
        expert_gate = torch.empty((num_experts, hidden_size, moe_intermediate_size), dtype=torch.float32, device=device)
        expert_up = torch.empty((num_experts, hidden_size, moe_intermediate_size), dtype=torch.float32, device=device)
        # down weights: [num_experts, moe_intermediate_size, hidden_size]
        expert_down = torch.empty((num_experts, moe_intermediate_size, hidden_size), dtype=torch.float32, device=device)
        # Fill with random and scale
        for e in range(num_experts):
            for h in range(hidden_size):
                for j in range(moe_intermediate_size):
                    seed_ehj = (self.seed + e * hidden_size * moe_intermediate_size + h * moe_intermediate_size + j)
                    v = tl.rand(seed_ehj).to(torch.float32)
                    expert_gate[e, h, j] = v / math.sqrt(hidden_size)
                    expert_up[e, h, j] = v / math.sqrt(hidden_size)
            for j in range(moe_intermediate_size):
                for o in range(hidden_size):
                    seed_jo = (self.seed + e * (moe_intermediate_size * hidden_size) + j * hidden_size + o)
                    v = tl.rand(seed_jo).to(torch.float32)
                    expert_down[e, j, o] = v / math.sqrt(moe_intermediate_size)

        # 5) Process each token and each selected expert
        result = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=device)  # final output (bf16 expected)

        for t in range(num_tokens):
            # Iterate K selected experts
            for k in range(num_experts_per_tok):
                # Get expert index
                exp = int(selected_experts[t, k].item())
                # hidden input vector for token t
                h_in = hidden[t, :].to(torch.float32)  # already created by Triton kernel above

                # gate_out: hidden_size
                gate_out = torch.empty((hidden_size,), dtype=torch.float32, device=device)
                # B gate: [hidden_size, hidden_size]
                B_gate = expert_gate[exp]  # shape [hidden_size, moe_intermediate_size] ? Wait, original uses hidden_size x intermediate_size for gate weights, but in the original, gate weights are [num_experts, hidden_size, intermediate_size]. To match, we need B_gate of shape [hidden_size, intermediate_size], then down is [intermediate_size, hidden_size]. However, the original code uses expert_gate_weights as [num_experts, hidden_size, intermediate_size] and then B_gate must be [hidden_size, intermediate_size]. We'll construct that via Triton by copying expert_gate[exp].permute(1, 0) into a buffer.
                # We need to launch per-row matmul with A_row = h_in and B_gate = expert_gate[exp].permute(1, 0). We'll create a temporary B_gate_buf for per-row matmul.
                B_gate_buf = expert_gate[exp].permute(1, 0).contiguous()  # [hidden_size, intermediate_size]
                gate_out = _launch_per_row_matmul(gate_out, h_in, B_gate_buf, hidden_size, B_gate_buf.stride(0), B_gate_buf.stride(1))

                # up_out: hidden_size
                up_out = torch.empty((hidden_size,), dtype=torch.float32, device=device)
                B_up = expert_up[exp].permute(1, 0).contiguous()  # [hidden_size, intermediate_size]
                up_out = _launch_per_row_matmul(up_out, h_in, B_up, hidden_size, B_up.stride(0), B_up.stride(1))

                # SiLU
                gate_silu = torch.empty((hidden_size,), dtype=torch.float32, device=device)
                _silu_kernel[(hidden_size,)](gate_silu, gate_out, hidden_size)

                # SwiGLU elementwise multiply
                activated = torch.empty((hidden_size,), dtype=torch.float32, device=device)
                _mul_kernel[(hidden_size,)](activated, gate_silu, up_out, hidden_size)

                # expert_outputs: hidden_size
                B_down = expert_down[exp].permute(1, 0).contiguous()  # [intermediate_size, hidden_size]
                expert_outputs = torch.empty((hidden_size,), dtype=torch.float32, device=device)
                expert_outputs = _launch_per_row_matmul(expert_outputs, activated, B_down, hidden_size, B_down.stride(0), B_down.stride(1))

                # Atomic add weighted contribution to result[t, :]
                weight = routing_weights[t, k].item()  # scalar
                vals = expert_outputs  # vector of length hidden_size
                _atomic_add_weighted_kernel[(hidden_size,)](result, vals, torch.tensor([weight], device=device, dtype=torch.float32), num_tokens, hidden_size, t)

        # 6) Return result cast to bfloat16 (to match original dtype)
        return result.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
