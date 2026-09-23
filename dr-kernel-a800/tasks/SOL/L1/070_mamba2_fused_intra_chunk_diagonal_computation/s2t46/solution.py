import torch
import triton
import triton.language as tl

# Constants from the original code
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
STATE_SIZE = 128
HEAD_DIM = 128

@triton.jit
def build_L_kernel(
    A_ptr,                  # [B, H, NC, CS] input A_cumsum
    L_ptr,                  # [B, H, NC, CS, CS] output L
    B_runtime, H_runtime, NC_runtime, CS_runtime,
    stride_A_b, stride_A_h, stride_A_nc, stride_A_cs,
    stride_L_b, stride_L_h, stride_L_nc, stride_L_k, stride_L_j,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)  # i is num_chunks index

    for j in tl.static_range(CHUNK_SIZE):
        s = 0.0
        # cumsum over k from 0..j
        for k in tl.static_range(j + 1):
            a = tl.load(
                A_ptr
                + b * stride_A_b
                + h * stride_A_h
                + i * stride_A_nc
                + k * stride_A_cs
            )
            s += a
        val = tl.exp(s)
        # Store only if i >= j (lower-triangular). Else we store 0.
        if i >= j:
            tl.store(
                L_ptr
                + b * stride_L_b
                + h * stride_L_h
                + i * stride_L_nc
                + j * stride_L_k
                + j * stride_L_j,  # both k and j index are j here
                val,
            )
        else:
            tl.store(
                L_ptr
                + b * stride_L_b
                + h * stride_L_h
                + i * stride_L_nc
                + j * stride_L_k
                + j * stride_L_j,
                0.0,
            )

@triton.jit
def compute_G_kernel(
    B_ptr,                  # [B, NC, CS, NG, SS] input B
    C_ptr,                  # [B, NC, CS, NG, SS] input C
    G_ptr,                  # [B, NC, CS, H] output G
    B_runtime, NC_runtime, CS_runtime, NG_runtime, SS_runtime, H_runtime,
    stride_B_b, stride_B_nc, stride_B_cs, stride_B_ng, stride_B_ss,
    stride_C_b, stride_C_nc, stride_C_cs, stride_C_ng, stride_C_ss,
    stride_G_b, stride_G_nc, stride_G_cs, stride_G_h,
):
    # Grid over (b, i, h)
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)

    # Accumulator for G[i, :, h]
    G_vec = tl.zeros((CHUNK_SIZE,), dtype=tl.float32)
    # Loop over j
    for j in tl.static_range(CHUNK_SIZE):
        # Expand heads for B and C: original has n_groups=8, we want num_heads=32
        # Emulate rep_interleave: h maps to original group index (h % 8)
        # For B and C, the ng dimension is N_GROUPS. We select the corresponding group via modulo.
        # We compute dot over k in 0..CS-1
        dot = 0.0
        for k in tl.static_range(CHUNK_SIZE):
            # Load B[j, k, h, s] for all s
            # Note: We loop s = 0..STATE_SIZE-1 and accumulate
            for s in tl.static_range(STATE_SIZE):
                B_val = tl.load(
                    B_ptr
                    + b * stride_B_b
                    + i * stride_B_nc
                    + k * stride_B_cs
                    + (h % (H_runtime // 4)) * stride_B_ng
                    + s * stride_B_ss
                )
                C_val = tl.load(
                    C_ptr
                    + b * stride_C_b
                    + i * stride_C_nc
                    + k * stride_C_cs
                    + (h % (H_runtime // 4)) * stride_C_ng
                    + s * stride_C_ss
                )
                dot += B_val * C_val
        G_vec[j] = dot
    # Store G_vec to G[b, i, :, h]
    for j in tl.static_range(CHUNK_SIZE):
        tl.store(
            G_ptr + b * stride_G_b + i * stride_G_nc + j * stride_G_cs + h * stride_G_h,
            G_vec[j],
        )

@triton.jit
def multiply_LG_kernel(
    L_ptr, G_ptr, M_ptr,
    B_runtime, NC_runtime, CS_runtime, H_runtime,
    stride_L_b, stride_L_h, stride_L_nc, stride_L_k, stride_L_j,
    stride_G_b, stride_G_nc, stride_G_cs, stride_G_h,
    stride_M_b, stride_M_nc, stride_M_k, stride_M_j, stride_M_h,
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)
    for j in tl.static_range(CHUNK_SIZE):
        L_val = tl.load(
            L_ptr
            + b * stride_L_b
            + h * stride_L_h
            + i * stride_L_nc
            + j * stride_L_k
            + j * stride_L_j
        )
        G_val = tl.load(
            G_ptr + b * stride_G_b + i * stride_G_nc + j * stride_G_cs + h * stride_G_h
        )
        M_val = L_val * G_val
        tl.store(
            M_ptr + b * stride_M_b + i * stride_M_nc + j * stride_M_k + h * stride_M_h,
            M_val,
        )

@triton.jit
def contract_M_hidden_into_Ydiag_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_runtime, NC_runtime, CS_runtime, H_runtime, HD_runtime,
    stride_M_b, stride_M_nc, stride_M_k, stride_M_j, stride_M_h,
    stride_h_b, stride_h_nc, stride_h_cs, stride_h_h, stride_h_d,
    stride_Y_b, stride_Y_nc, stride_Y_cs, stride_Y_h, stride_Y_d,
):
    # Grid over (b, i, k, h, d)
    b = tl.program_id(0)
    i = tl.program_id(1)
    k = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in tl.static_range(CHUNK_SIZE):
        M_val = tl.load(
            M_ptr + b * stride_M_b + i * stride_M_nc + k * stride_M_k + j * stride_M_j + h * stride_M_h
        )
        hidden_val = tl.load(
            hidden_ptr + b * stride_h_b + i * stride_h_nc + k * stride_h_cs + j * stride_h_h + d * stride_h_d
        )
        acc += M_val * hidden_val
    tl.store(
        Y_ptr + b * stride_Y_b + i * stride_Y_nc + k * stride_Y_cs + h * stride_Y_h + d * stride_Y_d,
        acc,
    )

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor):
        # Dimensions (runtime)
        B_runtime = hidden_states.shape[0]
        NC_runtime = hidden_states.shape[1]
        CS_runtime = hidden_states.shape[2]
        H_runtime = hidden_states.shape[3]
        HD_runtime = hidden_states.shape[4]

        # Device and dtype
        device = hidden_states.device
        dtype_float = torch.float32

        # 1) Allocate L: [B, H, NC, CS, CS] float32
        L = torch.empty((B_runtime, NUM_HEADS, NC_runtime, CS_runtime, CS_runtime),
                        dtype=dtype_float, device=device)
        # Strides for A and L
        stride_A_b, stride_A_h, stride_A_nc, stride_A_cs = (
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3)
        )
        stride_L_b, stride_L_h, stride_L_nc, stride_L_k, stride_L_j = (
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4)
        )
        # Launch build_L_kernel: grid over (B, H, NC)
        grid_L = (B_runtime, NUM_HEADS, NC_runtime)
        build_L_kernel[grid_L](
            A_cumsum, L,
            B_runtime, NUM_HEADS, NC_runtime, CS_runtime,
            stride_A_b, stride_A_h, stride_A_nc, stride_A_cs,
            stride_L_b, stride_L_h, stride_L_nc, stride_L_k, stride_L_j,
            num_warps=2, num_stages=2,
        )

        # 2) Allocate G: [B, NC, CS, H] float32
        G = torch.empty((B_runtime, NC_runtime, CS_runtime, NUM_HEADS),
                        dtype=dtype_float, device=device)
        # Strides for B, C, G
        stride_B_b, stride_B_nc, stride_B_cs, stride_B_ng, stride_B_ss = (
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4)
        )
        stride_C_b, stride_C_nc, stride_C_cs, stride_C_ng, stride_C_ss = (
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4)
        )
        stride_G_b, stride_G_nc, stride_G_cs, stride_G_h = (
            G.stride(0), G.stride(1), G.stride(2), G.stride(3)
        )
        # Launch compute_G_kernel: grid over (B, NC, H)
        grid_G = (B_runtime, NC_runtime, NUM_HEADS)
        compute_G_kernel[grid_G](
            B, C, G,
            B_runtime, NC_runtime, CS_runtime, N_GROUPS, STATE_SIZE, NUM_HEADS,
            stride_B_b, stride_B_nc, stride_B_cs, stride_B_ng, stride_B_ss,
            stride_C_b, stride_C_nc, stride_C_cs, stride_C_ng, stride_C_ss,
            stride_G_b, stride_G_nc, stride_G_cs, stride_G_h,
            num_warps=4, num_stages=2,
        )

        # 3) Allocate M: [B, NC, CS, CS, H] float32
        M = torch.empty((B_runtime, NC_runtime, CS_runtime, CS_runtime, NUM_HEADS),
                        dtype=dtype_float, device=device)
        stride_M_b, stride_M_nc, stride_M_k, stride_M_j, stride_M_h = (
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4)
        )
        # Launch multiply_LG_kernel: grid over (B, NC, H)
        grid_MG = (B_runtime, NC_runtime, NUM_HEADS)
        multiply_LG_kernel[grid_MG](
            L, G, M,
            B_runtime, NC_runtime, CS_runtime, NUM_HEADS,
            stride_L_b, stride_L_h, stride_L_nc, stride_L_k, stride_L_j,
            stride_G_b, stride_G_nc, stride_G_cs, stride_G_h,
            stride_M_b, stride_M_nc, stride_M_k, stride_M_j, stride_M_h,
            num_warps=4, num_stages=2,
        )

        # 4) Contract M*hidden to Y_diag: [B, NC, CS, H, HD] bfloat16
        Y_diag = torch.empty((B_runtime, NC_runtime, CS_runtime, NUM_HEADS, HD_runtime),
                             dtype=torch.bfloat16, device=device)

        stride_h_b, stride_h_nc, stride_h_cs, stride_h_h, stride_h_d = (
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            hidden_states.stride(3), hidden_states.stride(4)
        )
        stride_Y_b, stride_Y_nc, stride_Y_cs, stride_Y_h, stride_Y_d = (
            Y_diag.stride(0), Y_diag.stride(1), Y_diag.stride(2), Y_diag.stride(3), Y_diag.stride(4)
        )
        # Launch contraction kernel: grid over (B, NC, CS, H, HD)
        grid_contract = (B_runtime, NC_runtime, CS_runtime, NUM_HEADS, HD_runtime)
        contract_M_hidden_into_Ydiag_kernel[grid_contract](
            M, hidden_states, Y_diag,
            B_runtime, NC_runtime, CS_runtime, NUM_HEADS, HD_runtime,
            stride_M_b, stride_M_nc, stride_M_k, stride_M_j, stride_M_h,
            stride_h_b, stride_h_nc, stride_h_cs, stride_h_h, stride_h_d,
            stride_Y_b, stride_Y_nc, stride_Y_cs, stride_Y_h, stride_Y_d,
            num_warps=4, num_stages=2,
        )

        return Y_diag


def run(*args):
    return ModelNew()(*args)
