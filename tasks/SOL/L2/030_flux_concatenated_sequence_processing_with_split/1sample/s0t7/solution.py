import torch
import triton
import triton.language as tl


@triton.jit
def _concat_encoder_hidden_kernel(
    enc_ptr,         # *ptr to encoder_hidden_states [B, T, H]
    hid_ptr,         # *ptr to hidden_states [B, I, H]
    out_ptr,         # *ptr to out_cat [B, T+I, H]
    B, T, I, H,      # ints
):
    b = tl.program_id(0)
    L = T + I
    # Each program handles one batch
    # Loop over sequence positions l in [0, L)
    for l in range(0, L):
        # If l < T, read from encoder; else read from hidden
        if l < T:
            ptr = enc_ptr + b * H + l * H
        else:
            l_hid = l - T
            ptr = hid_ptr + b * H + l_hid * H
        # Store to out_cat[b, l, :]
        out_row_ptr = out_ptr + b * L * H + l * H
        # Load the entire row (H elements)
        for h in range(0, H):
            val = tl.load(ptr + h)
            tl.store(out_row_ptr + h, val)


@triton.jit
def _triton_gemm_right_kernel(
    A_ptr,           # *ptr to A_flat [M, K], M = B*(T+I), K = H
    W_ptr,           # *ptr to W [K, K], note: we load W^T as W[n, k]
    C_ptr,           # *ptr to C [M, K], output
    M, K,            # ints, M rows, K cols
    stride_am,       # stride for A rows (usually K)
    stride_ak,       # stride for A cols (usually 1)
    stride_cm,       # stride for C rows (usually K)
    stride_cn,       # stride for C cols (usually 1)
    BLOCK_M: tl.constexpr,  # tile size along M
    BLOCK_N: tl.constexpr,  # tile size along N (== K here)
    BLOCK_K: tl.constexpr,  # reduction tile along K
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # mask for valid rows/cols
    mask_m = offs_m < M
    mask_n = offs_n < K

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load W^T tile: we want [BLOCK_K, BLOCK_N], i.e., W[n, k] with n=offs_k, k=offs_n
        w_ptrs = W_ptr + offs_n[None, :] * offs_k[:, None]  # since W is [K,K], W[n,k] = W[n, k]
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, w)

    # Store result to C
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _split_outputs_kernel(
    C_ptr,            # *ptr to C [M, K], M = B*(T+I), K = H
    enc_out_ptr,      # *ptr to processed_encoder [B, T, H]
    hid_out_ptr,      # *ptr to processed_hidden [B, I, H]
    B, T, I, K,       # ints
    stride_cm,        # stride for C rows (usually K)
    stride_cn,        # stride for C cols (usually 1)
    enc_stride_b, enc_stride_t, enc_stride_h,  # strides for encoder output
    hid_stride_b, hid_stride_i, hid_stride_h,  # strides for hidden output
):
    # Each program handles one batch; we iterate over rows m
    b = tl.program_id(0)
    # Encode rows: m in [0, B*T)
    for m in range(0, B * T):
        batch = m // T
        seq = m % T
        c_row_ptr = C_ptr + m * K
        # Copy row to encoder output [B, T, H]
        for h in range(0, K):
            val = tl.load(c_row_ptr + h)
            tl.store(enc_out_ptr + batch * enc_stride_b + seq * enc_stride_t + h * enc_stride_h, val)

    # Image rows: m in [B*T, B*(T+I))
    for m in range(B * T, B * (T + I)):
        batch = m // (T + I)
        seq = m % (T + I)
        # If seq >= T, it's fine; if not, adjust (though m starts at B*T)
        c_row_ptr = C_ptr + m * K
        # Copy row to hidden output [B, I, H]
        for h in range(0, K):
            val = tl.load(c_row_ptr + h)
            tl.store(hid_out_ptr + batch * hid_stride_b + seq * hid_stride_i + h * hid_stride_h, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match for encoder and hidden inputs"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # Ensure contiguous tensors
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        W = process_weight.contiguous()

        device = enc.device

        # 1) Concatenate in Triton: out_cat [B, T+I, H]
        L = T + I
        out_cat = torch.empty((B, L, H), dtype=torch.float32, device=device)
        _concat_encoder_hidden_kernel[(B,)](
            enc, hid, out_cat,
            B, T, I, H,
            num_warps=1, num_stages=1,
        )

        # 2) GEMM in Triton: A_flat = out_cat.view(B*(T+I), H), W is [H, H], C = A @ W^T
        M = B * L
        A_flat = out_cat.reshape(M, H).contiguous()
        C = torch.empty((M, H), dtype=torch.float32, device=device)
        # Launch 2D grid over tiles
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (H + BLOCK_N - 1) // BLOCK_N
        _triton_gemm_right_kernel[(grid_m, grid_n)](
            A_flat, W, C,
            M, H,
            H, 1,          # stride_am = H (since row stride is H), stride_ak = 1
            H, 1,          # stride_cm = H, stride_cn = 1
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split C into encoder and hidden outputs in Triton
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)
        _split_outputs_kernel[(B,)](
            C,
            processed_encoder, processed_hidden,
            B, T, I, H,
            H, 1,                      # C strides
            *processed_encoder.stride(),
            *processed_hidden.stride(),
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
