import torch
import triton
import triton.language as tl


@triton.jit
def triton_concatenation(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_l, stride_o_h,
    BLOCK_H: tl.constexpr,
):
    # Grid: (B, L, H tiles) where L = T + I
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)  # position in [0, L)
    pid_k = tl.program_id(2)  # tile along H

    k = pid_k * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_k = k < H

    use_encoder = pid_l < T

    if use_encoder:
        e_off = pid_b * stride_e_b + pid_l * stride_e_t + k * stride_e_h
        o_off = pid_b * stride_o_b + pid_l * stride_o_l + k * stride_o_h
        val = tl.load(encoder_ptr + e_off, mask=mask_k, other=0.0)
    else:
        h_off = pid_b * stride_h_b + (pid_l - T) * stride_h_i + k * stride_h_h
        o_off = pid_b * stride_o_b + pid_l * stride_o_l + k * stride_o_h
        val = tl.load(hidden_ptr + h_off, mask=mask_k, other=0.0)

    tl.store(out_ptr + o_off, val, mask=mask_k)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    B, M, N, K,
    stride_a_b, stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_c_b, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles along M, tiles along N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + pid_b * stride_a_b + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # loads as tensor dtype; we cast to fp32

        b_ptrs = B_ptr + offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Cast to fp32 for stable accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + pid_b * stride_c_b + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D [B, dim, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        assert hidden_states.dtype == encoder_hidden_states.dtype == process_weight.dtype, "All tensors must have same dtype"

        L = T + I

        # Concatenate along sequence: [B, L, H]
        out_cat = torch.empty((B, L, H), dtype=hidden_states.dtype, device=hidden_states.device)
        BLOCK_H = 64
        grid_concat = (B, L, triton.cdiv(H, BLOCK_H))
        triton_concatenation[grid_concat](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(),
            *out_cat.stride(),
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Prepare weight^T: [H, H]
        W_T = process_weight.t()  # [H, H]

        # Perform GEMM in fp32 for


def run(*args):
    return ModelNew()(*args)
