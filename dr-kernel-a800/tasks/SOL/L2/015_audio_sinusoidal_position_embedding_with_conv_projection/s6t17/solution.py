import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_im2col_matmul_gelu(
    x_ptr,          # *ptr to input [B, C_in, H, W]
    w_ptr,          # *ptr to weights [C_out, C_in, 3, 3]
    bias_ptr,       # *ptr to bias [C_out]
    y_ptr,          # *ptr to output [B, C_out, H_out, W_out]
    B, C_in, H, W, C_out,
    H_out, W_out,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_y_b, stride_y_co, stride_y_ho, stride_y_wo,
    BLOCK_CO: tl.constexpr,  # tile over output channels
    BLOCK_M: tl.constexpr,   # tile over positions M = H_out * W_out
    NUM_COL: tl.constexpr,   # number of columns per position = C_in * 9
):
    # program ids
    b = tl.program_id(0)  # batch
    co_block = tl.program_id(1)  # tile over output channels
    m_block = tl.program_id(2)   # tile over positions (H_out * W_out)

    # compute indices for this block
    co_offsets = co_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    # map m_offsets -> (ho, wo)
    # m = ho * W_out + wo
    ho = m_offsets // W_out
    wo = m_offsets % W_out

    # build X_col matrix [BLOCK_M, NUM_COL]
    # NUM_COL = C_in * 9 (3x3 neighborhood)
    X_col = tl.zeros((BLOCK_M, NUM_COL), dtype=tl.float32)

    # loop over ci and kh,kw to fill X_col
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # compute input indices
                hi = ho * 2 + 1 - kh  # since stride=2, padding=1 => ho*2+1
                wi = wo * 2 + 1 - kw
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                idx = ((b * C_in + ci) * H + hi) * W + wi  # [BLOCK_M]
                # load input values with mask
                x_vals = tl.load(x_ptr + idx, mask=in_bounds, other=0.0)
                # column index within NUM_COL
                col_idx = ci * 9 + kh * 3 + kw
                X_col[:, col_idx] = x_vals

    # build W_col matrix [BLOCK_CO, NUM_COL]
    W_col = tl.zeros((BLOCK_CO, NUM_COL), dtype=tl.float32)
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                col_idx = ci * 9 + kh * 3 + kw
                base = co_offsets * (C_in * 9)
                w_idx = base + col_idx  # [BLOCK_CO]
                w_vals = tl.load(w_ptr + w_idx)  # [BLOCK_CO]
                W_col[:, col_idx] = w_vals

    # add bias to W_col
    b_vals = tl.load(bias_ptr + co_offsets, mask=co_offsets < C_out, other=0.0)  # [BLOCK_CO]
    W_col = W_col + b_vals[:, None]

    # compute matmul: y_mat = W_col @ X_col  -> [BLOCK_CO, BLOCK_M]
    y_mat = tl.zeros((BLOCK_CO, BLOCK_M), dtype=tl.float32)
    # broadcast multiply and reduce
    # W_col: [CO, COL], X_col: [M, COL] -> we need [CO, M], so we compute y_mat = sum over COL
    for c in range(0, NUM_COL):
        w_col_vec = W_col[:, c]            # [BLOCK_CO]
        x_col_vec = X_col[:, c]            # [BLOCK_M]
        # outer product contribution
        y_mat += w_col_vec[:, None] * x_col_vec[None, :]

    # apply GELU in-kernel (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    for co_i in range(0, BLOCK_CO):
        if co_i < BLOCK_CO:
            y_vec = y_mat[co_i, :]
            x3 = y_vec * y_vec * y_vec
            tanh_arg = c * (y_vec + 0.044715 * x3)
            y_vec = 0.5 * y_vec * (1.0 + tl.math.tanh(tanh_arg))
            y_mat[co_i, :] = y_vec

    # store to output y[b, co, ho, wo]
    # map m_offsets -> (ho, wo)
    for co_i in range(0, BLOCK_CO):
        if co_i < BLOCK_CO:
            y_vec = y_mat[co_i, :]
            for m_i in range(0, BLOCK_M):
                if m_i < BLOCK_M:
                    ho_i = ho[m_i]
                    wo_i = wo[m_i]
                    out_idx = ((b * C_out + co_i) * H_out + ho_i) * W_out + wo_i
                    tl.store(y_ptr + out_idx, y_vec[m_i])

    # nothing to return; y_ptr is updated in place


@triton.jit
def linear_project_kernel(
    x_ptr,              # *ptr to x [B, T, K]
    w_ptr,              # *ptr to conv_out_weight [N, K]
    y_ptr,              # *ptr to output [B, T, N]
    B, T, K, N,
    stride_x_b, stride_x_t, stride_x_k,
    stride_w_d, stride_w_k,
    stride_y_b, stride_y_t, stride_y_d,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_block = tl.program_id(2)

    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + k_offsets
        mask_k = k_idx < K

        # load x[b, t, k: k+BLOCK_K] -> vector of length BLOCK_K
        x_off = b * stride_x_b + t * stride_x_t + k_idx * stride_x_k
        x_vec = tl.load(x_ptr + x_off, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

        # load W[n_offsets, k: k+BLOCK_K] -> matrix [BLOCK_N, BLOCK_K]
        w_off = n_offsets[:, None] * stride_w_d + k_idx[None, :] * stride_w_k
        w_mat = tl.load(w_ptr + w_off, mask=(n_offsets[:, None] < N) & mask_k[None, :], other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # dot: acc[n] += sum_k w_mat[n,k] * x_vec[k]
        # acc += sum over K of w_mat[n,k] * x_vec[k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)  # [BLOCK_N]

    # store results
    y_off = b * stride_y_b + t * stride_y_t + n_offsets * stride_y_d
    tl.store(y_ptr + y_off, acc, mask=n_offsets < N)


@triton.jit
def add_pos_embed_kernel(
    y_ptr,              # *ptr to y [B, T, N]
    pos_ptr,            # *ptr to positional embedding [T, N]
    out_ptr,            # *ptr to output [B, T, N]
    B, T, N,
    stride_y_b, stride_y_t, stride_y_d,
    stride_pos_t, stride_pos_d,
    stride_out_b, stride_out_t, stride_out_d,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(0)
    t_block = tl.program_id(1)
    n_block = tl.program_id(2)

    t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # load y[b, t, n] and pos[t, n], then add
    for t_i in range(0, BLOCK_T):
        if t_i < BLOCK_T:
            for n_i in range(0, BLOCK_N):
                if n_i < BLOCK_N:
                    y_val = tl.load(y_ptr + b * stride_y_b + t_offsets[t_i] * stride_y_t + n_offsets[n_i] * stride_y_d)
                    pos_val = tl.load(pos_ptr + t_offsets[t_i] * stride_pos_t + n_offsets[n_i] * stride_pos_d)
                    out_val = y_val + pos_val
                    tl.store(out_ptr + b * stride_out_b + t_offsets[t_i] * stride_out_t + n_offsets[n_i] * stride_out_d, out_val)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        # Ensure all tensors are on the same device
        device = input_features.device
        dtype = input_features.dtype

        # Convert inputs to float32 for numerical stability, as Triton kernels operate in fp32
        x = input_features.to(torch.float32)
        w1 = conv2d1_weight.to(torch.float32)
        b1 = conv2d1_bias.to(torch.float32)
        w2 = conv2d2_weight.to(torch.float32)
        b2 = conv2d2_bias.to(torch.float32)
        w3 = conv2d3_weight.to(torch.float32)
        b3 = conv2d3_bias.to(torch.float32)
        conv_out_weight = conv_out_weight.to(torch.float32)  # [N, K], N=d_model, K=channels*freq*last_time
        pos = positional_embedding.to(torch.float32)          # [max_source_positions, d_model]

        B, C_in, H, W = x.shape
        # Conv1: (1, 384) with stride=2, padding=1 -> H1=40, W1 depends on W
        H1 = (H + 2 * 1 - 3) // 2 + 1
        W1 = (W + 2 * 1 - 3) // 2 + 1
        C_out1 = w1.shape[0]

        # Prepare output tensor for conv1
        y1 = torch.empty((B, C_out1, H1, W1), dtype=torch.float32, device=device)

        # Launch conv1 kernel: grid over (B, tiles over C_out1, tiles over positions M=H1*W1)
        BLOCK_CO1 = 64
        M = H1 * W1
        NUM_COL1 = C_in * 9  # 1*9 for input channel 1
        grid1 = (B, triton.cdiv(C_out1, BLOCK_CO1), triton.cdiv(M, 128))
        conv_im2col_matmul_gelu[grid1](
            x, w1, b1, y1,
            B, C_in, H, W, C_out1, H1, W1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_CO=BLOCK_CO1, BLOCK_M=128, NUM_COL=NUM_COL1,
            num_warps=4, num_stages=2,
        )

        # Conv2: (384 -> 384) on y1
        B2, C_in2, H2, W2 = y1.shape
        C_out2 = w2.shape[0]
        H2_out = (H2 + 2 * 1 - 3) // 2 + 1
        W2_out = (W2 + 2 * 1 - 3) // 2 + 1

        y2 = torch.empty((B2, C_out2, H2_out, W2_out), dtype=torch.float32, device=device)

        NUM_COL2 = C_in2 * 9
        grid2 = (B2, triton.cdiv(C_out2, 64), triton.cdiv(H2_out * W2_out, 128))
        conv_im2col_matmul_gelu[grid2](
            y1, w2, b2, y2,
            B2, C_in2, H2, W2, C_out2, H2_out, W2_out,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_CO=64, BLOCK_M=128, NUM_COL=NUM_COL2,
            num_warps=4, num_stages=2,
        )

        # Conv3: (384 -> 384) on y2
        B3, C_in3, H3, W3 = y2.shape
        C_out3 = w3.shape[0]
        H3_out = (H3 + 2 * 1 - 3) // 2 + 1
        W3_out = (W3 + 2 * 1 - 3) // 2 + 1

        y3 = torch.empty((B3, C_out3, H3_out, W3_out), dtype=torch.float32, device=device)

        NUM_COL3 = C_in3 * 9
        grid3 = (B3, triton.cdiv(C_out3, 64), triton.cdiv(H3_out * W3_out, 128))
        conv_im2col_matmul_gelu[grid3](
            y2, w3, b3, y3,
            B3, C_in3, H3, W3, C_out3, H3_out, W3_out,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_CO=64, BLOCK_M=128, NUM_COL=NUM_COL3,
            num_warps=4, num_stages=2,
        )

        # Flatten to (B, T, K), where T=W3_out and K=C_out3*H3_out*W3_out
        Bf, Cf, Hf, Wf = y3.shape
        T = Wf
        K = Cf * Hf * Wf
        x_flat = y3.view(Bf, T, K)

        # Linear projection: conv_out_weight [N, K] with N=d_model=1024
        N = conv_out_weight.shape[0]
        y_proj = torch.empty((Bf, T, N), dtype=torch.float32, device=device)

        stride_x_b, stride_x_t, stride_x_k = x_flat.stride()
        stride_w_d, stride_w_k = conv_out_weight.stride()
        stride_y_b, stride_y_t, stride_y_d = y_proj.stride()

        # Launch Triton linear projection kernel
        BLOCK_N = 128
        BLOCK_K = 256
        grid_linear = (Bf, T, triton.cdiv(N, BLOCK_N))
        linear_project_kernel[grid_linear](
            x_flat, conv_out_weight, y_proj,
            Bf, T, K, N,
            stride_x_b, stride_x_t, stride_x_k,
            stride_w_d, stride_w_k,
            stride_y_b, stride_y_t, stride_y_d,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Scale by embed_scale (sqrt(1024) = 32.0)
        y_proj = y_proj * embed_scale

        # Add positional embedding: pos is [max_source_positions, d_model], we need to slice first T columns
        # out shape: [B, T, N]
        out = torch.empty_like(y_proj)

        stride_y_b2, stride_y_t2, stride_y_d2 = y_proj.stride()
        stride_pos_t, stride_pos_d = pos.stride()
        stride_out_b, stride_out_t, stride_out_d = out.stride()

        BLOCK_N2 = 128
        BLOCK_T2 = 32
        grid_pos = (Bf, triton.cdiv(T, BLOCK_T2), triton.cdiv(N, BLOCK_N2))
        add_pos_embed_kernel[grid_pos](
            y_proj, pos, out,
            Bf, T, N,
            stride_y_b2, stride_y_t2, stride_y_d2,
            stride_pos_t, stride_pos_d,
            stride_out_b, stride_out_t, stride_out_d,
            BLOCK_N=BLOCK_N2, BLOCK_T=BLOCK_T2,
            num_warps=4, num_stages=2,
        )

        # Return in original dtype
        return out.to(dtype)


def run(*args):
    return ModelNew()(*args)
