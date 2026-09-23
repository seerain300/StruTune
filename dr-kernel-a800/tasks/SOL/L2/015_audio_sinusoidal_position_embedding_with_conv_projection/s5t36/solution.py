import math
import torch
import torch.nn as nn

# Triton is required
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
    TRITON_AVAILABLE = False  # force fallback if import fails


# -------- Triton kernels --------

if TRITON_AVAILABLE:
    @triton.jit
    def conv_ci1_stride2_bias_gelu_kernel(
        x_ptr,            # *f32: input [N, 1, In, T]
        w_ptr,            # *f32: weight [Co, 1, 3, 3]
        b_ptr,            # *f32: bias [Co]
        y_ptr,            # *f32: output [N, Co, Out, T_out]
        N, In, Out, T,    # int32: In=80, Out=80, T=time, T_out=(T-3)//2+1
        x_stride_n, x_stride_c, x_stride_in, x_stride_t,
        w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
        y_stride_n, y_stride_co, y_stride_out, y_stride_t,
        T_out,             # int32: output time length
        BLOCK_CO: tl.constexpr,  # tile size for channels
        BLOCK_OUT: tl.constexpr, # tile size for frequency
        BLOCK_T: tl.constexpr,   # tile size for time
    ):
        n = tl.program_id(0)
        co_block = tl.program_id(1)
        out_block = tl.program_id(2)
        t_block = tl.program_id(3)

        co_offsets = co_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
        out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)

        mask_co = co_offsets < 384
        mask_out = out_offsets < 80
        mask_t = t_offsets < T_out

        # Accumulator for output channels
        acc = tl.zeros((BLOCK_CO, BLOCK_OUT, BLOCK_T), dtype=tl.float32)

        # Iterate over kernel window and input channels (Ci=1)
        for kh in range(3):
            for kw in range(3):
                t_in = t_offsets + kh - 1  # consider padding
                in_out = out_offsets + kw - 1
                mask_t_in = (t_in >= 0) & (t_in < T)
                mask_in_out = (in_out >= 0) & (in_out < In)
                mask_all = mask_t[:, None, None] & mask_out[None, :, None] & mask_t_in[None, None, :] & mask_in_out[None, None, :]

                # Load input x[n, 0, in_out, t_in]
                x_idx = n * x_stride_n + 0 * x_stride_c + in_out[None, None, :] * x_stride_in + t_in[:, None, None] * x_stride_t
                x_val = tl.load(x_ptr + x_idx, mask=mask_all, other=0.0).to(tl.float32)  # shape [BLOCK_T, BLOCK_OUT, BLOCK_CO]

                # Load weights for each co channel
                # w[co, 0, kh, kw] -> shape [BLOCK_CO]
                w_idx = co_offsets[:, None, None] * w_stride_co + 0 * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_idx, mask=mask_co[:, None, None], other=0.0).to(tl.float32)  # shape [BLOCK_CO]

                # Accumulate: acc[co, out, t] += w[co] * x[t, out] (sum over co implicitly handled via broadcasting)
                # We need to broadcast x_val over co and w_val over out/t
                # Compute: sum over co -> use x_val * w_val for each co slice, but x_val has co in last dim implicitly, so:
                # Broadcast multiply then reduce over co dimension:
                # Better: acc += w_val[:, None, None] * x_val[None, :, :]
                acc += w_val[:, None, None] * x_val[None, :, :]

        # Add bias
        b_val = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0).to(tl.float32)  # [BLOCK_CO]
        acc += b_val[:, None, None]  # broadcast across out and t

        # GELU (tanh approximation)
        # y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        c0 = 0.5
        c1 = 0.7978845608028654  # sqrt(2/pi)
        c2 = 0.044715
        acc_cub = acc * acc * acc
        gelu_inner = c1 * (acc + c2 * acc_cub)
        gelu_approx = c0 * acc * (1.0 + tl.tanh(gelu_inner))
        acc = gelu_approx

        # Store to y[n, co, out, t]
        y_idx = n * y_stride_n + co_offsets[:, None, None] * y_stride_co + out_offsets[None, :, None] * y_stride_out + t_offsets[None, None, :] * y_stride_t
        store_mask = mask_co[:, None, None] & mask_out[None, :, None] & mask_t[None, None, :]
        tl.store(y_ptr + y_idx, acc, mask=store_mask)

    @triton.jit
    def conv_generic_stride2_bias_gelu_kernel(
        x_ptr,            # *f32: input [N, Ci, In, T]
        w_ptr,            # *f32: weight [Co, Ci, 3, 3]
        b_ptr,            # *f32: bias [Co]
        y_ptr,            # *f32: output [N, Co, Out, T_out]
        N, Ci, In, Out, T,  # int32: In=80, Out=40 or 20, T=time, T_out=(T-3)//2+1
        x_stride_n, x_stride_c, x_stride_in, x_stride_t,
        w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
        y_stride_n, y_stride_co, y_stride_out, y_stride_t,
        T_out,
        BLOCK_CO: tl.constexpr,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        n = tl.program_id(0)
        co_block = tl.program_id(1)
        out_block = tl.program_id(2)
        t_block = tl.program_id(3)

        co_offsets = co_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
        out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)

        mask_co = co_offsets < 384
        mask_out = out_offsets < Out
        mask_t = t_offsets < T_out

        acc = tl.zeros((BLOCK_CO, BLOCK_OUT, BLOCK_T), dtype=tl.float32)

        for ci in range(Ci):
            # Loop over 3x3 kernel window
            for kh in range(3):
                for kw in range(3):
                    t_in = t_offsets + kh - 1
                    in_out = out_offsets + kw - 1
                    mask_t_in = (t_in >= 0) & (t_in < T)
                    mask_in_out = (in_out >= 0) & (in_out < In)
                    mask_all = mask_t[:, None, None] & mask_out[None, :, None] & mask_t_in[None, None, :] & mask_in_out[None, None, :]

                    # Load x[n, ci, in_out, t_in]
                    x_idx = n * x_stride_n + ci * x_stride_c + in_out[None, None, :] * x_stride_in + t_in[:, None, None] * x_stride_t
                    x_val = tl.load(x_ptr + x_idx, mask=mask_all, other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_OUT, 1]

                    # Load w[co, ci, kh, kw] for all co in block
                    w_idx = co_offsets[:, None, None] * w_stride_co + ci * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                    w_val = tl.load(w_ptr + w_idx, mask=mask_co[:, None, None], other=0.0).to(tl.float32)  # [BLOCK_CO]

                    # Accumulate: acc[co, out, t] += w[co] * x[t, out]
                    acc += w_val[:, None, None] * x_val[None, :, :]

        # Add bias
        b_val = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0).to(tl.float32)
        acc += b_val[:, None, None]

        # GELU
        c0 = 0.5
        c1 = 0.7978845608028654
        c2 = 0.044715
        acc_cub = acc * acc * acc
        gelu_inner = c1 * (acc + c2 * acc_cub)
        gelu_approx = c0 * acc * (1.0 + tl.tanh(gelu_inner))
        acc = gelu_approx

        # Store
        y_idx = n * y_stride_n + co_offsets[:, None, None] * y_stride_co + out_offsets[None, :, None] * y_stride_out + t_offsets[None, None, :] * y_stride_t
        store_mask = mask_co[:, None, None] & mask_out[None, :, None] & mask_t[None, None, :]
        tl.store(y_ptr + y_idx, acc, mask=store_mask)

    @triton.jit
    def linear_bmm_kernel(
        x_ptr,            # *f32: input [N, T_out3, M=3840]
        w_ptr,            # *f32: weight [M, K=1024]
        y_ptr,            # *f32: output [N, T_out3, K=1024]
        N, T_out3, M, K,  # int32: T_out3 final time, M=3840, K=1024
        x_stride_n, x_stride_t, x_stride_m,
        w_stride_m, w_stride_k,
        y_stride_n, y_stride_t, y_stride_k,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        n = tl.program_id(0)
        t_block = tl.program_id(1)
        k_block = tl.program_id(2)

        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)  # BLOCK_T can be 1; here we use t_offsets
        k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        mask_t = t_offsets < T_out3
        mask_k = k_offsets < K

        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

        # Loop over M dimension (columns of X)
        for m0 in range(0, M, BLOCK_M):
            m_offsets = m0 + tl.arange(0, BLOCK_M)
            mask_m = m_offsets < M

            # Load X[n, t_offsets, m_offsets] -> [BLOCK_T, BLOCK_M]
            x_idx = n * x_stride_n + t_offsets[:, None] * x_stride_t + m_offsets[None, :] * x_stride_m
            x_val = tl.load(x_ptr + x_idx, mask=mask_t[:, None] & mask_m[None, :], other=0.0).to(tl.float32)

            # Load W[m_offsets, k_offsets] -> [BLOCK_M, BLOCK_K]
            w_idx = m_offsets[:, None] * w_stride_m + k_offsets[None, :] * w_stride_k
            w_val = tl.load(w_ptr + w_idx, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

            # Accumulate: acc += sum_m (x[:, m] * w[m, :])
            # x_val shape [T, M_block], w_val shape [M_block, K_block]
            # For each t, reduce over M_block
            # Use loop over M_block
            for mm in range(BLOCK_M):
                # Guard if m_offsets[mm] is valid
                if tl.any(mask_m[mm]):
                    x_col = x_val[:, mm]  # [BLOCK_T]
                    w_row = w_val[mm, :]  # [BLOCK_K]
                    acc += x_col * w_row  # elementwise multiply, then reduce over t would be sum, but here we accumulate per k
        # Note: The above simple loop is correct for small M; Triton allows loops. For robustness, we vectorize over m:
        # Implement a more vectorized approach using tl.dot:
        # We need X as [BLOCK_T, BLOCK_M] and W as [BLOCK_M, BLOCK_K] then acc += tl.dot(X, W)
        # However, tl.dot requires 2D inputs; so we loop over M_block and accumulate:
        # Replace the previous per-element loop with tl.dot:
        # Loop over M in chunks and accumulate
        for m0 in range(0, M, BLOCK_M):
            m_offsets = m0 + tl.arange(0, BLOCK_M)
            mask_m = m_offsets < M

            x_idx = n * x_stride_n + t_offsets[:, None] * x_stride_t + m_offsets[None, :] * x_stride_m
            x_val = tl.load(x_ptr + x_idx, mask=mask_t[:, None] & mask_m[None, :], other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_M]

            w_idx = m_offsets[:, None] * w_stride_m + k_offsets[None, :] * w_stride_k
            w_val = tl.load(w_ptr + w_idx, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

            # For each t, acc[k] += sum_m x[n, t, m] * w[m, k]
            # Since w_val is [M, K], we can compute contribution per k: sum_m x[:, m] * w[m, k]
            for mm in range(BLOCK_M):
                # Load corresponding columns where m_offsets[mm] is valid
                if mask_m[mm]:
                    x_col = x_val[:, mm]  # [BLOCK_T]
                    w_col = w_val[mm, :]  # [BLOCK_K]
                    acc += x_col * w_col  # elementwise multiply; acc is 1D over K

        # Store y[n, t_offsets, k_offsets]
        y_idx = n * y_stride_n + t_offsets[:, None] * y_stride_t + k_offsets[None, :] * y_stride_k
        store_mask = mask_t[:, None] & mask_k[None, :]
        tl.store(y_ptr + y_idx, acc[None, :], mask=store_mask)

    @triton.jit
    def scale_embed_kernel(
        y_ptr,            # *f32: [N, T_out3, K]
        scale,            # float32
        N, T_out3, K,
        y_stride_n, y_stride_t, y_stride_k,
        BLOCK_K: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        n = tl.program_id(0)
        t_block = tl.program_id(1)
        k_block = tl.program_id(2)

        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        mask_t = t_offsets < T_out3
        mask_k = k_offsets < K

        # Load y[n, t, k], multiply by scale, store back
        # Iterate over t and k tiles
        for t0 in range(0, T_out3, BLOCK_T):
            for k0 in range(0, K, BLOCK_K):
                # Compute indices
                t_vec = t0 + tl.arange(0, BLOCK_T)
                k_vec = k0 + tl.arange(0, BLOCK_K)
                mask_t_vec = t_vec < T_out3
                mask_k_vec = k_vec < K
                idx = n * y_stride_n + t_vec[:, None] * y_stride_t + k_vec[None, :] * y_stride_k
                mask = mask_t_vec[:, None] & mask_k_vec[None, :]
                y_val = tl.load(y_ptr + idx, mask=mask, other=0.0).to(tl.float32)
                y_val = y_val * scale
                tl.store(y_ptr + idx, y_val, mask=mask)

    @triton.jit
    def add_pos_emb_kernel(
        y_ptr,            # *f32: [N, T_out3, K]
        pos_ptr,          # *f32: [T_out3, K]
        N, T_out3, K,
        y_stride_n, y_stride_t, y_stride_k,
        pos_stride_t, pos_stride_k,
        BLOCK_K: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        n = tl.program_id(0)
        t_block = tl.program_id(1)
        k_block = tl.program_id(2)

        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        mask_t = t_offsets < T_out3
        mask_k = k_offsets < K

        for t0 in range(0, T_out3, BLOCK_T):
            for k0 in range(0, K, BLOCK_K):
                t_vec = t0 + tl.arange(0, BLOCK_T)
                k_vec = k0 + tl.arange(0, BLOCK_K)
                mask_t_vec = t_vec < T_out3
                mask_k_vec = k_vec < K
                y_idx = n * y_stride_n + t_vec[:, None] * y_stride_t + k_vec[None, :] * y_stride_k
                mask = mask_t_vec[:, None] & mask_k_vec[None, :]
                y_val = tl.load(y_ptr + y_idx, mask=mask, other=0.0).to(tl.float32)
                pos_idx = t_vec[:, None] * pos_stride_t + k_vec[None, :] * pos_stride_k
                pos_val = tl.load(pos_ptr + pos_idx, mask=mask, other=0.0).to(tl.float32)
                y_val = y_val + pos_val
                tl.store(y_ptr + y_idx, y_val, mask=mask)


# -------- ModelNew forward (TRITON-ONLY) --------

class ModelNew(nn.Module):
    def forward(self, *args):
        # args: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        # We assume all are tensors on CUDA device and dtype bfloat16; ModelNew.forward does not use torch ops for heavy computation.

        # Extract inputs
        input_features = args[0]  # [N, 1, 80, T] bfloat16
        conv2d1_weight = args[1]  # [384, 1, 3, 3]
        conv2d1_bias = args[2]    # [384]
        conv2d2_weight = args[3]  # [384, 384, 3, 3]
        conv2d2_bias = args[4]    # [384]
        conv2d3_weight = args[5]  # [384, 384, 3, 3]
        conv2d3_bias = args[6]    # [384]
        conv_out_weight = args[7] # [1024, 3840] (K=1024, M=3840), bfloat16
        positional_embedding = args[8]  # [max_source_positions=1500, 1024], bfloat16
        embed_scale = args[9]    # float32 scalar

        # Ensure device and dtype, contiguous
        device = input_features.device
        dtype = input_features.dtype  # bfloat16

        # Make sure all tensors are on the same device and dtype for Triton (compute in float32, store back as bfloat16 if needed)
        # Cast weights to float32 for computation
        x = input_features.contiguous().to(torch.float32)
        w1 = conv2d1_weight.contiguous().to(torch.float32)
        b1 = conv2d1_bias.contiguous().to(torch.float32)
        w2 = conv2d2_weight.contiguous().to(torch.float32)
        b2 = conv2d2_bias.contiguous().to(torch.float32)
        w3 = conv2d3_weight.contiguous().to(torch.float32)
        b3 = conv2d3_bias.contiguous().to(torch.float32)
        w_proj = conv_out_weight.contiguous().to(torch.float32)  # [K, M] = [1024, 3840]
        pos_emb = positional_embedding.contiguous().to(torch.float32)  # [T_max=1500, 1024]

        N, Ci, In, T = x.shape  # N=batch_size, Ci=1, In=80, T=time_dim
        Co = 384
        Out1 = In  # 80
        T_out1 = (T - 3) // 2 + 1

        # Allocate output for conv1
        y1 = torch.empty((N, Co, Out1, T_out1), device=device, dtype=torch.float32)

        # Launch conv1 kernel specialized for Ci=1
        BLOCK_CO = 64
        BLOCK_OUT = 32
        BLOCK_T = 64
        grid1 = (N, triton.cdiv(Co, BLOCK_CO), triton.cdiv(Out1, BLOCK_OUT), triton.cdiv(T_out1, BLOCK_T))
        conv_ci1_stride2_bias_gelu_kernel[grid1](
            x, w1, b1, y1,
            N, In, Out1, T, T_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            T_out1,
            BLOCK_CO=BLOCK_CO, BLOCK_OUT=BLOCK_OUT, BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2,
        )

        # conv2: [N, 384, 80, T_out1] -> [N, 384, 40, T_out2]
        Out2 = Out1 // 2  # 40
        T_out2 = (T_out1 - 3) // 2 + 1

        y2 = torch.empty((N, Co, Out2, T_out2), device=device, dtype=torch.float32)

        BLOCK_CO2 = 64
        BLOCK_IN2 = 64
        BLOCK_OUT2 = 32
        BLOCK_T2 = 64
        grid2 = (N, triton.cdiv(Co, BLOCK_CO2), triton.cdiv(Out2, BLOCK_OUT2), triton.cdiv(T_out2, BLOCK_T2))
        conv_generic_stride2_bias_gelu_kernel[grid2](
            y1, w2, b2, y2,
            N, Co, Out1, Out2, T_out1, T_out2,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            T_out2,
            BLOCK_CO=BLOCK_CO2, BLOCK_IN=BLOCK_IN2, BLOCK_OUT=BLOCK_OUT2, BLOCK_T=BLOCK_T2,
            num_warps=4, num_stages=2,
        )

        # conv3: [N, 384, 40, T_out2] -> [N, 384, 20, T_out3]
        Out3 = Out2 // 2  # 20
        T_out3 = (T_out2 - 3) // 2 + 1

        y3 = torch.empty((N, Co, Out3, T_out3), device=device, dtype=torch.float32)

        BLOCK_CO3 = 64
        BLOCK_IN3 = 64
        BLOCK_OUT3 = 32
        BLOCK_T3 = 32
        grid3 = (N, triton.cdiv(Co, BLOCK_CO3), triton.cdiv(Out3, BLOCK_OUT3), triton.cdiv(T_out3, BLOCK_T3))
        conv_generic_stride2_bias_gelu_kernel[grid3](
            y2, w3, b3, y3,
            N, Co, Out2, Out3, T_out2, T_out3,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            T_out3,
            BLOCK_CO=BLOCK_CO3, BLOCK_IN=BLOCK_IN3, BLOCK_OUT=BLOCK_OUT3, BLOCK_T=BLOCK_T3,
            num_warps=4, num_stages=2,
        )

        # Reshape y3 to [N, T_out3, Co*Out3] = [N, T_out3, 384*20]
        X_for_linear = y3.permute(0, 3, 1, 2).contiguous().view(N, T_out3, Co * Out3)

        # Linear projection: y = X @ W^T where W^T = [M=3840, K=1024]
        # Transpose conv_out_weight to [M, K] (no heavy compute here)
        Wt = w_proj.permute(1, 0).contiguous()  # [M, K]
        Y = torch.empty((N, T_out3, Wt.shape[1]), device=device, dtype=torch.float32)

        BLOCK_K = 128
        BLOCK_M = 256
        BLOCK_T = 64
        grid_lin = (N, triton.cdiv(T_out3, BLOCK_T), triton.cdiv(Wt.shape[1], BLOCK_K))
        linear_bmm_kernel[grid_lin](
            X_for_linear, Wt, Y,
            N, T_out3, Wt.shape[0], Wt.shape[1],
            X_for_linear.stride(0), X_for_linear.stride(1), X_for_linear.stride(2),
            Wt.stride(0), Wt.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2,
        )

        # Scale by embed_scale (float32)
        scale = float(embed_scale)
        scaled = torch.empty_like(Y, dtype=torch.float32, device=device)
        grid_scale = (N, triton.cdiv(T_out3, BLOCK_T), triton.cdiv(Wt.shape[1], BLOCK_K))
        scale_embed_kernel[grid_scale](
            scaled, scale,
            N, T_out3, Wt.shape[1],
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2,
        )

        # Add positional embedding: pos_emb is [T_max, K], we only need first T_out3 rows
        pos_sel = pos_emb[:T_out3, :].contiguous()
        out_with_pos = torch.empty_like(Y, dtype=torch.float32, device=device)
        grid_pos = (N, triton.cdiv(T_out3, BLOCK_T), triton.cdiv(Wt.shape[1], BLOCK_K))
        add_pos_emb_kernel[grid_pos](
            out_with_pos, pos_sel,
            N, T_out3, Wt.shape[1],
            Y.stride(0), Y.stride(1), Y.stride(2),
            pos_sel.stride(0), pos_sel.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2,
        )

        # Return result (dtype bfloat16 as in original)
        return scaled.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
