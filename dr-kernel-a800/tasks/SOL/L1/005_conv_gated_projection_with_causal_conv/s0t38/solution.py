import torch
import triton
import triton.language as tl


@triton.jit
def randn_fill_kernel(out_ptr, rows, cols, seed, BLOCK: tl.constexpr):
    """
    Triton RNG fill kernel: fill a 2D tensor [rows, cols] with random float32.
    Each program writes a BLOCK x BLOCK tile. Seed used for XOR-based random numbers.
    """
    pid = tl.program_id(axis=0)
    start_row = (pid // 1024) * BLOCK
    start_col = (pid % 1024) * BLOCK
    # Triton requires integer vectors for indexing
    rows_vec = start_row + tl.arange(0, BLOCK)
    cols_vec = start_col + tl.arange(0, BLOCK)
    # 2D indexing via broadcasting
    # tl.load/tl.store support broadcasting, but here we form pointers per element.
    for i in range(BLOCK):
        for j in range(BLOCK):
            row = start_row + i
            col = start_col + j
            # mask for boundary
            if row < rows and col < cols:
                idx = row * cols + col
                # Simple RNG using seed: rand = ((seed * 1103515245 + 12345) & 0x7fffffff) / 2**31
                seed_next = seed * 1103515245 + 12345
                rand = tl.float32((seed_next & 0x7fffffff) / (1 << 31))
                tl.store(out_ptr + idx, rand)
    # Note: The outer grid is set to rows * cols / BLOCK in the host, so we don't need
    # a return value; the loop writes all elements in the grid tile.


@triton.jit
def in_proj_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    BLOCK_M: tl.constexpr):
    """
    Compute BCx[b, s, m] = sum_h x[b, s, h] * w[m, h] + b[m], where m in [0, 3H).
    x: [B, S, H], w: [3H, H], b: [3H], out: [B, S, 3H].
    Grid: (B, S, cdiv(3H, BLOCK_M))
    """
    b_idx = tl.program_id(axis=0)
    s_idx = tl.program_id(axis=1)
    m_block = tl.program_id(axis=2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < 3 * H

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Accumulate over H
    for h in range(0, H):
        x_val = tl.load(x_ptr + b_idx * S * H + s_idx * H + h)
        # Load w[m, h] for the tile
        w_vals = tl.load(w_ptr + m_offsets * H + h, mask=mask_m, other=0.0)
        acc += x_val * w_vals

    # Add bias
    b_vals = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += b_vals

    # Store
    tl.store(out_ptr + b_idx * S * (3 * H) + s_idx * (3 * H) + m_offsets, acc, mask=mask_m)


@triton.jit
def pad_left_kernel(inp_ptr, out_ptr, B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, pad_left: tl.constexpr):
    """
    Left-pad inp (B, S, H) to out (B, S + pad_left, H) along sequence dim.
    For each (b, h, t):
      out[b, h, t] = inp[b, h, t - pad_left] if t >= pad_left else 0.
    Grid: (B, S + pad_left, H)
    """
    b_idx = tl.program_id(axis=0)
    t = tl.program_id(axis=1)
    h = tl.program_id(axis=2)

    # Compute source index; guard for pad region
    src_t = t - pad_left
    in_bounds = src_t >= 0
    val = tl.load(inp_ptr + b_idx * S * H + src_t * H + h, mask=in_bounds, other=0.0)
    tl.store(out_ptr + b_idx * (S + pad_left) * H + t * H + h, val, mask=True)


@triton.jit
def conv_groupsH_kernel(inp_ptr, w_ptr, b_ptr, out_ptr,
                         B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, K: tl.constexpr):
    """
    Grouped causal conv: inp (B, H, S) with weight (H, 1, K), bias (H,).
    Output (B, H, S): out[b, c, t] = sum_{k=0..K-1} inp[b, c, t + k] * w[c, 0, k] + b[c]
    Grid: (B, H, S)
    """
    b_idx = tl.program_id(axis=0)
    c = tl.program_id(axis=1)
    t = tl.program_id(axis=2)

    acc = 0.0
    for k in range(0, K):
        src_t = t + k
        # check bounds: src_t < S (since we wrote padded zeros elsewhere if needed)
        in_bounds = src_t < S
        val = tl.load(inp_ptr + b_idx * H * S + c * S + src_t, mask=in_bounds, other=0.0)
        wk = tl.load(w_ptr + c * K + k)
        acc += val * wk

    # add bias
    bc = tl.load(b_ptr + c)
    acc += bc

    # store
    tl.store(out_ptr + b_idx * H * S + c * S + t, acc, mask=True)


@triton.jit
def elementwise_mul_kernel(a_ptr, b_ptr, out_ptr,
                            B: tl.constexpr, S: tl.constexpr, H: tl.constexpr):
    """
    Elementwise multiply: out[b, s, h] = a[b, s, h] * b[b, s, h]
    Grid: (B, S, H)
    """
    b_idx = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    h = tl.program_id(axis=2)
    a_val = tl.load(a_ptr + b_idx * S * H + s * H + h)
    b_val = tl.load(b_ptr + b_idx * S * H + s * H + h)
    tl.store(out_ptr + b_idx * S * H + s * H + h, a_val * b_val)


@triton.jit
def out_proj_kernel(y_ptr, out_w_ptr, out_b_ptr, out_ptr,
                    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr):
    """
    Final linear: output[b, s, h] = sum_{h2} y[b, s, h2] * out_w[h2, h] + out_b[h]
    y: [B, S, H], out_w: [H, H], out_b: [H]
    output: [B, S, H]
    Grid: (B, cdiv(S, BLOCK_S), cdiv(H, BLOCK_H))
    """
    b_idx = tl.program_id(axis=0)
    s_block = tl.program_id(axis=1)
    h_out_block = tl.program_id(axis=2)

    s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    h_out_offsets = h_out_block * BLOCK_H + tl.arange(0, BLOCK_H)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Accumulate over input H dimension (y's channel dim)
    for h2 in range(0, H):
        y_vals = tl.load(y_ptr + b_idx * S * H + s_offsets * H + h2)
        out_w_vals = tl.load(out_w_ptr + h2 * H + h_out_offsets)
        acc += y_vals * out_w_vals

    # Add bias
    out_b_vals = tl.load(out_b_ptr + h_out_offsets)
    acc += out_b_vals

    # Store output[b, s, h]
    tl.store(out_ptr + b_idx * S * H + s_offsets * H + h_out_offsets, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        """
        Triton-only forward. All tensors are float32 and on CUDA device.
        Returns output tensor of shape (B, S, H), dtype float32.
        """
        B, S, H = x.shape
        M = 3 * H  # in_proj output channels

        # 1) Initialize inputs and parameters with Triton RNG kernel (randn_fill)
        device = x.device
        # Allocate outputs for random fill
        x_ptr = torch.empty((B, S, H), device=device, dtype=torch.float32)
        w_in_ptr = torch.empty((M, H), device=device, dtype=torch.float32)
        b_in_ptr = torch.empty((M,), device=device, dtype=torch.float32)
        # in_proj_weight and in_proj_bias: (3H, H) and (3H,)
        in_proj_weight = torch.empty((M, H), device=device, dtype=torch.float32)
        in_proj_bias = torch.empty((M,), device=device, dtype=torch.float32)
        # conv_weight: (H, 1, 4) -> we'll pass (H, 4) to kernel
        conv_weight_2d = torch.empty((H, 4), device=device, dtype=torch.float32)
        conv_bias = torch.empty((H,), device=device, dtype=torch.float32)
        # out_proj_weight: (H, H), out_proj_bias: (H,)
        out_proj_weight = torch.empty((H, H), device=device, dtype=torch.float32)
        out_proj_bias = torch.empty((H,), device=device, dtype=torch.float32)

        # Launch RNG fill for x
        BLOCK = 128
        grid_x = (B * S * H + BLOCK - 1) // BLOCK  # number of tiles
        seed_x = 1234
        randn_fill_kernel[(grid_x,)](x_ptr, B, S * H, seed_x, BLOCK=BLOCK, num_warps=4, num_stages=2)

        # Launch RNG fill for in_proj_weight, in_proj_bias
        grid_wb = (M * H + BLOCK - 1) // BLOCK
        randn_fill_kernel[(grid_wb,)](w_in_ptr, M, H, seed_x, BLOCK=BLOCK, num_warps=4, num_stages=2)
        grid_b = (M + BLOCK - 1) // BLOCK
        randn_fill_kernel[(grid_b,)](b_in_ptr, M, 1, seed_x, BLOCK=BLOCK, num_warps=4, num_stages=2)

        # Assign filled tensors as inputs/params
        in_proj_weight.copy_(w_in_ptr)  # (M, H)
        in_proj_bias.copy_(b_in_ptr)    # (M,)
        # conv_weight_2d: (H, 4)
        conv_weight_2d.copy_(torch.empty((H, 4), device=device, dtype=torch.float32))
        # Fill conv_weight_2d with random as well
        grid_wc = (H * 4 + BLOCK - 1) // BLOCK
        randn_fill_kernel[(grid_wc,)](conv_weight_2d, H, 4, seed_x, BLOCK=BLOCK, num_warps=4, num_stages=2)
        conv_bias.copy_(torch.empty((H,), device=device, dtype=torch.float32))
        grid_cb = (H + BLOCK - 1) // BLOCK
        randn_fill_kernel[(grid_cb,)](conv_bias, H, 1, seed_x, BLOCK=BLOCK, num_warps=4, num_stages=2)

        # out_proj_weight and out_proj_bias
        out_proj_weight.copy_(torch.empty((H, H), device=device, dtype=torch.float32))
        out_proj_bias.copy_(torch.empty((H,), device=device, dtype=torch.float32))
        grid_ow = (H * H + BLOCK - 1) // BLOCK
        randn_fill_kernel[(grid_ow,)](out_proj_weight, H, H, seed_x, BLOCK=BLOCK, num_warps=4, num_stages=2)
        grid_ob = (H + BLOCK - 1) // BLOCK
        randn_fill_kernel[(grid_ob,)](out_proj_bias, H, 1, seed_x, BLOCK=BLOCK, num_warps=4, num_stages=2)

        # 2) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias, output (B, S, 3H)
        BCx = torch.empty((B, S, M), device=device, dtype=torch.float32)
        BLOCK_M = 64
        grid_in = (B, S, triton.cdiv(M, BLOCK_M))
        in_proj_kernel[grid_in](
            x_ptr, in_proj_weight, in_proj_bias, BCx,
            B=B, S=S, H=H, M=M, BLOCK_M=BLOCK_M, num_warps=4, num_stages=2
        )

        # 3) Split BCx into B, C, x_proj along last dim
        # B: (B, S, H), C: (B, S, H), x_proj: (B, S, H)
        # We need to load slices; since BCx is a tensor, we implement slicing via indexing in Triton:
        # Allocate B, C, x_proj as outputs of elementwise kernels below (we'll compute them now)

        # Compute B, C, x_proj from BCx using Triton elementwise indexing (no torch slicing)
        # We can compute B, C, x_proj directly from original x? No, we must use BCx. So we load.
        # But we don't have BCx split tensors here; we can compute by indexing:
        # For each m, where m in [0..H-1], [H..2H-1], [2H..3H-1], we need to materialize B, C, x_proj.
        # Simpler: since we have BCx, we can compute B, C, x_proj by launching elementwise kernels
        # that load BCx[:, :, 0:H], BCx[:, :, H:2H], BCx[:, :, 2H:3H]. But that requires building pointers.
        # Instead, we compute them as follows:
        B_tensor = torch.empty((B, S, H), device=device, dtype=torch.float32)
        C_tensor = torch.empty((B, S, H), device=device, dtype=torch.float32)
        x_proj_tensor = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # We'll fill these using in_proj kernel's output by reading BCx and slicing via elementwise kernel.
        # However, Triton kernels don't support reading from BCx to create B/C/x_proj directly. We can
        # instead compute B, C, x_proj from original x and in_proj_weight for [:H], [H:2H], [2H:3H] using
        # in_proj_kernel for each segment. This is acceptable as it is Triton-only and produces identical math.
        # For simplicity and correctness, we'll compute B, C, x_proj by invoking in_proj_kernel with appropriate
        # weight/bias slices.

        # Compute B: in_proj_kernel with weight[:H, :], bias[:H]
        w_B = in_proj_weight[:H, :].contiguous()
        b_B = in_proj_bias[:H].contiguous()
        B_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_B = (B, S, triton.cdiv(H, BLOCK_M))
        in_proj_kernel[grid_B](
            x_ptr, w_B, b_B, B_out,
            B=B, S=S, H=H, M=H, BLOCK_M=BLOCK_M, num_warps=4, num_stages=2
        )

        # Compute C: in_proj_kernel with weight[H:2H, :], bias[H:2H]
        w_C = in_proj_weight[H:2*H, :].contiguous()
        b_C = in_proj_bias[H:2*H].contiguous()
        C_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_C = (B, S, triton.cdiv(H, BLOCK_M))
        in_proj_kernel[grid_C](
            x_ptr, w_C, b_C, C_out,
            B=B, S=S, H=H, M=H, BLOCK_M=BLOCK_M, num_warps=4, num_stages=2
        )

        # Compute x_proj: in_proj_kernel with weight[2H:3H, :], bias[2H:3H]
        w_xp = in_proj_weight[2*H:3*H, :].contiguous()
        b_xp = in_proj_bias[2*H:3*H].contiguous()
        x_proj_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_xp = (B, S, triton.cdiv(H, BLOCK_M))
        in_proj_kernel[grid_xp](
            x_ptr, w_xp, b_xp, x_proj_out,
            B=B, S=S, H=H, M=H, BLOCK_M=BLOCK_M, num_warps=4, num_stages=2
        )

        # 4) Gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_gate = (B, S, H)
        elementwise_mul_kernel[grid_gate](
            B_out, x_proj_out, Bx,
            B=B, S=S, H=H, num_warps=4, num_stages=2
        )

        # 5) Left-pad Bx by pad_left = K - 1 = 3
        pad_left = 3
        Bx_pad = torch.empty((B, S + pad_left, H), device=device, dtype=torch.float32)
        grid_pad = (B, S + pad_left, H)
        pad_left_kernel[grid_pad](
            Bx, Bx_pad, B=B, S=S, H=H, pad_left=pad_left, num_warps=4, num_stages=2
        )

        # 6) Grouped causal conv: conv_out (B, H, S)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_conv = (B, H, S)
        # conv_weight_2d is (H, 4); conv_bias is (H,)
        conv_groupsH_kernel[grid_conv](
            Bx_pad, conv_weight_2d, conv_bias, conv_out,
            B=B, S=S, H=H, K=4, num_warps=4, num_stages=2
        )

        # 7) Output gating: y = C * conv_out
        y = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_mul = (B, H, S)
        elementwise_mul_kernel[grid_mul](
            C_out, conv_out, y,
            B=B, S=S, H=H, num_warps=4, num_stages=2
        )

        # 8) Transpose to (B, S, H) for final linear
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # 9) Final out_proj: y_T -> output
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)
        BLOCK_S, BLOCK_H = 64, 64
        grid_out = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))
        out_proj_kernel[grid_out](
            y_T, out_proj_weight, out_proj_bias, output,
            B=B, S=S, H=H, BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H, num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
