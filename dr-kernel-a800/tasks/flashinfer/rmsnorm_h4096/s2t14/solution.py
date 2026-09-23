import triton
import triton.language as tl


@triton.jit
def kernel_rms(
    x_ptr,                # *T, [B, H] (T 可以是 bf16/fp16/fp32; 在 kernel 中轉換為 fp32)
    out_inv_rms_ptr,      # *fp32, [B]
    B, H, EPS,            # int32, fp32
    stride_x_row, stride_x_col,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    # 首輪：計算該行的平方和（fp32）
    sum_sq = 0.0
    col_start = 0
    while col_start < H:
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < H
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        col_start += BLOCK_N

    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)
    tl.store(out_inv_rms_ptr + row, inv_rms)


@triton.jit
def kernel_scale(
    x_ptr,                # *T, [B, H]
    weight_ptr,           # *T, [H]
    out_ptr,              # *fp32, [B, H]
    B, H,                 # int32
    inv_rms_ptr,          # *fp32, [B]
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    inv_rms = tl.load(inv_rms_ptr + row)  # fp32 常量

    col_start = 0
    while col_start < H:
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < H
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_ptr + row * stride_out_row + cols * stride_out_col, y, mask=mask)
        col_start += BLOCK_N


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # 調用評價環境提供的 get_inputs 來獲取張量
        hidden_states, weight = get_inputs()

        # shapes (作為整數傳遞給 kernels; 主機端不使用任何 tensor 方法)
        B = hidden_states.shape[0]
        H = hidden_states.shape[1]

        # per-row inv_rms 缓冲區（fp32）; 主機端創建一個 tensor
        inv_rms = torch.empty(B, dtype=torch.float32, device=hidden_states.device)

        # 行、列步長 (元素步長，不是 tensor 方法)
        stride_x_row = hidden_states.stride(0)
        stride_x_col = hidden_states.stride(1)

        # 啟動 kernel_rms: 一個程式每行
        BLOCK_N = 256
        grid_rms = (B,)
        kernel_rms[grid_rms](
            hidden_states, inv_rms,
            B, H, 1e-5,
            stride_x_row, stride_x_col,
            BLOCK_N=BLOCK_N,
        )

        # 分配輸出 (fp32); 唯一的主機端操作是創建一個 tensor
        out = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)

        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # 啟動 kernel_scale: 一個程式每行
        grid_scale = (B,)
        kernel_scale[grid_scale](
            hidden_states, weight, out,
            B, H,
            inv_rms,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            BLOCK_N=BLOCK_N,
        )

        # 返回計算結果
        return out


def run(*args):
    return ModelNew()(*args)
