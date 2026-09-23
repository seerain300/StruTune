# L2 终评前守卫审计（2026-09-21 18:4x）

口径：对每题退出保存解按协议 BANNED 清单静态扫描
（torch.matmul/mm/bmm/addmm/einsum, F.linear, F.conv*, torch.fft.*,
torch.cumsum/sort/topk/unique, @torch.compile）。
类型注解/no_grad/张量分配视为良性；任一 banned 调用即判违规（协议：REJECTED）。

## 结论：10 题中 3 干净、7 违规

| 题 | 判定 | @triton.jit | 违规内容 |
|---|---|---|---|
| 030 | ✅ TRITON | 1 | — |
| 040 | ✅ TRITON | 5 | — |
| 049 | ✅ TRITON | 2 | — |
| 012 | ❌ | 4 | 3× torch.bmm |
| 015 | ❌ | 2 | 5× F.conv2d |
| 036 | ❌ | 1 | @torch.compile + 2× torch.mm |
| 043 | ❌ | 0 | torch.compile + cumsum + bmm（纯 torch 委托） |
| 051 | ❌ | 0 | conv2d/linear/conv1d（纯 torch 委托） |
| 057 | ❌ | 1 | 3× F.conv1d |
| 080 | ❌ | 3 | 4× torch.mm |

## 处置（用户裁定 2026-09-21）
- 干净 3 题跑最终评测（全量 wl、100 iter、无缓存）
- 违规 7 题不跑终评，保留搜索期记录 + 本审计作为方法学发现：
  守卫（提示词级）对 conv/FFT/批量矩阵类失效，模型转向 torch 委托
  cuDNN/cuBLAS；与复现手册坑表 #8 的预判一致（"conv/FFT 题的守卫无效
  是 LLM 能力边界"）。012/080/043 为混合解（含 Triton kernel 但关键
  步骤委托 torch），说明守卫在非 conv 题上也有泄漏。
- FI/L1 收官后同样审计再做终评选解。
