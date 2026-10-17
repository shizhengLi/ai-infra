# 实验 064：FP8 KV native pair conversion

日期：2026-10-17  
状态：完成；性能候选通过局部 gate，数值 bit-exact gate 不通过，默认关闭

## 目标

实验 063 的 fused kernel 已经把缩放、clamp、FP8 转换和 page store 合并，但转换路径仍
先把四个元素展开成 `float4`。本实验在 L20 SM89 上使用 CUDA 12.8 的原生 FP8 转换接口，
以 `__half2`/`__nv_bfloat162` 为单位做缩放，再用 `__nv_fp8x4_e4m3` 打包写入，验证是否
能进一步降低 FP8 KV 写入开销。

实现为默认关闭的 `--kv-cache-native-pair`，且要求同时启用
`--kv-cache-fused-store`。legacy cast/store 和实验 063 的 float fused 路径均保持不变。

## 原理和实现

- 每个 warp 处理一个 KV token row，每个 lane 处理 4 个连续元素。
- BF16 使用 `__nv_bfloat162`，FP16 使用 `__half2`；scale 转成对应的 pair 类型后调用
  `__hmul2`。
- 每两个 pair 交给 `__nv_fp8x4_e4m3`，利用 SM89 的 FP8 转换指令完成 E4M3 饱和转换和
  32-bit packed store。
- `native_pair=False` 仍走实验 063 的 float4 + 显式 clamp 路径。
- JIT 编译必须使用 `/usr/local/cuda-12.8/bin/nvcc`。当前 `/usr/bin/nvcc` 是 CUDA 11.5，
  会在 FlashInfer 的 `--compress-mode=size` 或本 kernel 的 C++20 编译上失败；本实验没有
  修改全局工具链配置。

## 正确性结果

GPU：L20 GPU 2，row width 1024，K/V 各 1024 rows。

- `scale=0.125` 和 `scale=0.03125`（可被 BF16/FP16 精确表示）时，BF16/FP16 native
  输出与 float fused 的 FP8 bytes 均完全一致。
- `scale=0.12345` 时，native 与 float fused 的 K 输出 byte mismatch 为 BF16
  38,328/1,048,576（3.65%），FP16 2,491/1,048,576（0.24%）。原因是 native 路径先把
  scale 舍入到 BF16/FP16，再做 pair multiply；float fused 保留 FP32 scale。
- 一次服务 smoke 的 native/float 输出 hash 相同，但这不是 held-out 质量评测，不能替代
  byte-level 和长上下文质量 gate。

因此 native pair 不能宣称为 bit-exact drop-in replacement，必须保持显式 opt-in；若要默认
启用，需要改为保留 FP32 scale 的 pair conversion，或增加 scale 可表示性/误差门禁。

## CUDA event 微基准

L20 GPU 2，BF16 输入，row width 1024；每个配置预热 30 次，随后 7 轮、每轮 200 次，
报告中位数。测量包含 Python/JIT wrapper 的 kernel launch，但两条路径完全一致。

| Rows | float fused ms | native pair ms | native 相对变化 |
| ---: | ---: | ---: | ---: |
| 256 | 0.03456 | 0.01886 | -45.42% |
| 1,024 | 0.01410 | 0.01133 | -19.65% |
| 4,096 | 0.01132 | 0.01134 | +0.18% |
| 16,384 | 0.04176 | 0.03863 | -7.52% |

预注册的局部目标是 row 数 256/1024/4096/16384 中至少三个配置改善 20%。本轮严格按
中位数只有 256 达到 20%，因此“普适 kernel 加速” gate 不通过；结果显示收益依赖于
row 数、occupancy 和 launch/内存行为，不能只用 1K rows 的结果推广。

## Qwen3-32B TP4 端到端

配置：L20 GPU 2/3/6/7，FP8 W8A8 权重，FP8 E4M3 KV，`first_batch` calibration，
FlashInfer，CUDA Graph disabled，memory ratio 0.8，输出 32 tokens，并发 1，重复 2 次。
native 与 float fused 的唯一变量是 `--kv-cache-native-pair`。

| Input | float fused tok/s | native tok/s | 吞吐变化 | float TPOT ms | native TPOT ms | TPOT 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 13.36 | 13.78 | +3.15% | 41.85 | 39.55 | -5.49% |
| 8,192 | N/A | 12.70 | N/A | N/A | 42.32 | N/A |
| 16,384 | N/A | 7.90 | N/A | N/A | 43.60 | N/A |

8K/16K 的同 commit float fused 对照未在本轮重复，因此不把 native 的绝对值与实验 063
结果直接做因果比较。4K 配对结果只说明在本次机器状态下有小幅改善，仍需更多重复和并发
矩阵确认。

原始结果：

- [native 4096](../results/064_native_4096.md)
- [native 8192](../results/064_native_8192.md)
- [native 16384](../results/064_native_16384.md)
- [float fused 4096](../results/064_float_4096.md)

## 结论

1. CUDA 原生 pair conversion 在 L20 上可用，BF16/FP16 smoke 和服务启动均通过。
2. 在 scale 可精确表示时输出 byte 一致；真实 calibration scale 下 BF16 存在可观 byte 差异，
   这是当前实现的主要风险。
3. 局部 kernel 延迟在部分 row 数下降，但没有达到跨四个规模的 20% 普适 gate；4K 端到端
   仅观察到 +3.15% 吞吐和 -5.49% TPOT，证据不足以切默认。
4. 保持两个开关默认关闭：`--kv-cache-fused-store` 和 `--kv-cache-native-pair`。后续优先
   研究“保留 FP32 scale 的 pair conversion”或把 native 路径限制到可表示 scale，再进入
   page size >1/Graph shape 实验。
