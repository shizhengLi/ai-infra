#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

struct StoreFP8KernelParams {
  void *__restrict__ k_cache;
  void *__restrict__ v_cache;
  const void *__restrict__ indices;
  const void *__restrict__ k;
  const void *__restrict__ v;
  const float *__restrict__ scale;
  std::size_t kv_cache_stride;
  std::size_t kv_input_stride;
  std::size_t length;
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, typename InputT, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    store_kv_cache_fp8(const __grid_constant__ StoreFP8KernelParams params) {
  using namespace device;

  constexpr auto kWarpPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);

  const auto &[k_cache, v_cache, indices, k, v, scale, kv_cache_stride,
               kv_input_stride, length] = params;
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;
  PDL::wait<kUsePDL>();

  if (warp_id < length) {
    const auto lane_id = threadIdx.x % kWarpThreads;
    const auto pos = static_cast<const T *>(indices)[warp_id];
    auto *dst_k = reinterpret_cast<__nv_fp8_e4m3 *>(
        pointer::offset(k_cache, pos * kv_cache_stride));
    auto *dst_v = reinterpret_cast<__nv_fp8_e4m3 *>(
        pointer::offset(v_cache, pos * kv_cache_stride));
    const auto *src_k = reinterpret_cast<const InputT *>(
        pointer::offset(k, warp_id * kv_input_stride));
    const auto *src_v = reinterpret_cast<const InputT *>(
        pointer::offset(v, warp_id * kv_input_stride));
    const float inv_scale = 1.0f / *scale;

    for (std::size_t i = lane_id; i < kElementSize; i += kWarpThreads) {
      float k_value = static_cast<float>(src_k[i]) * inv_scale;
      float v_value = static_cast<float>(src_v[i]) * inv_scale;
      k_value = fminf(fmaxf(k_value, -448.0f), 448.0f);
      v_value = fminf(fmaxf(v_value, -448.0f), 448.0f);
      dst_k[i] = __nv_fp8_e4m3(k_value);
      dst_v[i] = __nv_fp8_e4m3(v_value);
    }
  }

  PDL::launch<kUsePDL>();
}

template <std::size_t kElementSize, bool kInputIsBFloat16,
          std::size_t kNumThreads = 128, std::size_t kMaxOccupancy = 1,
          bool kUsePDL = false>
struct StoreFP8Kernel {
  static void run(const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView k, const tvm::ffi::TensorView v,
                  const tvm::ffi::TensorView scale) {
    using namespace host;
    auto D = SymbolicSize{"D"};
    auto L = SymbolicSize{"L"};
    auto X = SymbolicSize{"X"};
    auto Y = SymbolicSize{"Y"};
    auto indices_dtype_ = SymbolicDType{};
    auto cache_dtype_ = SymbolicDType{};
    auto input_dtype_ = SymbolicDType{};
    auto scale_dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({-1, D})
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(cache_dtype_)
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({L, D})
        .with_strides({Y, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(input_dtype_)
        .verify(k)
        .verify(v);
    TensorMatcher({L})
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);
    TensorMatcher({1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<float>(scale_dtype_)
        .verify(scale);

    const auto cache_dtype = cache_dtype_.unwrap();
    const auto input_dtype = input_dtype_.unwrap();
    RuntimeCheck(cache_dtype.bits == 8 && cache_dtype.lanes == 1,
                 "FP8 store requires a one-byte cache output");
    if constexpr (kInputIsBFloat16) {
      RuntimeCheck(input_dtype.code == DLDataTypeCode::kDLBfloat &&
                       input_dtype.bits == 16,
                   "FP8 store template expects bfloat16 input");
    } else {
      RuntimeCheck(input_dtype.code == DLDataTypeCode::kDLFloat &&
                       input_dtype.bits == 16,
                   "FP8 store template expects float16 input");
    }
    RuntimeCheck(element_size(k_cache) == kElementSize,
                 "FP8 store element size mismatch");

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto length = static_cast<std::size_t>(L.unwrap());
    const auto kv_cache_stride = X.unwrap();
    const auto kv_input_stride = Y.unwrap() * input_dtype.bits / 8;
    const auto params = StoreFP8KernelParams{
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .scale = static_cast<const float *>(scale.data_ptr()),
        .kv_cache_stride = static_cast<std::size_t>(kv_cache_stride),
        .kv_input_stride = static_cast<std::size_t>(kv_input_stride),
        .length = length,
    };

    constexpr auto kWarpPerBlock = kNumThreads / 32;
    static_assert(kNumThreads % 32 == 0);
    const auto num_blocks = div_ceil(length, kWarpPerBlock);
    const auto kernel = use_int32
                            ? store_kv_cache_fp8<kNumThreads, kMaxOccupancy,
                                                 kUsePDL, kElementSize,
                                                 std::conditional_t<
                                                     kInputIsBFloat16,
                                                     __nv_bfloat16, __half>,
                                                 int32_t>
                            : store_kv_cache_fp8<kNumThreads, kMaxOccupancy,
                                                 kUsePDL, kElementSize,
                                                 std::conditional_t<
                                                     kInputIsBFloat16,
                                                     __nv_bfloat16, __half>,
                                                 int64_t>;
    LaunchKernel(num_blocks, kNumThreads, device)
        .with_attr(kUsePDL)(kernel, params);
  }

private:
  static auto element_size(const tvm::ffi::TensorView tensor) -> std::size_t {
    return static_cast<std::size_t>(tensor.size(1));
  }
};

} // namespace
