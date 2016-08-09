#ifndef DEVICE_ARRAY_H
#define DEVICE_ARRAY_H

#include <cstdint>
#include <vector>
#include <mxnet/ndarray.h>
#include <cuda.h>

#if defined(__LP64__)
typedef int64_t jint;
#else
typedef int32_t jint;
#endif

// Keep in sync with the definition at CUDAnative.jl src/array.jl
template <typename T, size_t N>
struct DeviceArray {
  T* ptr;
  jint shape[N];
  jint len;
};
namespace mxnet {

template <typename T, size_t N>
CUdeviceptr convert(const NDArray& ndarray);

std::vector<CUdeviceptr> convert(const std::vector<NDArray>& ndarrays);

}
#endif // DEVICE_ARRAY_H
