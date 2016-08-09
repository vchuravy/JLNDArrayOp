#include "device_array.h"

namespace mxnet {

template <typename T, size_t N>
std::vector<CUdeviceptr> convert(std::vector<NDArray> ndarrays) {
  std::vector<CUdeviceptr> cuarrays;
  for (NDArray ndarray : ndarrays) {
    // convert from a NDArray to Julia
    // We have to revert the order of shape to get 
    // The expected Julia behaviour.
    const TShape shape = ndarray.shape();
    CHECK_EQ(shape.ndim(), N);

    DeviceArray<T, N> cuarray;
    cuarray.ptr = static_cast<T*>(ndarray.data().dptr_);
    cuarray.len = shape.ProdShape(0, N);
    // Switch to Julia convention
    for (size_t i = 0; i < N; ++i) {
      cuarray.shape[i] = shape[(N - 1) - i ];
    }

    // cuMemAlloc
    CUdeviceptr dptr;
    cuMemAlloc(&dptr, sizeof(cuarray));

    // cuMemcpyHtoD
    cuMemcpyHtoD(dptr, &cuarray, sizeof(cuarray));
    cuarrays.push_back(dptr);
  }
  return cuarrays;
}

}
