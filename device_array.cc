#include "device_array.h"

namespace mxnet {

std::vector<CUdeviceptr> convert(const std::vector<NDArray>& ndarrays) {
  std::vector<CUdeviceptr> cuarrays;
  for (NDArray ndarray : ndarrays) {
    int dtype = ndarray.dtype();
    CUdeviceptr dptr;
    MSHADOW_TYPE_SWITCH(dtype, DType, {
      switch(ndarray.shape().ndim()) {
        case 1:
          dptr = convert<DType, 1>(ndarray);
          break;
        case 2:
          dptr = convert<DType, 2>(ndarray);
          break;
        case 3:
          dptr = convert<DType, 3>(ndarray);
          break;
        case 4:
          dptr = convert<DType, 4>(ndarray);
          break;
        case 5:
          dptr = convert<DType, 5>(ndarray);
          break;
        default:
          LOG(FATAL) << "Can't convert an NDArray with ndim > 5";
      }
    });
    cuarrays.push_back(dptr);
  }
  return cuarrays;
}

template <typename T, size_t N>
CUdeviceptr convert(const NDArray& ndarray) {
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
  return dptr;
}

}
