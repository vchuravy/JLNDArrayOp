#include "jlrtc.h"

namespace mxnet {

JLRtc::JLRtc(const std::string& name,
          std::vector<NDArray> const& args,
          char* ptx,
          unsigned int grid_dim_X,
          unsigned int grid_dim_Y,
          unsigned int grid_dim_Z,
          unsigned int block_dim_X,
          unsigned int block_dim_Y,
          unsigned int block_dim_Z) {
  name_ = name;
  ptx_ = ptx;

  for (auto& ndarray: args) {
    dtypes_.push_back(ndarray.dtype());
    shapes_.push_back(ndarray.shape());
  }

}

void JLRtc::verify(std::vector<NDArray> const& ndargs) {
  CHECK_EQ(dtypes_.size(), ndargs.size());
  CHECK(ndargs.size());

  for (size_t i = 0; i < ndargs.size(); ++i) {
    CHECK_EQ(dtypes_[i], ndargs[i].dtype());
    CHECK_EQ(shapes_[i], ndargs[i].shape());
  }
}

CUfunction JLRtc::getFunc(int dev_id) {
  cudaError_enum err;
  CUfunction func;
  if (func_.find(dev_id) != func_.end()) {
        func = func_[dev_id];
  } else {
    CUmodule module;
    CHECK_EQ(err = cuModuleLoadDataEx(&module, ptx_, 0, 0, 0), CUDA_SUCCESS)
      << "CudaError: " << err;
    CHECK_EQ(err = cuModuleGetFunction(&func, module, name_.c_str()), CUDA_SUCCESS)
      << "CudaError: " << err;
    module_[dev_id] = module;
    func_[dev_id] = func;
  }

  return func;
}

void JLRtc::launch(CUfunction func, std::vector<NDArray> const& ndargs, RunContext& rctx) {
    std::vector<CUdeviceptr> args = convert(ndargs);

    cudaError_t cuerr;
    cudaError_enum err;

    CHECK_EQ(err = cuLaunchKernel(func,
                            grid_dim_X_, grid_dim_Y_, grid_dim_Z_,
                            block_dim_X_, block_dim_Y_, block_dim_Z_,
                            0, rctx.get_stream<mshadow::gpu>()->stream_,
                            (void**) args.data(), 0), CUDA_SUCCESS) << "CudaError: " << err;

    CHECK_EQ(cuerr = cudaStreamSynchronize(rctx.get_stream<mshadow::gpu>()->stream_),
             cudaSuccess) << "CudaError: " << cuerr;
    // To prevent memory leaks we free the Julia Arrays that are wrappers to the NDArrays
    for (auto dptr: args) {
      CHECK_EQ(err = cuMemFree(dptr), CUDA_SUCCESS) << "CudaError: " << err;
    }
}

}
