#include "jlrtc.h"

namespace mxnet {

JLRtc::JLRtc(const std::string& name,
             char* ptx) {

}

void JLRtc::push(std::vector<NDArray> const& input,
                 std::vector<NDArray> const& output,
                 unsigned int grid_dim_X,
                 unsigned int grid_dim_Y,
                 unsigned int grid_dim_Z,
                 unsigned int block_dim_X,
                 unsigned int block_dim_Y,
                 unsigned int block_dim_Z) {
  CHECK_EQ(num_input_, input.size());
  CHECK_EQ(num_output_, output.size());
  CHECK(output.size());

  cudaError_enum err;
  CUfunction func;
  int dev_id = output[0].ctx().dev_id;
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

  auto op = [this, func, input, output,
             grid_dim_X, grid_dim_Y, grid_dim_Z,
             block_dim_X, block_dim_Y, block_dim_Z](RunContext rctx) {
    std::vector<CUdeviceptr> args;
    for (CUdeviceptr dptr: convert(input)) args.push_back(dptr);
    for (CUdeviceptr dptr: convert(output)) args.push_back(dptr);

    cudaError_enum err;
    cudaError_t cuerr;

    CHECK_EQ(err = cuLaunchKernel(func,
                            grid_dim_X, grid_dim_Y, grid_dim_Z,
                            block_dim_X, block_dim_Y, block_dim_Z,
                            0, rctx.get_stream<mshadow::gpu>()->stream_,
                            (void**) args.data(), 0), CUDA_SUCCESS) << "CudaError: " << err;

    CHECK_EQ(cuerr = cudaStreamSynchronize(rctx.get_stream<mshadow::gpu>()->stream_),
             cudaSuccess) << "CudaError: " << cuerr;
    // To prevent memory leaks we free the Julia Arrays that are wrappers to the NDArrays
    for (auto dptr: args) {
      CHECK_EQ(err = cuMemFree(dptr), CUDA_SUCCESS) << "CudaError: " << err;
    }
  };

  std::vector<Engine::VarHandle> var_in, var_out;
  for (auto& i : input) var_in.push_back(i.var());
  for (auto& i : output) var_out.push_back(i.var());
  Engine::Get()->PushSync(op, output[0].ctx(), var_in, var_out);
}
}
