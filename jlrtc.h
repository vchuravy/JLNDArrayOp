#ifndef JLRTC_H
#define JLRTC_h

#include <vector>
#include <unordered_map>
#include <cuda.h>
#include <mxnet/ndarray.h>
#include "device_array.h"
namespace mxnet {

class JLRtc {
  public:
    JLRtc() {};
    JLRtc(const std::string& name,
          std::vector<NDArray> const& args,
          char* ptx,
          unsigned int grid_dim_X,
          unsigned int grid_dim_Y,
          unsigned int grid_dim_Z,
          unsigned int block_dim_X,
          unsigned int block_dim_Y,
          unsigned int block_dim_Z);

    void verify(std::vector<NDArray> const& ndargs);
    CUfunction getFunc(int dev_id);
    void launch(CUfunction func, std::vector<NDArray> const& ndargs, RunContext& rctx);

  private:
    std::string name_;
    char* ptx_;
    std::vector<int> dtypes_;
    std::vector<TShape> shapes_;
    std::unordered_map<int, CUmodule> module_;
    std::unordered_map<int, CUfunction> func_;
    unsigned int grid_dim_X_;
    unsigned int grid_dim_Y_;
    unsigned int grid_dim_Z_;
    unsigned int block_dim_X_;
    unsigned int block_dim_Y_;
    unsigned int block_dim_Z_;
};
}
#endif
