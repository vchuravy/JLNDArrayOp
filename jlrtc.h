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
    JLRtc(const std::string& name,
          std::vector<NDArray> const& input,
          std::vector<NDArray> const& output,
          char* ptx);

    void push(std::vector<NDArray> const& input,
              std::vector<NDArray> const& output,
              unsigned int grid_dim_X,
              unsigned int grid_dim_Y,
              unsigned int grid_dim_Z,
              unsigned int block_dim_X,
              unsigned int block_dim_Y,
              unsigned int block_dim_Z);

  private:
    std::string name_;
    char* ptx_;
    std::vector<int> in_dtypes_;
    std::vector<int> out_dtypes_;
    std::vector<TShape> in_shapes_;
    std::vector<TShape> out_shapes_;
    std::unordered_map<int, CUmodule> module_;
    std::unordered_map<int, CUfunction> func_;
};
}
#endif
