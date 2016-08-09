/*!
 * Copyright (c) 2015 by Contributors
 * \file ndarray_op.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./jlndarray_op-inl.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

namespace mxnet {
namespace op {
template<>
Context JLNDArrayOp<cpu>::get_ctx() {
  return Context::CPU();
}

template<>
Operator *CreateOp<cpu>(JLNDArrayOpParam param) {
  return new JLNDArrayOp<cpu>(param);
}

#if MXNET_USE_CUDA
template<>
Context JLNDArrayOp<gpu>::get_ctx() {
  int dev_id;
  CHECK_EQ(cudaGetDevice(&dev_id), cudaSuccess);
  return Context::GPU(dev_id);
}

template<>
Operator* CreateOp<gpu>(JLNDArrayOpParam param) {
  return new JLNDArrayOp<gpu>(param);
}
#endif  // MXNET_USE_CUDA

template<typename xpu>
void JLNDArrayOp<xpu>::Forward(const OpContext &ctx,
                   const std::vector<TBlob> &in_data,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &out_data,
                   const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  Context ndctx = get_ctx();
  std::vector<NDArray> args;
  // ndvar is for the outer sync point
  std::vector<Engine::VarHandle> ndvar;
  // var_in, var_out is for the inner sync point
  std::vector<Engine::VarHandle> var_in, var_out;

  for (auto& i : req) CHECK_NE(i, kAddTo);

  for (auto& blob : in_data) {
    NDArray nd = NDArray(blob, ndctx.dev_id);
    args.push_back(nd);
    var_in.push_back(nd.var());
  }
  for (auto& blob : out_data) {
    NDArray nd = NDArray(blob, ndctx.dev_id);
    args.push_back(nd);
    var_out.push_back(nd.var());
    ndvar.push_back(nd.var());
  }

  std::sort(ndvar.begin(), ndvar.end());
  ndvar.resize(std::unique(ndvar.begin(), ndvar.end()) - ndvar.begin());

  // Verify signature of arguments & get function from jlrtc
  forwards_jlrtc_.verify(args);
  CUfunction func = forwards_jlrtc_.getFunc(ndctx.dev_id);

  auto op = [this, func, args](RunContext rctx) {
    forwards_jlrtc_.launch(func, args, rctx);
  };

  // Issue sync on input arguments and output arguments
  Engine::Get()->PushSync(op, ndctx, var_in, var_out);

  // Issue read sync on output arguments
  Engine::Get()->PushSync([args, ctx](RunContext rctx) {ctx.async_on_complete(); },
                          ndctx, ndvar, {});
}

template<typename xpu>
void JLNDArrayOp<xpu>::Backward(const OpContext &ctx,
                    const std::vector<TBlob> &out_grad,
                    const std::vector<TBlob> &in_data,
                    const std::vector<TBlob> &out_data,
                    const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &in_grad,
                    const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  Context ndctx = get_ctx();
  std::vector<NDArray> args;
  // ndvar for the outer sync point
  std::vector<Engine::VarHandle> ndvar;
  // var_in, var_out is for the inner sync point
  std::vector<Engine::VarHandle> var_in, var_out;

  for (auto& i : req) CHECK_NE(i, kAddTo);

  for (auto& blob : in_data) {
    NDArray nd = NDArray(blob, ndctx.dev_id);
    args.push_back(nd);
    var_in.push_back(nd.var());
  }
  for (auto& blob : out_data) {
    NDArray nd = NDArray(blob, ndctx.dev_id);
    args.push_back(nd);
    var_in.push_back(nd.var());
  }
  for (auto& blob : in_grad) {
    NDArray nd = NDArray(blob, ndctx.dev_id);
    args.push_back(nd);
    // Why is in_grad and not out_grad set as ndvar.
    ndvar.push_back(nd.var());
    var_out.push_back(nd.var());
  }
  std::sort(ndvar.begin(), ndvar.end());
  ndvar.resize(std::unique(ndvar.begin(), ndvar.end()) - ndvar.begin());
  for (auto& blob : out_grad) {
    NDArray nd = NDArray(blob, ndctx.dev_id);
    args.push_back(nd);
    var_in.push_back(nd.var());
  }

  // Verify signature of arguments & get function from jlrtc
  backwards_jlrtc_.verify(args);
  CUfunction func = backwards_jlrtc_.getFunc(ndctx.dev_id);

  auto op = [this, func, args](RunContext rctx) {
    backwards_jlrtc_.launch(func, args, rctx);
  };

  // Issue sync on input arguments and output arguments
  Engine::Get()->PushSync(op, ndctx, var_in, var_out);

  // Issue outer sync
  Engine::Get()->PushSync([args, ctx](RunContext rctx){ ctx.async_on_complete(); },
                          ndctx, ndvar, {});
}

Operator* JLNDArrayOpProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(JLNDArrayOpParam);

MXNET_REGISTER_OP_PROPERTY(_JLNDArray, JLNDArrayOpProp)
.describe("Stub for implementing an operator implemented in native frontend language with ndarray.")
.add_arguments(JLNDArrayOpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
