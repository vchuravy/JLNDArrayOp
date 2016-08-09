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
  std::vector<void*> ptrs;
  std::vector<Engine::VarHandle> ndvar;
  std::vector<int> tags;
  for (auto& i : req) CHECK_NE(i, kAddTo);

  for (auto& blob : in_data) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(0);
  }
  for (auto& blob : out_data) {
    NDArray* nd = new NDArray(blob, ndctx.dev_id);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    ndvar.push_back(nd->var());
    tags.push_back(1);
  }
  std::sort(ndvar.begin(), ndvar.end());
  ndvar.resize(std::unique(ndvar.begin(), ndvar.end()) - ndvar.begin());

  std::vector<NDArray> ndcpy;
  for (auto& i : ptrs) {
    ndcpy.push_back(*reinterpret_cast<NDArray*>(i));
  }

  // TODO: issue call to push in JLRtc directly
  // CHECK(param_.pinfo->forward(ptrs.size(), ptrs.data(), tags.data(), param_.pinfo->p_forward));
  Engine::Get()->PushSync([ndcpy, ctx](RunContext rctx) {ctx.async_on_complete(); },
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
  std::vector<void*> ptrs;
  std::vector<Engine::VarHandle> ndvar;
  std::vector<int> tags;
  for (auto& i : req) CHECK_NE(i, kAddTo);

  for (auto& blob : in_data) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(0);
  }
  for (auto& blob : out_data) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(1);
  }
  for (auto& blob : in_grad) {
    NDArray* nd = new NDArray(blob, ndctx.dev_id);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    ndvar.push_back(nd->var());
    tags.push_back(2);
  }
  std::sort(ndvar.begin(), ndvar.end());
  ndvar.resize(std::unique(ndvar.begin(), ndvar.end()) - ndvar.begin());
  for (auto& blob : out_grad) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(3);
  }

  std::vector<NDArray> ndcpy;
  for (auto& i : ptrs) {
    ndcpy.push_back(*reinterpret_cast<NDArray*>(i));
  }

  // TODO: issue call to push in JLRtc directly
  // CHECK(param_.pinfo->backward(ptrs.size(), ptrs.data(), tags.data(), param_.pinfo->p_backward));
  Engine::Get()->PushSync([ndcpy, ctx](RunContext rctx){ ctx.async_on_complete(); },
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
