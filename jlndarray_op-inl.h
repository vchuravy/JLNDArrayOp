/*!
 * Copyright (c) 2015 by Contributors
 * \file native_op-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_JLNDARRAY_OP_INL_H_
#define MXNET_OPERATOR_JLNDARRAY_OP_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "../../src/operator/operator_common.h"
#include <mxnet/c_api.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include "jlrtc.h"

namespace mxnet {
namespace op {

struct JLNDArrayOpParam : public dmlc::Parameter<JLNDArrayOpParam> {
  void *info;

  NDArrayOpInfo *pinfo;
  unsigned int grid_dim_X;
  unsigned int grid_dim_Y;
  unsigned int grid_dim_Z;
  unsigned int block_dim_X;
  unsigned int block_dim_Y;
  unsigned int block_dim_Z;

  std::vector<int> ndims;
  std::vector<TShape> shapes;
  std::vector<int> dtypes;

  DMLC_DECLARE_PARAMETER(JLNDArrayOpParam) {
    DMLC_DECLARE_FIELD(info);
  }
};

template<typename xpu>
class JLNDArrayOp : public Operator {
 public:
  explicit JLNDArrayOp(JLNDArrayOpParam p) {
    this->param_ = p;
    // TODO: Setup JLRtc
    // call into this->param_.backward/forward to get ptx.
    // and name?
    this->forwards_jlrtc_ = JLRtc("", NULL,
                                   this->param_.ndims,
                                   this->param_.shapes,
                                   this->param_.dtypes,
                                   this->param_.grid_dim_X,
                                   this->param_.grid_dim_Y,
                                   this->param_.grid_dim_Z,
                                   this->param_.block_dim_X,
                                   this->param_.block_dim_Y,
                                   this->param_.block_dim_Z);
    // TODO: How to get ndims, shapes and dtypes for backwards?
    // this->backwards_jlrtc_ = JLRtc("", std::vector<NDArray>(), NULL, 0,0,0,0,0,0);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args);

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args);

  virtual ExecType exec_type() const {
    return kAsync;
  }

 private:
  JLNDArrayOpParam param_;
  Context get_ctx();
  JLRtc forwards_jlrtc_;
  JLRtc backwards_jlrtc_;
};  // NDArrayOp

template<typename xpu>
Operator* CreateOp(JLNDArrayOpParam param);

#if DMLC_USE_CXX11
class JLNDArrayOpProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    char ** args = NULL;
    CHECK(param_.pinfo->list_arguments(&args, param_.pinfo->p_list_arguments));
    std::vector<std::string> ret;
    for (int i = 0; args[i] != NULL; ++i) {
      ret.push_back(args[i]);
    }
    return ret;
  }

  std::vector<std::string> ListOutputs() const override {
    char ** args = NULL;
    CHECK(param_.pinfo->list_outputs(&args, param_.pinfo->p_list_outputs));
    std::vector<std::string> ret;
    for (int i = 0; args[i] != NULL; ++i) {
      ret.push_back(args[i]);
    }
    return ret;
  }

  int NumOutputs() const override {
    return param_.num_outputs_;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
    for (auto iter = kwargs.begin(); iter != kwargs.end(); ++iter) {
      if (iter->first == "info") {
        sscanf(iter->second.c_str(), "%p", &param_.pinfo);
      }
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }


  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    std::vector<unsigned*> shapes;
    std::vector<int> ndims;
    for (auto iter = in_shape->begin(); iter != in_shape->end(); ++iter) {
      shapes.push_back(iter->data());
      ndims.push_back(iter->ndim());
    }
    shapes.resize(param_.num_inputs_+param_.num_outputs_);
    ndims.resize(param_.num_inputs_+param_.num_outputs_);
    CHECK(param_.pinfo->infer_shape(shapes.size(), ndims.data(), shapes.data(),
                                    param_.pinfo->p_infer_shape));
    for (unsigned i = 0; i < in_shape->size(); ++i) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, TShape(shapes[i], shapes[i]+ndims[i]));
    }
    out_shape->clear();
    for (unsigned i = param_.num_inputs_; i < shapes.size(); ++i) {
      out_shape->push_back(TShape(shapes[i], shapes[i]+ndims[i]));
    }

    std::vector<TShape> tshapes;
    for (TShape shape: *in_shape) tshapes.push_back(tshape);
    for (TShape shape: *out_shape) tshapes.push_back(tshape);

    this->param_.ndims = ndims;
    this->param_.shapes = tshapes;
    return true;
  }

  OperatorProperty* Copy() const override {
    JLNDArrayOpProp *prop_sym = new JLNDArrayOpProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "_NDArray";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    int num_dep;
    int *rdeps;
    CHECK(param_.pinfo->declare_backward_dependency(out_grad.data(), in_data.data(),
                                                    out_data.data(), &num_dep, &rdeps,
                                                    param_.pinfo->p_declare_backward_dependency));
    std::vector<int> deps;
    deps.insert(deps.end(), rdeps, rdeps+num_dep);
    return deps;
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  JLNDArrayOpParam param_;
};  // class PythonProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NDARRAY_OP_INL_H_
