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
  int num_inputs_, num_outputs_;
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
    this->backwards_jlrtc_ = JLRtc("", std::vector<NDArray>(), NULL, 0,0,0,0,0,0);
    this->forwards_jlrtc_ = JLRtc("", std::vector<NDArray>(), NULL, 0,0,0,0,0,0);
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
    param_.num_inputs_ = ListArguments().size();
    param_.num_outputs_ = ListOutputs().size();
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
