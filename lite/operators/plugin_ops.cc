// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/operators/plugin_ops.h"
#include <algorithm>
#include <cmath>
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool PluginOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool PluginOp::InferShapeImpl() const {
  auto x_dim = param_.X->dims();
  auto y_dim = param_.Y->dims();
  if (x_dim == y_dim) {
    param_.Out->Resize(x_dim);
    auto out_lod = param_.Out->mutable_lod();
    *out_lod = param_.X->lod();
  } else {
    size_t max_dim =
        (x_dim.size() > y_dim.size() ? x_dim.size() : y_dim.size());
    int axis = param_.axis;
    axis = (axis == -1 ? std::abs(static_cast<int>(x_dim.size() - y_dim.size()))
                       : axis);
    std::vector<int64_t> x_dims_array(max_dim);
    std::vector<int64_t> y_dims_array(max_dim);
    std::vector<int64_t> out_dims_array(max_dim);

    if (x_dim.size() > y_dim.size()) {
      for (int i = 0; i < axis; ++i) {
        y_dims_array[i] = 1;
      }
      if (axis + y_dim.size() < max_dim) {
        for (size_t i = axis + y_dim.size(); i < max_dim; ++i) {
          y_dims_array[i] = 1;
        }
      }
      x_dims_array = x_dim.Vectorize();
      for (size_t i = 0; i < y_dim.size(); ++i) {
        y_dims_array[i + axis] = y_dim[i];
      }
    } else {
      for (int i = 0; i < axis; ++i) {
        x_dims_array[i] = 1;
      }
      if (axis + x_dim.size() < max_dim) {
        for (size_t i = axis + x_dim.size(); i < max_dim; ++i) {
          x_dims_array[i] = 1;
        }
      }
      y_dims_array = y_dim.Vectorize();
      for (size_t i = 0; i < x_dim.size(); ++i) {
        x_dims_array[i + axis] = x_dim[i];
      }
    }
    for (size_t i = 0; i < max_dim; i++) {
      if (x_dims_array[i] == -1 || y_dims_array[i] == -1) {
        out_dims_array[i] = 1;
      } else {
        out_dims_array[i] = (std::max)(x_dims_array[i], y_dims_array[i]);
      }
    }
    param_.Out->Resize(DDim(out_dims_array));
    auto out_lod = param_.Out->mutable_lod();
    *out_lod = param_.X->lod();
  }

  return true;
}

bool PluginOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Y_name = opdesc.Input("Y").front();
  auto Out_name = opdesc.Output("Out").front();

  param_.X = GetMutableVar<lite::Tensor>(scope, X_name);
  param_.Y = GetMutableVar<lite::Tensor>(scope, Y_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);
  param_.axis = opdesc.GetAttr<int>("axis");

  param_.computesize = opdesc.GetAttr<bool>("computesize");

  input_tensor_ptrs_cache_.push_back(param_.X);
  input_tensor_ptrs_cache_.push_back(param_.Y);
  output_tensor_ptrs_cache_.push_back(param_.Out);

// //for test
//   if (opdesc.HasAttr("fuse_scale")) {
//     param_.fuse_scale = opdesc.GetAttr<bool>("fuse_scale");
//     param_.scale = opdesc.GetAttr<float>("scale");
//     param_.alpha = opdesc.GetAttr<float>("alpha");
//     param_.bias = opdesc.GetAttr<float>("bias");
// }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(plugin_add, paddle::lite::operators::PluginOp);
//for test
//REGISTER_LITE_OP(elementwise_add, paddle::lite::operators::PluginOp);


