/*
 * Copyright 2023-2024 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utility>
#include <string.h>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

#include "neutron_delegate_utils.h"
#include "neutron_delegate.h"
#include "enum_mapping.h"

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

extern "C" {
#include "neutron/NeutronDriver.h"
}
#include "neutron/NeutronConverter.h"

using namespace std;

namespace tflite {
namespace neutron {

// Neutron delegate kernel.
class NeutronDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit NeutronDelegateKernel(const NeutronDelegateOptions& opt)
      : options(opt){}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    if (options.is_neutron_model) {
      return InitOfflineCompiledModel(context, params);
    } else {
      return InitInlineCompiledModel(context, params);
    }
  }

  TfLiteStatus InitOfflineCompiledModel(TfLiteContext* context,
                    const TfLiteDelegateParams* params) {
    operations.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      auto &delegate_op = operations[i];
      // Get this node information.
      const int node_index = params->nodes_to_replace->data[i];
      TfLiteNode* node = nullptr;
      TfLiteRegistration* node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
        context,
        context->GetNodeAndRegistration(context, node_index, &node,
                                        &node_registration),
        kTfLiteOk);

      for (int index = 0; index < node->inputs->size; index ++)
        delegate_op.inputs.push_back(node->inputs->data[index]);
      for (int index = 0; index < node->outputs->size; index ++)
        delegate_op.outputs.push_back(node->outputs->data[index]);

      // Get address to microcode data.
      auto mIndex = delegate_op.inputs[node->inputs->size - 3];
      TF_LITE_ENSURE(context, mIndex < context->tensors_size);
      auto mTensor = &context->tensors[mIndex];
      // Set microcode address in neutron structure
      delegate_op.mcfg.microcode = static_cast<const void *>(mTensor->data.raw);

      // Get address to weights data.
      auto wIndex = delegate_op.inputs[node->inputs->size - 2];
      TF_LITE_ENSURE(context, wIndex < context->tensors_size);
      auto wTensor = &context->tensors[wIndex];
      // Set weights address in neutron structure.
      delegate_op.mcfg.weights = static_cast<const void *>(wTensor->data.raw);

      // Get address to kernel data.
      auto kernelIndex = delegate_op.inputs[node->inputs->size - 1];
      TF_LITE_ENSURE(context, kernelIndex < context->tensors_size);
      auto kernelTensor = &context->tensors[kernelIndex];
      // Set kernels address in neutron structure.
      delegate_op.mcfg.kernels = static_cast<const void *>(kernelTensor->data.raw);
      delegate_op.builtin_code = BuiltinOperator_CUSTOM;
    }
    return kTfLiteOk;
  }

  TfLiteStatus InitInlineCompiledModel(TfLiteContext* context,
                    const TfLiteDelegateParams* params) {
    // Convert model to neutron format
    auto modelPre = PrepareModel(context, params);
    flatbuffers::FlatBufferBuilder fbb;
    flatbuffers::Offset<tflite::Model> root = tflite::Model::Pack(fbb, modelPre.get());
    tflite::FinishModelBuffer(fbb, root);
    uint32_t size = fbb.GetSize();
    uint8_t *data = fbb.GetBufferPointer();
    std::vector<uint8_t> buff(data, data + size);

    auto  cvt_out = converter::convertModel(buff, options.target);
    model = std::unique_ptr<ModelT>(tflite::GetModel(cvt_out.data())->UnPack());

    TF_LITE_ENSURE_EQ(context, model->subgraphs.size(), 1);

    // Map the neutron tensor index to tflite tensor index.
    auto &neutron_ops = model->subgraphs[0]->operators;
    std::map<int, int> tensor_map;
    for (int i = 0; i < model->subgraphs[0]->inputs.size(); i ++){
      auto neutron_idx = model->subgraphs[0]->inputs[i];
      int j;
      int tflite_idx;

      for (j = 0; j < params->input_tensors->size; j ++) {
        tflite_idx = params->input_tensors->data[j];
        if (context->tensors[tflite_idx].allocation_type != kTfLiteMmapRo)
            break;
      }
      tflite_idx = params->input_tensors->data[j + i];
      tensor_map[neutron_idx] = tflite_idx;
    }
    for (int i = 0; i < model->subgraphs[0]->outputs.size(); i ++){
      auto neutron_idx = model->subgraphs[0]->outputs[i];
      auto tflite_idx = params->output_tensors->data[i];
      tensor_map[neutron_idx] = tflite_idx;
    }

    operations.resize(neutron_ops.size());
    for (int idx = 0; idx < neutron_ops.size(); idx ++) {
      auto &op_code = model->operator_codes[neutron_ops[idx]->opcode_index];
      auto &delegate_op = operations[idx];

      size_t input_size, output_size;
      if (op_code->builtin_code == BuiltinOperator_SLICE) {
        const int kMaxDim = 5; //Slice op only supports 1D-5D input arrays
        input_size = neutron_ops[idx]->inputs.size() - 2;
        output_size = neutron_ops[idx]->outputs.size();

        auto index_begin = neutron_ops[idx]->inputs[1];
        const auto &tensor_begin = model->subgraphs[0]->tensors[index_begin];
        auto begin_data = (int32_t*)model->buffers[tensor_begin->buffer]->data.data();

        auto index_size = neutron_ops[idx]->inputs[2];
        const auto &tensor_size = model->subgraphs[0]->tensors[index_size];
        auto size_data = (int32_t*)model->buffers[tensor_size->buffer]->data.data();

        TF_LITE_ENSURE_EQ(context, tensor_begin->type, TensorType_INT32);
        TF_LITE_ENSURE_EQ(context, tensor_size->type, TensorType_INT32);
        TF_LITE_ENSURE_EQ(context, tensor_begin->shape.size(), 1);
        TF_LITE_ENSURE_EQ(context, tensor_size->shape.size(), 1);
        TF_LITE_ENSURE_EQ(context, tensor_begin->shape[0], tensor_size->shape[0]);
        TF_LITE_ENSURE(context, tensor_begin->shape[0] <= kMaxDim);

        delegate_op.params.slice.begin_count = tensor_begin->shape[0];
        delegate_op.params.slice.size_count = tensor_size->shape[0];
        for (int i = 0; i < tensor_begin->shape[0]; i ++) {
            delegate_op.params.slice.begin[i] = begin_data[i];
            delegate_op.params.slice.size[i] = size_data[i];
        }
      } else if (op_code->builtin_code == BuiltinOperator_CUSTOM
                       && op_code->custom_code == NEUTRON_CUSTOM_NAME) {
        input_size = neutron_ops[idx]->inputs.size() - 3;
        output_size = neutron_ops[idx]->outputs.size() - 1;

	// Get address to microcode data.
        auto indexM = neutron_ops[idx]->inputs[input_size];
        const auto &tensorM = model->subgraphs[0]->tensors[indexM];
        delegate_op.mcfg.microcode = model->buffers[tensorM->buffer]->data.data();

	// Get address to weights data.
        auto indexW = neutron_ops[idx]->inputs[input_size + 1];
        const auto &tensorW = model->subgraphs[0]->tensors[indexW];
        delegate_op.mcfg.weights = model->buffers[tensorW->buffer]->data.data();

	// Get address to kernels data.
        auto indexK = neutron_ops[idx]->inputs[input_size + 2];
        const auto &tensorK = model->subgraphs[0]->tensors[indexK];
        delegate_op.mcfg.kernels = model->buffers[tensorK->buffer]->data.data();
      } else if (op_code->builtin_code == BuiltinOperator_RESHAPE) {
        //The reshape output shape is set by neutron-convertor
        input_size = 1;
        output_size = 1;
      } else {
        TF_LITE_KERNEL_LOG(context, "Failed to build Neutron graph.\n");
        return kTfLiteDelegateError;
      }
      delegate_op.builtin_code = op_code->builtin_code;

      for (int j = 0; j < input_size; j ++) {
        auto neutron_idx = neutron_ops[idx]->inputs[j];
        TF_LITE_ENSURE(context, tensor_map.count(neutron_idx) > 0);
        delegate_op.inputs.push_back(tensor_map[neutron_idx]);
      }
      for (int j = 0; j < output_size; j ++) {
        auto neutron_idx = neutron_ops[idx]->outputs[j];
        int tflite_idx;
        if(tensor_map.count(neutron_idx) > 0) {
          tflite_idx = tensor_map[neutron_idx];
        } else {
          context->AddTensors(context, 1, &tflite_idx);
          auto tmp_tensor = &(context->tensors[tflite_idx]);
          auto &neutron_tensor = model->subgraphs[0]->tensors[neutron_idx];
          tmp_tensor->type = SchemaTypeToTfLiteType(neutron_tensor->type);
          tmp_tensor->allocation_type = kTfLiteDynamic;
          tmp_tensor->name = "tmp_tensor";
          auto dims = ConvertVectorToTfLiteIntArray(neutron_tensor->shape);
          TF_LITE_ENSURE_OK(context,
               context->ResizeTensor(context, tmp_tensor, dims));
          tensor_map[neutron_idx] = tflite_idx;

        }
        delegate_op.outputs.push_back(tflite_idx);
      }
    }
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    for (auto& op : operations) {
      if (op.builtin_code == BuiltinOperator_CUSTOM) {
        // Allocate arrays for inputs and outputs
        op.dcfg.inputs = new const void*[op.inputs.size()];
        op.dcfg.outputs = new void*[op.outputs.size()];

        // Prepare data for through neutron driver.
        auto neutronRC = neutronModelPrepare(&op.mcfg, &op.nmh);
        TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
      }
    }
    return kTfLiteOk;
  }


  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    for (auto &delegate_op : operations) {
      if (delegate_op.builtin_code == BuiltinOperator_CUSTOM) {
        // Set reference for all inputs
        for (int index = 0; index < delegate_op.inputs.size(); index ++) {
            auto tensor_index = delegate_op.inputs[index];
            auto input = &context->tensors[tensor_index];
            delegate_op.dcfg.inputs[index] = input->data.raw;
        }

        for (int index = 0; index < delegate_op.outputs.size(); index ++) {
            auto tensor_index = delegate_op.outputs[index];
            auto output = &context->tensors[tensor_index];
            delegate_op.dcfg.outputs[index] = output->data.raw;
        }

        // Run neutron compute.
        auto neutronRC = neutronRunBlocking(delegate_op.nmh, &delegate_op.dcfg);
        TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
      } else if (delegate_op.builtin_code == BuiltinOperator_SLICE){
        auto input = &context->tensors[delegate_op.inputs[0]];
        auto output = &context->tensors[delegate_op.outputs[0]];
        ComputeSlice(input, output, delegate_op.params.slice);
      } else if (delegate_op.builtin_code == BuiltinOperator_RESHAPE) {
        auto input = &context->tensors[delegate_op.inputs[0]];
        auto output = &context->tensors[delegate_op.outputs[0]];
        ComputeReshape(input, output, delegate_op.params.reshape);
      }
    }
    return kTfLiteOk;
  }

  ~NeutronDelegateKernel() {
    for (auto& op : operations) {
      if (op.builtin_code == BuiltinOperator_CUSTOM) {
        // Unprepare to free resources in neutron driver
        neutronModelUnprepare(op.nmh);
        // Delete arrays for inputs and outputs
        delete[] op.dcfg.inputs;
        delete[] op.dcfg.outputs;
      }
    }
  }

 private:
  struct OperationDataType {
    vector<int> inputs;
    vector<int> outputs;

    // Aggregate neutron model and data structures into one
    NeutronModelConfig mcfg;
    NeutronDataConfig dcfg;
    NeutronModelHandle nmh;

    union {
      SliceParams slice;
      ReshapeParams reshape;
    } params;
    BuiltinOperator builtin_code;
  };
  std::unique_ptr<ModelT> model;

  int slice_input;

  vector<OperationDataType> operations;
  NeutronDelegateOptions options;
};

// NeutronDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class NeutronDelegate : public SimpleDelegateInterface {
 public:
  explicit NeutronDelegate(const NeutronDelegateOptions& options)
      : options_(options) {
      }
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    bool ret;
    if (options_.is_neutron_model) {
      ret = (registration->builtin_code == kTfLiteBuiltinCustom &&
             strcmp(registration->custom_name, NEUTRON_CUSTOM_NAME) == 0);
    } else {
      ret = IsNodeSupportedByNeutron(context, node,
                                     registration->builtin_code);
    }
    return ret;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
    // Initialize the neutron driver library
    NeutronError err = neutronInit();
    TF_LITE_ENSURE_EQ(context, err, ENONE);

    TfLiteIntArray* plan;
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));

    for (int node_index : tflite::TfLiteIntArrayView(plan)) {
      TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
          context, node_index, &node, &registration));
      if (registration->builtin_code == kTfLiteBuiltinCustom &&
          strcmp(registration->custom_name, NEUTRON_CUSTOM_NAME) == 0) {
        options_.is_neutron_model = true;
	return kTfLiteOk;
      }
    }

    options_.is_neutron_model = false;
    return kTfLiteOk; 
  }

  const char* Name() const override {
    static constexpr char kName[] = "NeutronDelegate";
    return kName;
  }

  unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return make_unique<NeutronDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  NeutronDelegateOptions options_;
};

}  // namespace neutron
}  // namespace tflite

NeutronDelegateOptions NeutronDelegateOptionsDefault() {
  NeutronDelegateOptions options;
  options.target = "imx95";

  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteNeutronDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* NeutronDelegateCreate(const NeutronDelegateOptions* options) {
  auto delegate = make_unique<tflite::neutron::NeutronDelegate>(
          options ? *options : NeutronDelegateOptionsDefault());
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(move(delegate), 
             kTfLiteDelegateFlagsAllowDynamicTensors);
}

// Destroys a delegate created with `NeutronDelegateCreate` call.
void NeutronDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
