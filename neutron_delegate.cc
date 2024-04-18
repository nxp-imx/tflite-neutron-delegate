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

#include "neutron_delegate.h"

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

extern "C" {
#include "neutron/NeutronDriver.h"
}

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
    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);

    dcfg.resize(params->nodes_to_replace->size);
    mcfg.resize(params->nodes_to_replace->size);
    nmh.resize(params->nodes_to_replace->size);

    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
        context,
        context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                        &delegated_node_registration),
        kTfLiteOk);

      for (int index = 0; index < delegated_node->inputs->size; index ++)
        inputs_[i].push_back(delegated_node->inputs->data[index]);
      for (int index = 0; index < delegated_node->outputs->size; index ++)
        outputs_[i].push_back(delegated_node->outputs->data[index]);
    }
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    TF_LITE_ENSURE(context, node->inputs->size > 2);

    for (int i = 0; i < inputs_.size(); ++i) {
      // Allocate arrays for inputs and outputs
      dcfg[i].inputs = new const void *[node->inputs->size];
      dcfg[i].outputs = new void *[node->outputs->size];

      // Get address to microcode data.
      auto microcodeIndex = inputs_[i][inputs_[i].size() - 3];
      TF_LITE_ENSURE(context, microcodeIndex < context->tensors_size);
      auto microcodeTensor = &context->tensors[microcodeIndex];
      // Set microcode address in neutron structure
      mcfg[i].microcode = static_cast<const void *>(microcodeTensor->data.raw);

      // Get address to weights data.
      auto weightsIndex = inputs_[i][inputs_[i].size() - 2];
      TF_LITE_ENSURE(context, weightsIndex < context->tensors_size);
      auto weightsTensor = &context->tensors[weightsIndex];
      // Set weights address in neutron structure.
      mcfg[i].weights = static_cast<const void *>(weightsTensor->data.raw);

      // Get address to kernel data.
      auto kernelIndex = inputs_[i][inputs_[i].size() - 1];
      TF_LITE_ENSURE(context, kernelIndex < context->tensors_size);
      auto kernelTensor = &context->tensors[kernelIndex];
      // Set kernels address in neutron structure.
      mcfg[i].kernels = static_cast<const void *>(kernelTensor->data.raw);

      // Prepare data for through neutron driver.
      auto neutronRC = neutronModelPrepare(&mcfg[i], &nmh[i]);
      TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
    }
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    TF_LITE_ENSURE(context, node->inputs->size > 2);

    for (int i = 0; i < inputs_.size(); ++i) {
      // Set reference for all inputs.
      for (int index = 0; index < inputs_[i].size() - 2; index ++) {
        auto tensor = &context->tensors[inputs_[i][index]];
        dcfg[i].inputs[index] = static_cast<const void *>(tensor->data.raw);
      }

      // Set reference for all outputs.
      for (int index = 0; index < outputs_[i].size() - 1; index ++) {
        auto tensor = &context->tensors[outputs_[i][index]];
        dcfg[i].outputs[index] = static_cast<void *>(tensor->data.raw);
      }
      // Run neutron compute.
      auto neutronRC = neutronRunBlocking(nmh[i], &dcfg[i]);
      TF_LITE_ENSURE_EQ(context, neutronRC, ENONE);
    }
    return kTfLiteOk;
  }

  ~NeutronDelegateKernel() {
    // Unprepare to free resources in neutron driver
    for (int i = 0; i < inputs_.size(); ++i) {
      neutronModelUnprepare(nmh[i]);
      delete dcfg[i].inputs;
      delete dcfg[i].outputs;
    }
  }

 private:
  NeutronDelegateOptions options;
  std::vector<NeutronModelConfig> mcfg;
  std::vector<NeutronDataConfig> dcfg;
  std::vector<NeutronModelHandle> nmh;

  std::vector<std::vector<int>> inputs_, outputs_;
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
    if (registration->builtin_code == kTfLiteBuiltinCustom &&
        strcmp(registration->custom_name, NEUTRON_CUSTOM_NAME) == 0) {
      return true;
    }

    return false;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
    NeutronError err = neutronInit();
    TF_LITE_ENSURE_EQ(context, err, ENONE);
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

  ~NeutronDelegate() {
    neutronDeinit();
  }

 private:
  const NeutronDelegateOptions options_;
};

}  // namespace neutron
}  // namespace tflite

NeutronDelegateOptions NeutronDelegateOptionsDefault() {
  NeutronDelegateOptions options;
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
