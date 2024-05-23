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

#include "neutron_delegate_utils.h"
#include "enum_mapping.h"

#include <numeric>

namespace tflite {
namespace neutron {

typedef std::initializer_list<TfLiteType> TypeList;
const TypeList supported_dtypes{kTfLiteUInt8, kTfLiteInt8};
const TypeList supported_bias_dtypes{kTfLiteInt32};

inline std::string CharPtrToStr(const char *in) {
  if (in == nullptr) {
    return std::string("");
  } else {
    return std::string(in);
  }
}

template <class A, class B>
std::vector<A> TfLiteArrayToVector(const B* int_array) {
  std::vector<A> values;
  if (!int_array) {
  return values;
  }

  values.resize(int_array->size);
  for (size_t i = 0; i < int_array->size; i++) {
  values[i] = int_array->data[i];
  }

  return values;
}

int64_t GetQuantizedMin(TensorType type) {
  switch (type) {
  case TensorType::TensorType_INT32:
    return std::numeric_limits<int32_t>::min();
  case TensorType::TensorType_UINT8:
    return std::numeric_limits<uint8_t>::min();
  case TensorType::TensorType_INT64:
    return std::numeric_limits<int64_t>::min();
  case TensorType::TensorType_INT16:
    return std::numeric_limits<int16_t>::min();
  case TensorType::TensorType_INT8:
    return std::numeric_limits<int8_t>::min();
  default:
    printf("Tensor element type is not quantized!\n");
  }
  return std::numeric_limits<int8_t>::min();
}

int64_t GetQuantizedMax(TensorType type) {
  switch (type) {
  case TensorType::TensorType_INT32:
    return std::numeric_limits<int32_t>::max();
  case TensorType::TensorType_UINT8:
    return std::numeric_limits<uint8_t>::max();
  case TensorType::TensorType_INT64:
    return std::numeric_limits<int64_t>::max();
  case TensorType::TensorType_INT16:
    return std::numeric_limits<int16_t>::max();
  case TensorType::TensorType_INT8:
    return std::numeric_limits<int8_t>::max();
  default:
    printf("Tensor element type is not quantized!\n");
  }
  return std::numeric_limits<int8_t>::max();
}

std::vector<float> GetFloatMin(TensorType type,
                               std::vector<float> scale,
                               std::vector<int64_t> offset) {
  assert(scale.size() == offset.size());
  auto size = scale.size();
  std::vector<float> ret;

  ret.reserve(size);
  for (int i = 0; i < size; i ++) {
      ret[i] = scale[i] * static_cast<float>(GetQuantizedMin(type) - offset[i]);
  }
  return ret;
}

std::vector<float> GetFloatMax(TensorType type,
                               std::vector<float> scale,
                               std::vector<int64_t> offset) {
  assert(scale.size() == offset.size());
  auto size = scale.size();
  std::vector<float> ret;

  ret.reserve(size);
  for (int i = 0; i < size; i ++) {
      ret[i] = scale[i] * static_cast<float>(GetQuantizedMax(type) - offset[i]);
  }
  return ret;
}

void SetBuiltinOptions(OperatorT *op, int32_t op_code, void* data){
  switch (op_code) {
    case BuiltinOperator_ADD: {
      auto params = reinterpret_cast<TfLiteAddParams*>(data);
      auto option = AddOptionsT();
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_CONV_2D: {
      auto params = reinterpret_cast<TfLiteConvParams*>(data);
      auto option = Conv2DOptionsT();
      option.padding = TfLitePaddingToSchemaPadding(params->padding);
      option.stride_w = params->stride_width;
      option.stride_h = params->stride_height;
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      option.dilation_w_factor = params->dilation_width_factor;
      option.dilation_h_factor = params->dilation_height_factor;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      auto params = reinterpret_cast<TfLiteDepthwiseConvParams*>(data);
      auto option = DepthwiseConv2DOptionsT();
      option.padding = TfLitePaddingToSchemaPadding(params->padding);
      option.stride_w = params->stride_width;
      option.stride_h = params->stride_height;
      option.depth_multiplier = params->depth_multiplier;
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      option.dilation_w_factor = params->dilation_width_factor;
      option.dilation_h_factor = params->dilation_height_factor;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_MAX_POOL_2D:
    case BuiltinOperator_AVERAGE_POOL_2D: {
      auto params = reinterpret_cast<TfLitePoolParams*>(data);
      auto option = Pool2DOptionsT();
      option.padding = TfLitePaddingToSchemaPadding(params->padding);
      option.stride_w = params->stride_width;
      option.stride_h = params->stride_height;
      option.filter_width = params->filter_width;
      option.filter_height = params->filter_height;
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_FULLY_CONNECTED: {
      auto params = reinterpret_cast<TfLiteFullyConnectedParams*>(data);
      auto option = FullyConnectedOptionsT();
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      option.weights_format = FullyConnectedOptionsWeightsFormatToSchema(params->weights_format);
      option.keep_num_dims = params->keep_num_dims;
      option.asymmetric_quantize_inputs = params->asymmetric_quantize_inputs;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_PAD: {
      auto option = PadOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_RESHAPE: {
      auto params = reinterpret_cast<TfLiteReshapeParams*>(data);
      auto option = ReshapeOptionsT();
      for (int i = 0; i < params->num_dimensions; i ++) {
          option.new_shape.push_back(params->shape[i]);
      }
      op->builtin_options.Set(option);
      break;
    }
    default: {
       printf("Can't support this op_code:%d!\n", op_code);
       exit(0);
    }
  }
}

std::unique_ptr<ModelT> PrepareModel(TfLiteContext* context,
                                     const TfLiteDelegateParams* params) {
  ModelT *modelT = new ModelT;

  // Copy model version.
  modelT->version = 3;

  // Copy model buffers.
  // The model must always have the first buffer (sentinel) an empty buffer used for empty tensors/metadata.
  modelT->buffers.emplace_back(new BufferT);

  // Copy model graphs.
  modelT->subgraphs.reserve(1);

  // Create new graph.
  modelT->subgraphs.emplace_back(new SubGraphT);
  SubGraphT *graphNew = modelT->subgraphs.back().get();

  // Copy graph tensors.
  graphNew->tensors.reserve(context->tensors_size);
  for (int i = 0; i < context->tensors_size; i ++) {
    auto tensor = &context->tensors[i];
    // Create new tensor.
    graphNew->tensors.emplace_back(new TensorT);
    TensorT *tensorNew = graphNew->tensors.back().get();

    // Copy tensor data.
    tensorNew->shape = TfLiteArrayToVector<int, TfLiteIntArray>(tensor->dims);
    tensorNew->type = TfLiteTypeToSchemaType(tensor->type);
    if (tensor->bytes == 0 || tensor->data.raw == nullptr || tensor->allocation_type != kTfLiteMmapRo) {
      tensorNew->buffer = 0;
    } else {
      // Create new buffer.
      modelT->buffers.emplace_back(new BufferT);
      modelT->buffers.back()->data = std::vector<uint8_t>(tensor->data.uint8,
                       tensor->data.uint8 + tensor->bytes);
      tensorNew->buffer = modelT->buffers.size() - 1;
    }
    tensorNew->name = CharPtrToStr(tensor->name);
    if (tensor->quantization.type == kTfLiteNoQuantization) {
      tensorNew->quantization = std::unique_ptr<QuantizationParametersT>(nullptr);
    } else {
      tensorNew->quantization = std::unique_ptr<QuantizationParametersT>(new QuantizationParametersT);
      auto affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
      tensorNew->quantization->scale = TfLiteArrayToVector<float, TfLiteFloatArray>(affine_quantization->scale);
      tensorNew->quantization->zero_point = TfLiteArrayToVector<int64_t, TfLiteIntArray>(affine_quantization->zero_point);
      tensorNew->quantization->min = GetFloatMin(tensorNew->type, tensorNew->quantization->scale, tensorNew->quantization->zero_point);
      tensorNew->quantization->max = GetFloatMax(tensorNew->type, tensorNew->quantization->scale, tensorNew->quantization->zero_point);
      tensorNew->quantization->quantized_dimension = affine_quantization->quantized_dimension;
    }
    tensorNew->is_variable = tensor->is_variable;
    assert(tensor->sparsity == nullptr);
    tensorNew->sparsity = std::unique_ptr<SparsityParametersT>(nullptr);
    tensorNew->shape_signature = TfLiteArrayToVector<int, TfLiteIntArray>(tensor->dims_signature);
  }

  // Copy graph inputs.
  auto inputs = params->input_tensors;
  graphNew->inputs.reserve(inputs->size);
  for (int i = 0; i < inputs->size; i ++) {
    if (context->tensors[inputs->data[i]].allocation_type != kTfLiteMmapRo) {
      graphNew->inputs.push_back(inputs->data[i]);
    }
  }

  // Copy graph outputs.
  auto outputs = params->output_tensors;
  graphNew->outputs.reserve(outputs->size);
  for (int i = 0; i < outputs->size; i ++) {
    graphNew->outputs.push_back(outputs->data[i]);
  }

  // Copy model operator codes.
  auto addModelOperatorCode = [&](const TfLiteRegistration* reg) -> uint32_t {
    // Reuse operator code if already added.
    for (size_t idx = 0; idx < modelT->operator_codes.size(); ++idx) {
      auto opcode = *(modelT->operator_codes[idx]);
      if (opcode.builtin_code == reg->builtin_code
          && reg->custom_name && opcode.custom_code == reg->custom_name
          && opcode.version == reg->version) {
        return static_cast<uint32_t>(idx);
      }
    }
    // Create new operator code.
    modelT->operator_codes.emplace_back(new OperatorCodeT);
    modelT->operator_codes.back()->builtin_code = (BuiltinOperator)reg->builtin_code;
    if (reg->custom_name)
      modelT->operator_codes.back()->custom_code = reg->custom_name;
    else
      modelT->operator_codes.back()->custom_code = "";
    modelT->operator_codes.back()->version = reg->version;
    return modelT->operator_codes.size() - 1;
  };

  // Copy graph operators.
  auto nodes_index = params->nodes_to_replace;
  graphNew->operators.reserve(nodes_index->size);
  for (int i = 0; i < nodes_index->size; i ++) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    context->GetNodeAndRegistration(context, nodes_index->data[i], &node, &reg);
    // Create new operator.
    graphNew->operators.emplace_back(new OperatorT);
    OperatorT *opNew = graphNew->operators.back().get();

    // Copy operator data.
    opNew->opcode_index = addModelOperatorCode(reg);
    opNew->inputs.reserve(node->inputs->size);
    for (int i = 0; i < node->inputs->size; i ++) {
      opNew->inputs.push_back(node->inputs->data[i]);
    }
    opNew->outputs.reserve(node->outputs->size);
    for (int i = 0; i < node->outputs->size; i ++) {
      opNew->outputs.push_back(node->outputs->data[i]);
    }
    SetBuiltinOptions(opNew, reg->builtin_code, node->builtin_data);
    assert(node->custom_initial_data == nullptr);
    opNew->custom_options_format = CustomOptionsFormat_FLEXBUFFERS;
    opNew->intermediates = TfLiteArrayToVector<int, TfLiteIntArray>(node->intermediates);
  }

  // Copy graph name.
  graphNew->name = "neutron-delegate";
  // Copy model description.
  modelT->description = "neutron-delegate";

  return std::move(std::unique_ptr<ModelT>(modelT));
}

TfLiteStatus ComputeReshape(TfLiteTensor* input,
                            TfLiteTensor* output,
                            ReshapeParams& op_params) {

  memcpy(output->data.raw, input->data.raw, input->bytes);
  return kTfLiteOk;
}

TfLiteStatus ComputeRequantize(TfLiteTensor* input,
                               TfLiteTensor* output) {
  int i = 0;
  int size = input->bytes;

  if (input->type == kTfLiteUInt8) {
#ifdef USE_NEON
    for (; i <= size - 16; i += 16) {
      const uint8x16_t input_vec = vld1q_u8(input->data.uint8 + i);
      uint8x16_t tmp = vdupq_n_u8(128);
      uint8x16_t sub_result = vsubq_u8(input_vec, tmp);
      vst1q_u8(output->data.uint8 + i, sub_result);
    }
#endif
    for (; i < size; i ++){
      output->data.int8[i] = input->data.uint8[i] - 128;
    }
  } else {
#ifdef USE_NEON
    for (; i <= size - 16; i += 16) {
      const int8x16_t input_vec = vld1q_s8(input->data.int8 + i);
      int8x16_t tmp = vdupq_n_s8(128);
      int8x16_t add_result = vaddq_s8(input_vec, tmp);
      vst1q_s8(output->data.int8 + i, add_result);
    }
#endif
    for (; i < size; i ++){
      output->data.uint8[i] = input->data.int8[i] + 128;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus ComputePad(TfLiteTensor* input,
                        TfLiteTensor* output,
			PadParams& op_params) {
  const int8_t pad_value = static_cast<int8_t>(output->params.zero_point);

  optimized_ops::Pad(op_params, GetTensorShape(input),
                     GetTensorData<int8_t>(input),
                     &pad_value, GetTensorShape(output),
                     GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus ComputeSlice(TfLiteTensor* input,
                          TfLiteTensor* output,
                          SliceParams& op_params) {
  const int kMaxDim = 5;

  // The Slice op implementation only accepts 5-D sizes. That constraint is, for
  // the present, maintained here.
  //
  // The dimensions in the kernel used to be in reverse-order, and TFLite
  // arranged the begins and sizes vectors accordingly. This macro incorporates
  // the needed reversing.
#define TF_LITE_SLICE(data_type)                                               \
  {                                                                            \
    optimized_ops::Slice<data_type>(op_params, GetTensorShape(input), input,   \
                                      GetTensorShape(output), output);         \
  }

  switch (input->type) {
    case kTfLiteFloat32:
      TF_LITE_SLICE(float);
      break;
    case kTfLiteInt32:
      TF_LITE_SLICE(int32_t);
      break;
    case kTfLiteInt64:
      TF_LITE_SLICE(int64_t);
      break;
    case kTfLiteInt8:
      TF_LITE_SLICE(int8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_SLICE(int16_t);
      break;
    case kTfLiteUInt8:
      TF_LITE_SLICE(uint8_t);
      break;
    case kTfLiteBool:
      TF_LITE_SLICE(bool);
      break;
    case kTfLiteString:
      TF_LITE_SLICE(string);
      break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void* GetNeutronInputData(ModelT *model, int op_idx, int input_idx) {
  auto &neutron_op = model->subgraphs[0]->operators[op_idx];
  auto index = neutron_op->inputs[input_idx];
  const auto &tensor = model->subgraphs[0]->tensors[index];

  return model->buffers[tensor->buffer]->data.data();
}

inline void Expect(bool condition, bool* supported){
  if (!condition) {
    *supported = false;
  }
}

inline void ExpectTypeIn(TfLiteType actual_type,
                         std::initializer_list<TfLiteType> allowed_types,
                         bool* supported) {
  auto find_ret = std::find(allowed_types.begin(),
                            allowed_types.end(), actual_type);
  if (find_ret == allowed_types.end())
    *supported = false;
}

inline void ExpectInputTypeIn(TfLiteContext* context,
                              const TfLiteNode* node,
                              int index,
                              std::initializer_list<TfLiteType> allowed_types,
                              bool* supported) {
  if (index >= node->inputs->size) {
    *supported = false;
  } else {
    const TfLiteType input_type =
            context->tensors[node->inputs->data[index]].type;
    ExpectTypeIn(input_type, allowed_types, supported);
  }
}

inline void ExpectBiasTypeIn(TfLiteContext* context,
                             const TfLiteNode* node,
                             int index,
                             std::initializer_list<TfLiteType> allowed_types,
                             bool* supported) {
  if (index >= node->inputs->size) {
    *supported = true;
  } else {
    const TfLiteType input_type =
            context->tensors[node->inputs->data[index]].type;
    ExpectTypeIn(input_type, allowed_types, supported);
  }
}

inline void ExpectInputConstant(TfLiteContext* context,
                                const TfLiteNode* node,
                                int index,
                                bool* supported) {
  if (index >= node->inputs->size or
    context->tensors[node->inputs->data[index]].allocation_type != kTfLiteMmapRo) {
    *supported = false;
  }
}

inline void ExpectInputNotConstant(TfLiteContext* context,
                                   const TfLiteNode* node,
                                   int index,
                                   bool* supported) {
  if (index >= node->inputs->size or
    context->tensors[node->inputs->data[index]].allocation_type == kTfLiteMmapRo) {
    *supported = false;
  }
}

inline void ExpectOutputTypeIn(TfLiteContext* context,
                               const TfLiteNode* node,
                               int index,
                               std::initializer_list<TfLiteType> allowed_types,
                               bool* supported) {
  if (index >= node->outputs->size) {
    *supported = false;
  } else {
    const TfLiteType input_type =
            context->tensors[node->outputs->data[index]].type;
    ExpectTypeIn(input_type, allowed_types, supported);
  }
}

inline std::vector<int> GetInputTensorShape(TfLiteContext* context,
                                            const TfLiteNode* node,
                                            int index) {
  auto &tensor = context->tensors[node->inputs->data[index]];
  return TfLiteArrayToVector<int, TfLiteIntArray>(tensor.dims);
}

inline std::vector<int> GetOutputTensorShape(TfLiteContext* context,
                                            const TfLiteNode* node,
                                            int index) {
  auto &tensor = context->tensors[node->outputs->data[index]];
  return TfLiteArrayToVector<int, TfLiteIntArray>(tensor.dims);
}

inline void CheckMaxPool2d(const std::vector<int> &input_shape,
                           const std::vector<int> &output_shape,
                           const TfLitePoolParams *params,
                           const int32_t num_macs,
                           bool *supported) {
  Expect(input_shape.size() == 4 && output_shape.size() == 4, supported);
  Expect(params->stride_height == 1 || params->stride_height == 2, supported);
  Expect(input_shape[0] == 1 && output_shape[0] == 1, supported);
  Expect(params->computed.padding.width < params->filter_width, supported);
  Expect(params->computed.padding.height < params->filter_height, supported);
}

inline void CheckPad(const std::vector<int> &input_shape,
                     const std::vector<int> &output_shape,
                     const int32_t num_macs,
                     bool *supported) {
  Expect(input_shape.size() == 4 && output_shape.size() == 4, supported);
  Expect(output_shape[3] % num_macs == 0, supported);
  Expect(input_shape[3] <= output_shape[3], supported);
  Expect((output_shape[3] - input_shape[3]) <= (num_macs + 1), supported);
}

bool IsNodeSupportedByNeutron(TfLiteContext* context,
                              const TfLiteNode* node,
                              int32_t builtin_code) {
  bool supported = true;
  auto data = node->builtin_data;
  auto input_shape = GetInputTensorShape(context, node, 0);
  auto output_shape = GetOutputTensorShape(context, node, 0);
  auto num_macs = 16; //i.MX 95

  switch (builtin_code){
    case kTfLiteBuiltinAdd: {
      /*
        -Input tensors must be INT8/UINT8.
        -Output tensor must be INT8/UINT8.
        -The offset in memory between the two input operands
         must be smaller than 524288 (512k) WORDS.
      */
      ExpectInputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectInputTypeIn(context, node, 1, supported_dtypes, &supported);
      ExpectOutputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectInputNotConstant(context, node, 0, &supported);
      ExpectInputNotConstant(context, node, 1, &supported);
      break;
    }
    case kTfLiteBuiltinDepthwiseConv2d: {
      /*
        -Input tensor must be INT8/UINT8.
        -Filter tensor must be INT8/UINT8.
        -Bias tensor must be INT32.
        -Output tensor must be INT8/UINT8.
        -Filter tensor must be constant.
        -Bias tensor must be constant.
        -The depth multiplier must be 1.
      */
      auto params = reinterpret_cast<TfLiteDepthwiseConvParams*>(data);
      ExpectInputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectInputTypeIn(context, node, 1, supported_dtypes, &supported);
      ExpectBiasTypeIn(context, node, 2, supported_bias_dtypes, &supported);
      ExpectOutputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectInputConstant(context, node, 1, &supported);
      ExpectInputConstant(context, node, 2, &supported);
      Expect(params->depth_multiplier == 1, &supported);
      Expect(input_shape.size() == 4 && output_shape.size() == 4, &supported);
      Expect(params->dilation_width_factor == 1, &supported);
      Expect(params->dilation_height_factor == 1, &supported);
      Expect(params->stride_width == 1 || params->stride_width == 2, &supported);
      Expect(params->stride_height == 1 || params->stride_height == 2, &supported);
      break;
    }
    case kTfLiteBuiltinAveragePool2d: {
      /*
        -Input tensor must be INT8/UINT8.
        -Output tensor must be INT8/UINT8.
        -Input batch size must be 1.
        -The number of input channels must be a multiple of NUM_MACS.
      */
      ExpectInputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectOutputTypeIn(context, node, 0, supported_dtypes, &supported);
      break;
    }
    case kTfLiteBuiltinMaxPool2d: {
      /*
        -Input tensor must be INT8/UINT8.
        -Output tensor must be INT8/UINT8.
        -Input and output batch size must be 1.
        -Stride for height/width must be 1 or 2.
        -The number of input and output channels must be a multiple of NUM_MACS.
        -The top and bottom padding must be smaller than the kernel height.
        -The left and right padding must be smaller than the kernel width.
      */
      auto params = reinterpret_cast<TfLitePoolParams*>(data);
      ExpectInputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectOutputTypeIn(context, node, 0, supported_dtypes, &supported);
      CheckMaxPool2d(input_shape, output_shape, params, num_macs, &supported);
      break;
    }
    case kTfLiteBuiltinConv2d: {
      /*
        -Input tensor must be INT8/UINT8.
        -Filter tensor must be INT8/UINT8.
        -Bias tensor must be INT32.
        -Output tensor must be INT8/UINT8.
        -Filter tensor must be constant.
        -Bias tensor must be constant.
      */
      auto params = reinterpret_cast<TfLiteConvParams*>(data);
      ExpectInputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectInputTypeIn(context, node, 1, supported_dtypes, &supported);
      ExpectBiasTypeIn(context, node, 2, supported_bias_dtypes, &supported);
      ExpectOutputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectInputConstant(context, node, 1, &supported);
      ExpectInputConstant(context, node, 2, &supported);
      Expect(input_shape.size() == 4 && output_shape.size() == 4, &supported);
      Expect(params->dilation_width_factor == 1, &supported);
      Expect(params->dilation_height_factor == 1, &supported);
      break;
    }
    case kTfLiteBuiltinFullyConnected: {
      /*
        -Input tensor must be INT8/UINT8.
        -Weights tensor must be INT8/UINT8.
        -Bias tensor must be INT32.
        -Output tensor must be INT8/UINT8.
        -Weights tensor must be constant.
        -Bias tensor must be constant.
      */
      ExpectInputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectInputTypeIn(context, node, 1, supported_dtypes, &supported);
      ExpectBiasTypeIn(context, node, 2, supported_bias_dtypes, &supported);
      ExpectOutputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectInputConstant(context, node, 1, &supported);
      ExpectInputConstant(context, node, 2, &supported);
      break;
    }
    case kTfLiteBuiltinPad: {
      /*
        -Input tensor must be INT8/UINT8.
        -Weights tensor must be INT8/UINT8.
        -Only channel padding is supported for a NHWC input/output tensor:
           -The number of output channels must be a multiple of NUM_MACS.
           -The difference between the number of input/output tensors
            must be smaller than numMacs + 1.
      */
      ExpectInputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectOutputTypeIn(context, node, 0, supported_dtypes, &supported);
      CheckPad(input_shape, output_shape, num_macs, &supported);
      break;
    }
    case kTfLiteBuiltinReshape: {
      ExpectInputTypeIn(context, node, 0, supported_dtypes, &supported);
      ExpectInputTypeIn(context, node, 1, {kTfLiteInt32}, &supported);
      ExpectInputConstant(context, node, 1, &supported);
      break;
    }
    default: {
      supported = false;
      break;
    }
  }

  return supported;
}

}  // namespace neutron
}  // namespace tflite

