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

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace neutron {

std::unique_ptr<ModelT> PrepareModel(TfLiteContext* context,
                                     const TfLiteDelegateParams* params);

TfLiteStatus ComputeSlice(TfLiteTensor* input,
                          TfLiteTensor* output,
                          SliceParams op_params);

TfLiteStatus ComputeReshape(TfLiteTensor* input,
                            TfLiteTensor* output,
                            ReshapeParams op_params);

bool IsNodeSupportedByNeutron(TfLiteContext* context,
                              const TfLiteNode* node,
                              int32_t builtin_code);

}  // namespace neutron
}  // namespace tflite

