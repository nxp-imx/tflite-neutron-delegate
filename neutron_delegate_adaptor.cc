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

#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"

#include "neutron_delegate.h"

namespace tflite {
namespace neutron {

TfLiteDelegate* CreateNeutronDelegateFromOptions(char** options_keys,
                                               char** options_values,
                                               size_t num_options) {
  NeutronDelegateOptions options = NeutronDelegateOptionsDefault();

  // Parse key-values options to NeutronDelegateOptions by mimicking them as
  // command-line flags.
  return NeutronDelegateCreate(&options);
}

}  // namespace neutron
}  // namespace tflite

extern "C" {

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
TFL_CAPI_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*)) {
  return tflite::neutron::CreateNeutronDelegateFromOptions(
      options_keys, options_values, num_options);
}

TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  NeutronDelegateDelete(delegate);
}

}  // extern "C"
