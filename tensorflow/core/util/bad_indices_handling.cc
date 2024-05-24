/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/util/bad_indices_handling.h"

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

BadIndicesOnCpu BadIndicesOnCpuFromString(absl::string_view str) {
  if (str.empty()) return BadIndicesOnCpu::kError;
  if (str == "error") return BadIndicesOnCpu::kError;
  if (str == "ignore") return BadIndicesOnCpu::kIgnore;
  LOG(ERROR) << "Unknown bad indices handling: " << str;
  return BadIndicesOnCpu::kError;
}

}  // namespace tensorflow
