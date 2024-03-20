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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_UTILS_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow::serving {
enum class BatchPaddingPolicy;  // Forward-declaring for the ABSL_DECLARE_FLAG.
}  // namespace tensorflow::serving

// Exposed for testing only.
ABSL_DECLARE_FLAG(tensorflow::serving::BatchPaddingPolicy,
                  tensorflow_batch_padding_policy);

namespace tensorflow {
namespace serving {

// Returns the next allowed batch size, which is the smallest allowed batch size
// greater than or equal to the given batch size. If allowed_batch_sizes,
// returns batch_size as is.
int GetNextAllowedBatchSize(int batch_size,
                            const std::vector<int32>& allowed_batch_sizes,
                            bool disable_padding);

// Returns the largest allowed batch size that is smaller than or equal to
// batch_size. Returns batch_size if no such size exists.
int GetPrevAllowedBatchSize(int batch_size,
                            const std::vector<int32>& allowed_batch_sizes,
                            bool disable_padding);

// See the description of the --tensorflow_batch_padding_policy flag (in the
// .cc file) for the documentation.
enum class BatchPaddingPolicy {
  kPadUp,
  kBatchDown,
  kMinimizeTpuCostPerRequest,
};
bool AbslParseFlag(absl::string_view text, BatchPaddingPolicy* out,
                   std::string* error);
std::string AbslUnparseFlag(BatchPaddingPolicy in);

// Trims the batch to the next allowed batch size when possible and when
// configured by the --tensorflow_batch_padding_policy flag.
//
// When trimming, this function puts the trimmed tasks go into the
// out_trimmed_tasks vector in the same order as they were in the batch.
template <typename TaskType>
void MaybeBatchDown(Batch<TaskType>& batch,
                    const std::vector<int32>& allowed_batch_sizes,
                    bool disable_padding, ModelBatchStats* model_batch_stats,
                    std::vector<std::unique_ptr<TaskType>>& out_trimmed_tasks) {
  bool minimize_tpu_cost_per_request;
  switch (absl::GetFlag(FLAGS_tensorflow_batch_padding_policy)) {
    case BatchPaddingPolicy::kPadUp:
      // This is the default behavior of batch resource when it is given a batch
      // size that doesn't match any of the allowed batch sizes.
      return;
    case BatchPaddingPolicy::kBatchDown:
      minimize_tpu_cost_per_request = false;
      break;
    case BatchPaddingPolicy::kMinimizeTpuCostPerRequest:
      if (model_batch_stats == nullptr) {
        LOG(DFATAL)
            << "MINIMIZE_TPU_COST_PER_REQUEST batching policy has been chosen "
               "but no ModelBatchStats passed to the batch scheduler; will "
               "fall back on the PAD_UP policy.";
        return;
      }
      minimize_tpu_cost_per_request = true;
      break;
  }

  int32 batch_size = batch.size();

  int32 pad_up_size =
      GetNextAllowedBatchSize(batch_size, allowed_batch_sizes, disable_padding);
  if (pad_up_size == batch_size) {
    return;  // Good, no padding is necessary.
  }

  int32 batch_down_size =
      GetPrevAllowedBatchSize(batch_size, allowed_batch_sizes, disable_padding);
  if (batch_down_size == batch_size) {
    return;  // Can't batch down (e.g. no smaller batch size available).
  }

  if (minimize_tpu_cost_per_request) {
    std::optional<absl::Duration> down_batch_cost =
        model_batch_stats->batch_size(batch_down_size).tpu_cost().mean();
    std::optional<absl::Duration> up_batch_cost =
        model_batch_stats->batch_size(pad_up_size).tpu_cost().mean();
    if (!down_batch_cost.has_value() || !up_batch_cost.has_value()) {
      // We have no data about batch costs, let's just not do anything.
      return;
    }

    auto batch_down_cost_per_request = *down_batch_cost / batch_down_size;
    auto pad_up_cost_per_request = *up_batch_cost / batch_size;

    if (pad_up_cost_per_request < batch_down_cost_per_request) {
      // Abort batching down because it's cheaper to pad up.
      return;
    }
  }

  // Batch down.
  batch.TryTrimToNewSize(batch_down_size, out_trimmed_tasks);
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_UTILS_H_
