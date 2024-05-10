/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_performance_model_base.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/fusions.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace {

// Returns whether a fusion uses the parameter at the given index elementwise
// from its root. Also works if 'fusion' is a multi-output fusion.
bool FusionUsesParameterElementwiseFromRoot(
    const HloInstruction* fusion, int parameter_index,
    const GpuHloCostAnalysis* cost_analysis) {
  // This checks whether there is a path from fused_expression_root() to the
  // parameter that only goes through elementwise, Tuple and GetTupleElement
  // ops.
  return cost_analysis->CommonElementwiseUtilization(
             fusion->fused_parameter(parameter_index),
             fusion->fused_expression_root()) == 1.f;
}

int GetCoalescingWasteFactor(PrimitiveType element_type,
                             const se::DeviceDescription& gpu_device_info) {
  int64_t element_size_bytes =
      element_type == PrimitiveType::TUPLE ||
              element_type == PrimitiveType::TOKEN
          ? 4 /* Dummy value. TODO(jreiffers): Model this case. */
          : ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  // Assume we use one element from the cache line and waste the remaining
  // bandwidth. For example, if we're reading f32s, we use 1/16nd of the cache
  // line.
  return gpu_device_info.dram_to_l2_transaction_size_bytes() /
         element_size_bytes;
}

// Limit the bandwidth for low occupancy cases. Each SM can issue at most
// one 32B memory transaction per clock. H100 needs at least 56.8 active SMs
// (1830 MHz) to saturate the memory bandwidth (3.35 TB/s).
float AdjustBandwidth(const se::DeviceDescription& gpu_device_info,
                      float bandwidth, int64_t num_blocks) {
  float per_block_bandwidth = gpu_device_info.clock_rate_ghz() * 1.0e9f *
                              gpu_device_info.memory_transactions_per_clock();
  float max_bandwidth = num_blocks * per_block_bandwidth;

  return std::min(bandwidth, max_bandwidth);
}

}  // namespace

std::optional<EstimateRunTimeData> GpuPerformanceModelCache::Get(
    const HloInstruction& instruction) {
  absl::MutexLock lock(&mutex_);

  auto it = instruction_runtime_data_.find(&instruction);
  if (it != instruction_runtime_data_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<absl::Duration> GpuPerformanceModelCache::Get(
    const HloInstruction& producer, const HloInstruction& consumer) {
  absl::MutexLock lock(&mutex_);

  auto it = fusion_runtime_data_.find(&producer);
  if (it != fusion_runtime_data_.end()) {
    auto jt = it->second.find(&consumer);
    if (jt != it->second.end()) {
      return jt->second;
    }
  }
  return std::nullopt;
}

void GpuPerformanceModelCache::Set(const HloInstruction& instruction,
                                   const EstimateRunTimeData& runtime_data) {
  absl::MutexLock lock(&mutex_);

  instruction_runtime_data_[&instruction] = runtime_data;
}

void GpuPerformanceModelCache::Set(const HloInstruction& producer,
                                   const HloInstruction& consumer,
                                   absl::Duration runtime) {
  absl::MutexLock lock(&mutex_);
  fusion_runtime_data_[&producer][&consumer] = runtime;
}

void GpuPerformanceModelCache::Invalidate(const HloInstruction& instruction) {
  absl::MutexLock lock(&mutex_);

  // Remove runtime data for the instruction.
  instruction_runtime_data_.erase(&instruction);

  // Remove cache for all producer-consumer pairs where the instruction is
  // producer.
  fusion_runtime_data_.erase(&instruction);

  // Iterate through operands to find all producer-consumer pairs where
  // instruction is consumer and remove them from cache.
  for (auto* operand : instruction.operands()) {
    if (operand->opcode() == HloOpcode::kGetTupleElement) {
      operand = operand->mutable_operand(0);
    }
    auto it = fusion_runtime_data_.find(operand);
    if (it != fusion_runtime_data_.end()) {
      it->second.erase(&instruction);
    }
  }
}

/*static*/
LaunchDimensions GpuPerformanceModelBase::EstimateFusionLaunchDimensions(
    const HloFusionAnalysis& fusion_analysis) {
  auto emitter =
      GetFusionEmitter(PreBufferAssignmentFusionInfo{fusion_analysis});
  if (const auto* kernel_emitter =
          dynamic_cast<const KernelFusionInterface*>(emitter.get())) {
    return kernel_emitter->launch_dimensions();
  }

  // This estimate should never be reached in fusion code. Fusions that don't
  // implement KernelFusionInterface, don't generate GPU kernels, so there is
  // nothing to fuse. Keep this estimate as a simple fallback.
  //
  // We assume that the kernel launches 1 thread per output element and 128
  // threads per block. In multi-output fusions, only look at one root.
  VLOG(5) << "Using fallback launch dimensions estimate for "
          << fusion_analysis.fusion().ToString();
  int64_t num_threads_per_block = 128;
  int64_t estimated_num_threads =
      ShapeUtil::ElementsInRecursive(fusion_analysis.fusion_root(0).shape());
  int64_t num_blocks =
      CeilOfRatio(estimated_num_threads, num_threads_per_block);
  return LaunchDimensions(num_blocks, num_threads_per_block);
}

/*static*/
int64_t GpuPerformanceModelBase::GetOperandBytesAccessed(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
    const HloInstruction* operand) {
  // When called for a producer-consumer fusion, the operand can be from a
  // different instruction. GpuHloCostAnalysis can't fail gracefully in this
  // case, so we need an explicit check.
  if (!instr->IsUserOf(operand)) {
    return 0;
  }

  return cost_analysis->operand_bytes_accessed(*instr,
                                               instr->operand_index(operand));
}

/*static*/
float GpuPerformanceModelBase::GetOperandUtilization(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
    const HloInstruction* operand) {
  if (operand->IsMultiOutputFusion()) {
    // If 'operand' is a multi-output fusion, we need to check which of its
    // outputs are used by 'instr'.
    float res = 0.f;
    for (int64_t i = 0; i < instr->operand_count(); ++i) {
      if (instr->operand(i)->opcode() == HloOpcode::kGetTupleElement &&
          instr->operand(i)->operand(0) == operand) {
        res += cost_analysis->operand_utilization(*instr, i);
      }
    }
    return res;
  }
  // When called for a producer-consumer fusion, the operand can be from a
  // different instruction. GpuHloCostAnalysis can't fail gracefully in this
  // case, so we need an explicit check.
  if (!instr->IsUserOf(operand)) {
    return 0.f;
  }

  return cost_analysis->operand_utilization(*instr,
                                            instr->operand_index(operand));
}

/*static*/
float GpuPerformanceModelBase::GetCommonUtilization(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* producer,
    int64_t producer_idx_of_operand, const HloInstruction* consumer) {
  const auto* operand = producer->operand(producer_idx_of_operand);

  if (!consumer || !consumer->IsUserOf(operand)) {
    return 0.f;
  }

  if (producer->IsElementwise() ||
      (producer->opcode() == HloOpcode::kFusion &&
       FusionUsesParameterElementwiseFromRoot(producer, producer_idx_of_operand,
                                              cost_analysis))) {
    if (consumer->opcode() == HloOpcode::kFusion) {
      int64_t consumer_idx_of_common_operand = consumer->operand_index(operand);
      float res = 0.f;
      std::vector<int64_t> consumer_indices_of_producer;
      if (producer->IsMultiOutputFusion()) {
        for (int64_t i = 0; i < consumer->operand_count(); ++i) {
          if (consumer->operand(i)->opcode() == HloOpcode::kGetTupleElement &&
              consumer->operand(i)->operand(0) == producer) {
            consumer_indices_of_producer.push_back(i);
          }
        }
      } else {
        consumer_indices_of_producer.push_back(
            consumer->operand_index(producer));
      }
      for (int64_t consumer_idx_of_producer : consumer_indices_of_producer) {
        res += cost_analysis->CommonElementwiseUtilization(
            consumer->fused_parameter(consumer_idx_of_common_operand),
            consumer->fused_parameter(consumer_idx_of_producer));
      }
      return res;
    } else if (consumer->IsElementwise()) {
      return 1.f;
    }
  }
  return 0.f;
}

/*static*/
int64_t GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* producer,
    const HloInstruction* consumer, const HloInstruction* operand) {
  float producer_utilization_by_consumer =
      GetOperandUtilization(cost_analysis, consumer, producer);

  int64_t bytes_accessed_by_producer =
      GetOperandBytesAccessed(cost_analysis, producer, operand);

  int64_t bytes_accessed_by_consumer =
      GetOperandBytesAccessed(cost_analysis, consumer, operand);

  float common_utilization =
      producer->IsUserOf(operand)
          ? GetCommonUtilization(cost_analysis, producer,
                                 producer->operand_index(operand), consumer)
          : 0.f;

  int64_t operand_size = cost_analysis->GetShapeSize(operand->shape());
  int64_t common_bytes_accessed =
      std::llround(operand_size * common_utilization);

  return std::llround(bytes_accessed_by_producer *
                      producer_utilization_by_consumer) +
         bytes_accessed_by_consumer - common_bytes_accessed;
}

/*static*/
absl::Duration GpuPerformanceModelBase::ReadTime(
    const se::DeviceDescription& gpu_device_info, int64_t num_blocks,
    int64_t n_bytes_net, int64_t n_bytes_total) {
  float bandwidth = gpu_device_info.memory_bandwidth();
  if (n_bytes_net < gpu_device_info.l2_cache_size()) {
    bandwidth *= kL2CacheSpeedup;
    if (n_bytes_net <
        gpu_device_info.l1_cache_size_per_SM() * gpu_device_info.core_count()) {
      bandwidth *= kL1CacheSpeedup;
    }
  }

  bandwidth = AdjustBandwidth(gpu_device_info, bandwidth, num_blocks);
  return absl::Seconds(n_bytes_total / bandwidth);
}

/*static*/
absl::Duration GpuPerformanceModelBase::ReadTimeWithDRAMHeuristic(
    const se::DeviceDescription& gpu_device_info, int64_t num_blocks,
    int64_t n_bytes_net, int64_t n_bytes_total, PrimitiveType element_type,
    bool coalesced) {
  int waste_factor =
      coalesced ? 1 : GetCoalescingWasteFactor(element_type, gpu_device_info);

  // The first read of the input buffer always happens from DRAM. If reads are
  // no coaleced, bandwidth is reduced by the waste factor.
  float dram_bandwidth = gpu_device_info.memory_bandwidth() / waste_factor;

  // Two things can happed on re-reading the buffer:
  //   - If the buffer fits into cache, the L1/L2 cache speedup is applied.
  //   - If the buffer doesn't fit, it will be read from DRAM and the same
  //     coalessing waste factor is applied.
  float rest_bandwidth = gpu_device_info.memory_bandwidth();
  if (n_bytes_net < gpu_device_info.l2_cache_size()) {
    rest_bandwidth *= kL2CacheSpeedup;
    if (n_bytes_net <
        gpu_device_info.l1_cache_size_per_SM() * gpu_device_info.core_count()) {
      rest_bandwidth *= kL1CacheSpeedup;
    }
  } else {
    rest_bandwidth /= waste_factor;
  }

  dram_bandwidth = AdjustBandwidth(gpu_device_info, dram_bandwidth, num_blocks);
  rest_bandwidth = AdjustBandwidth(gpu_device_info, rest_bandwidth, num_blocks);

  // n_bytes_net > n_bytes_total can happen when we compute read time of
  // shared operand. This is a flaw in the interface that should be fixed.
  int64_t n_bytes_read_dram = std::min(n_bytes_net, n_bytes_total);

  // Number of bytes that we be re-read, potentially from cache.
  int64_t n_bytes_read_cache = n_bytes_total - n_bytes_read_dram;

  return absl::Seconds(n_bytes_read_dram / dram_bandwidth) +
         absl::Seconds(n_bytes_read_cache / rest_bandwidth);
}

/*static*/ absl::Duration GpuPerformanceModelBase::ProducerInputAccessTime(
    const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info, int64_t num_blocks,
    const HloInstruction* producer, const HloFusionAnalysis& fusion_analysis,
    const GpuPerformanceModelOptions& config,
    const HloInstruction* fused_consumer) {
  absl::Duration ret = absl::ZeroDuration();
  float producer_output_utilization =
      fused_consumer
          ? GetOperandUtilization(cost_analysis, fused_consumer, producer)
          : 1.f;

  for (int i = 0; i < producer->operand_count(); ++i) {
    // Information about data read taking into account utilization.
    // If `operand_utilization` is 0, `operand_bytes_accessed` should be also 0.
    int64_t operand_bytes_accessed =
        cost_analysis->operand_bytes_accessed(*producer, i);
    float operand_utilization =
        cost_analysis->operand_utilization(*producer, i);

    // An estimate how much data would need to fit into L1/L2 cache to speed up
    // the operand access.
    // If `operand_utilization` < 1, only a part of the full operand size should
    // be read. Otherwise, `operand_bytes_accessed / operand_utilization` is the
    // size of the operand without reuse.
    int64_t n_bytes_net = std::llround(operand_bytes_accessed /
                                       std::max(operand_utilization, 1.0f));

    // Look if common operand of producer and consumer will be accessed more
    // efficiently on merge.
    float common_utilization = GetCommonUtilization(
        cost_analysis, producer, /*producer_idx_of_operand=*/i, fused_consumer);

    CHECK_LE(common_utilization, producer_output_utilization);
    float n_bytes_total = operand_bytes_accessed *
                          (producer_output_utilization - common_utilization);
    ret += ReadTime(gpu_device_info, num_blocks, n_bytes_net, n_bytes_total);
  }
  return ret;
}

/*static*/
absl::Duration GpuPerformanceModelBase::WriteTime(
    const se::DeviceDescription& gpu_device_info, int64_t bytes_written) {
  return absl::Seconds(1.0f * bytes_written /
                       gpu_device_info.memory_bandwidth());
}

/*static*/
absl::Duration GpuPerformanceModelBase::ComputeTime(
    const se::DeviceDescription& gpu_device_info, int64_t flops, int num_blocks,
    int num_threads_per_block) {
  int64_t n_active_fpus_per_core =
      std::min(num_threads_per_block, gpu_device_info.fpus_per_core());

  int64_t n_active_core = std::min(num_blocks, gpu_device_info.core_count());
  int64_t fpu_count = n_active_core * n_active_fpus_per_core;

  int64_t flop_per_ns_per_fpu = gpu_device_info.clock_rate_ghz() * /*fma:*/ 2;
  int64_t flop_per_ns_effective = flop_per_ns_per_fpu * fpu_count;
  return absl::Nanoseconds(1.0f * flops / flop_per_ns_effective);
}

/*static*/
absl::Duration GpuPerformanceModelBase::CombineComputeAndMemoryAccessTime(
    absl::Duration compute_time, absl::Duration memory_access_time,
    const GpuPerformanceModelOptions& config) {
  return compute_time + memory_access_time -
         std::min(compute_time, memory_access_time) *
             config.memory_compute_parallelism;
}

/*static*/
void GpuPerformanceModelBase::VLogOperandRead(const HloInstruction* operand,
                                              int64_t n_bytes_total,
                                              int64_t n_bytes_net,
                                              bool coalesced) {
  VLOG(8) << "operand " << operand->name()
          << ", n_bytes_total: " << n_bytes_total
          << ", n_bytes_net: " << n_bytes_net << ", coalesced: " << coalesced;
}

}  // namespace gpu
}  // namespace xla
