#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels
{

void append_paged_kv(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                     const int* block_table, const int* context_lens, int max_blocks_per_sequence,
                     const device::Context& context = {});

}  // namespace firefly::kernels
