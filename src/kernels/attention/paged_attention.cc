#include "firefly/kernels/attention/detail/launch.h"

namespace firefly::kernels::attention_detail
{
void launch_paged(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
                  const DecodeConfig& decode_config, float scale, const device::Context& context)
{
    bool use_split_decode = options.prefer_split_decode && query.shape()[1] == 1 && query.shape()[3] <= 1024;
    if (decode_config.force_single) use_split_decode = false;
    if (decode_config.force_split && query.shape()[1] == 1 && query.shape()[3] <= 1024)
    {
        use_split_decode = true;
    }

    if (use_split_decode)
    {
        launch_paged_decode(query, key, value, output, options, decode_config, scale, context);
    }
    else
    {
        launch_paged_prefill(query, key, value, output, options, scale, context);
    }
}
}  // namespace firefly::kernels::attention_detail
