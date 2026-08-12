#pragma once

#include <cstdlib>
#include <iostream>
#include <utility>

#include "firefly/core/tensor.h"

namespace firefly::test
{

/**
 * @brief Unwraps a tensor result for standalone test programs that cannot propagate `Status` from `main`.
 * @param result Tensor allocation result produced by `Tensor::create`.
 * @return Successfully allocated tensor.
 * @note A failed test prerequisite is printed and terminates the test process without throwing an exception.
 */
inline Tensor require_tensor(Result<Tensor> result)
{
    if (!result)
    {
        std::cerr << result.error().describe() << '\n';
        std::exit(1);
    }
    return std::move(result.value());
}

}  // namespace firefly::test
