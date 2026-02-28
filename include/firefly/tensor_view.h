#pragma once

#if defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L
#include <mdspan>
namespace firefly_impl = std;
#else
#include <mdspan/mdspan.hpp>
namespace firefly_impl = Kokkos;
#endif

namespace firefly
{

using firefly_impl::dextents;
using firefly_impl::extents;
using firefly_impl::full_extent;
using firefly_impl::mdspan;

using firefly_impl::default_accessor;
using firefly_impl::layout_left;
using firefly_impl::layout_right;
using firefly_impl::layout_stride;

}  // namespace firefly