#pragma once

#include <mdspan/mdspan.hpp>
namespace firefly_impl = Kokkos;

namespace firefly
{

/** @brief Dynamic-rank extent descriptor re-exported for tensor views. */
using firefly_impl::dextents;
/** @brief Static or partially dynamic extent descriptor re-exported for tensor views. */
using firefly_impl::extents;
/** @brief Slice sentinel selecting the full extent of an mdspan dimension. */
using firefly_impl::full_extent;
/** @brief Non-owning multidimensional view type used by typed tensor accessors. */
using firefly_impl::mdspan;

/** @brief Default mdspan accessor policy. */
using firefly_impl::default_accessor;
/** @brief Column-major mdspan layout policy. */
using firefly_impl::layout_left;
/** @brief Row-major mdspan layout policy. */
using firefly_impl::layout_right;
/** @brief Arbitrary-stride mdspan layout policy used by `Tensor::view`. */
using firefly_impl::layout_stride;

}  // namespace firefly
