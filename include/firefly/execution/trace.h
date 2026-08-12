#pragma once

#ifdef FIREFLY_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
/**
 * @brief Begins an NVTX range on the calling thread.
 * @param name Null-terminated range label consumed by NVTX tracing tools.
 * @note The range must be balanced by a matching `FIREFLY_NVTX_POP()` on the same thread.
 */
#define FIREFLY_NVTX_PUSH(name) nvtxRangePush(name)
/** @brief Ends the most recently pushed NVTX range on the calling thread. */
#define FIREFLY_NVTX_POP() nvtxRangePop()
#else
/**
 * @brief Compiles to a no-op when Firefly is built without NVTX support.
 * @param name Ignored range label retained for source compatibility.
 */
#define FIREFLY_NVTX_PUSH(name) ((void)0)
/** @brief Compiles to a no-op when Firefly is built without NVTX support. */
#define FIREFLY_NVTX_POP() ((void)0)
#endif
