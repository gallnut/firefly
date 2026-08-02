#pragma once

#ifdef FIREFLY_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#define FIREFLY_NVTX_PUSH(name) nvtxRangePush(name)
#define FIREFLY_NVTX_POP() nvtxRangePop()
#else
#define FIREFLY_NVTX_PUSH(name) ((void)0)
#define FIREFLY_NVTX_POP() ((void)0)
#endif
