#pragma once
// DO NOT MERGE: host-side phase timers for the patchA study. Enabled by GGML_HOST_PROF=1.
#include "ggml.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
GGML_API int64_t ggml_host_prof_now(void);                              // 0 when disabled
GGML_API void    ggml_host_prof_add(const char * name, int64_t t_start); // no-op when t_start == 0
GGML_API void    ggml_host_prof_count(const char * name);
GGML_API void    ggml_host_prof_reset(void);
#ifdef __cplusplus
}
#endif
