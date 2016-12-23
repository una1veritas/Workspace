#pragma once

#include "debug_table.h"

__global__ void weaving_kernel(long * frame, const long weftlen, const char t[], const long n, const char p[], const long m
#ifdef DEBUG_TABLE
	, long * dev_debug_table
#endif DEBUG_TABLE
);

__global__ void weaving_cdp_kernel(long * frame, const long weftlen, const char t[], const long n, const char p[], const long m
#ifdef DEBUG_TABLE
	, long * table
#endif
);
__global__ void warps_cdp_kernel(long * frame, const char * t, const long n, const char * p, const long m, const long depth, const long warp_start, const long warp_last
#ifdef DEBUG_TABLE
	, long * table
#endif
);

void weaving_setframe(long * frame, const long n, const long m);
long cu_levdist(long * frame, const char t[], const long n, const char p[], const long m);
