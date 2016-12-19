#pragma once

#include "debug_table.h"

__global__ void cu_setframe(long * frame, const long n, const long m);
__global__ void cu_levdist_kernel(long * frame, const long weftlen, const char t[], const long n, const char p[], const long m
#ifdef DEBUG_TABLE
	, long * dev_debug_table
#endif DEBUG_TABLE
);
__global__ void cu_weft_kernel(long * frame, const long weftlen, const char * t, const long n, const char * p, const long m, const long depth, const long thread_start, const long thread_last
#ifdef DEBUG_TABLE
	, long * dev_debug_table
#endif
);
__global__ void printThreads(void);

long cu_levdist(long * frame, const char t[], const long n, const char p[], const long m);
