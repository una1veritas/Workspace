#pragma once

#include "debug_table.h"

__global__ void weaving_kernel(long * frame, const long weftlen, const char t[], const long n, const char p[], const long m
#ifdef DEBUG_TABLE
	, long * dev_debug_table
#endif DEBUG_TABLE
);

long cu_levdist(long * frame, const char t[], const long n, const char p[], const long m);
