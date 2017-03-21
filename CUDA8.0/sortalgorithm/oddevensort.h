#pragma once
#ifndef __ODDEVENSORT_HEADER__
#define __ODDEVENSORT_HEADER__

__global__ void exchEven(int *A, const unsigned int n);
__global__ void exchOdd(int *A, const unsigned int n);
__global__ void exch256(int *A, const unsigned int n, const unsigned int offset);

void cu_oddevensort_gmem(int *devArray, const unsigned int nsize);
void cu_oddevensort(int *devArray, const unsigned int nsize);

#endif
