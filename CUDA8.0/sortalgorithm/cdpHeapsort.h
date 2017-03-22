#pragma once
#ifndef __CDPHEAPSORT_HEADER__
#define __CDPHEAPSORT_HEADER__

__global__ void dev_shakeheap(int *A, const unsigned int nsize, const unsigned int parity);

void cu_makeheap(int *devArray, const unsigned int nsize);

#endif
