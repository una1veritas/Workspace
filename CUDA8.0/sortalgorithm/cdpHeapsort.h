#pragma once
#ifndef __CDPHEAPSORT_HEADER__
#define __CDPHEAPSORT_HEADER__

__global__ void dev_shakeheap(int *a, const unsigned int nsize, const unsigned int parity);
__global__ void dev_swapheaptop(int * a, const unsigned int heapsize);

void cu_makeheap(int *devArray, const unsigned int nsize);

void cu_heapsort(int * devArray, const unsigned int nsize);


#endif
