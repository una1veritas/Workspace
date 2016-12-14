/*
 * culevdist.h
 *
 *  Created on: 2016/11/26
 *      Author: sin
 */

#ifndef SRC_CULEVDIST_H_
#define SRC_CULEVDIST_H_

__global__ void cu_init_row(long * row, const long n, const long offset);
__global__ void cu_dptable(long * weftbuff, const long * inframe, long * outframe, 
	const char t[], const long tsize, const char p[], const long psize, 
	long * devtable);
__global__ void cu_dptable_kernel(long * w2, long * w1, long * w0, const long dcol, const long * inframe, long * outframe,
	const char t[], const long tsize, const char p[], const long psize,
	long * devtable);

long cu_lvdist(long * inbound, long * outbound, const char t[], const long tsize, const char p[], const long psize);

#endif /* SRC_CULEVDIST_H_ */
