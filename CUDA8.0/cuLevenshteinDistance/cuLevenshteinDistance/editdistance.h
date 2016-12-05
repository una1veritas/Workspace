/*
 * editdistance.h
 *
 *  Created on: 2016/11/26
 *      Author: sin
 */

#ifndef SRC_EDITDISTANCE_H_
#define SRC_EDITDISTANCE_H_

__global__ void cu_init_row(long * row, const long n, const long offset);
__global__ void cu_dptable(long * wavebuff, 
	long * frame, const char t[], const long tsize, const char p[], const long psize, 
	long * devtable);

long cu_lvdist(long * inbound, long * outbound, const char t[], const long tsize, const char p[], const long psize);

long lvdist(long * dist, long * boundary, const char t[], const long tsize, const char p[], const long msize);


#endif /* SRC_EDITDISTANCE_H_ */
