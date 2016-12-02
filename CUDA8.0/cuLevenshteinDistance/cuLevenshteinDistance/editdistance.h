/*
 * editdistance.h
 *
 *  Created on: 2016/11/26
 *      Author: sin
 */

#ifndef SRC_EDITDISTANCE_H_
#define SRC_EDITDISTANCE_H_

__global__ void cu_dptable_by1block(long * dist, const char t[], const long tsize, const char p[], const long psize);
long cu_lvdist(long * dist, const char t[], const long tsize, const char p[], const long psize);

long lvdist(long * dist, const char t[], const long tsize, const char p[], const long msize);

__global__ void cu_dptable_init(long * table, const char t[], const long n, const char p[], const long m);
__global__ void cu_dptable_topleft(long * table, const char t[], const long n, const char p[], const long m, long dix, long dcol);
__global__ void cu_dptable_center(long * table, const char t[], const long n, const char p[], const long m, const long dix, const long dcol);
__global__ void cu_dptable_bottomright(long * table, const char t[], const long n, const char p[], const long m, const long dix, const long dcol);

#endif /* SRC_EDITDISTANCE_H_ */
