/*
 * editdistance.h
 *
 *  Created on: 2016/11/26
 *      Author: sin
 */

#ifndef SRC_EDITDISTANCE_H_
#define SRC_EDITDISTANCE_H_

__global__ void cu_dptable(long * dist, const char t[], const long tsize, const char p[], const long psize);
long cu_lvdist(long * dist, const char t[], const long tsize, const char p[], const long psize);

long lvdist(long * dist, const char t[], const long tsize, const char p[], const long msize);


#endif /* SRC_EDITDISTANCE_H_ */
