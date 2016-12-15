/*
 * culevdist.h
 *
 *  Created on: 2016/11/26
 *      Author: sin
 */

#ifndef SRC_CULEVDIST_H_
#define SRC_CULEVDIST_H_

long pow2(const long val);

long wv_edist(long * frame, const char t[], const long n, const char p[], const long m);
void wv_setframe(long * frame, const char t[], const long n, const char p[], const long m);
long cu_levdist(long * frame, const char t[], const long tsize, const char p[], const long psize);

#endif /* SRC_CULEVDIST_H_ */
