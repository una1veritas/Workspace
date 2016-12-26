/*
 * levdist.h
 *
 *  Created on: 2016/12/12
 *      Author: sin
 */

#ifndef SRC_LEVDIST_H_
#define SRC_LEVDIST_H_

#include "debug_table.h"

#define min(x, y)   ((x) > (y)? (y) : (x))
#define max(x, y)   ((x) < (y)? (y) : (x))

long pow2(const long);

long r_edist(char s[], int m, char t[], int n);
long dp_edist(long * table, char t[], long n, char p[], long m);
long weaving_edist(long * frame, const char t[], const long n, const char p[], const long m);
void weaving_setframe(long * frame, const long n, const long m);

#endif /* SRC_LEVDIST_H_ */
