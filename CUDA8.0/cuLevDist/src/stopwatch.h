/*
 * stopwatch.h
 *
 *  Created on: 2016/11/20
 *      Author: Sin Shimozono
 */

#ifndef SRC_STOPWATCH_H_
#define SRC_STOPWATCH_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

struct stopwatch {
	struct timeval start, stop;
	struct rusage usage;
//	clock_t cpumicros;
//	struct tm * tmstart, *tmstop;
	unsigned long secs, millis, micros;
};

typedef struct stopwatch stopwatch;

void stopwatch_start(stopwatch * w);
void stopwatch_lap(stopwatch * w);
void stopwatch_stop(stopwatch * w);
void stopwatch_reset(stopwatch * w);
unsigned long stopwatch_millis(stopwatch * w);
unsigned long stopwatch_micros(stopwatch * w);
unsigned long stopwatch_secs(stopwatch * w);

#endif /* SRC_STOPWATCH_H_ */
