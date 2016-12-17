/*
 * stopwatch.c
 *
 *  Created on: 2016/11/20
 *      Author: Sin Shimozono
 */

#include "stopwatch.h"

void stopwatch_start(stopwatch * w) {
	w->secs = 0;
	w->millis = 0;
	w->micros = 0;
	getrusage(RUSAGE_SELF, &w->usage);
	w->start = w->usage.ru_utime;
}

void stopwatch_stop(stopwatch * w) {
	getrusage(RUSAGE_SELF, &w->usage);
	w->stop = w->usage.ru_utime;
	w->secs = w->stop.tv_sec - w->start.tv_sec;
	w->micros = w->stop.tv_usec - w->start.tv_usec;
	w->millis = w->micros / 1000;
	w->micros %= 1000;
	w->millis %= 1000;
}

void stopwatch_lap(stopwatch * w) {
	stopwatch_stop(w);
}

void stopwatch_reset(stopwatch * w) {
	stopwatch_start(w);
}

unsigned long stopwatch_secs(stopwatch * w) {
	return w->secs;
}

unsigned long stopwatch_millis(stopwatch * w) {
	return w->millis;
}

unsigned long stopwatch_micros(stopwatch * w) {
	return w->micros;
}
