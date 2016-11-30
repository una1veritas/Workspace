#include "winstopwatch.h"

void stopwatch_start(stopwatch * sw) {
	sw->start = timeGetTime();
}

void stopwatch_stop(stopwatch * sw) {
	sw->stop = timeGetTime();
}

long stopwatch_secs(stopwatch * sw) {
	return (long) (sw->stop - sw->start);
}

long stopwatch_millis(stopwatch * sw) {
	return (long) (sw->stop - sw->start);
}

long stopwatch_micros(stopwatch * sw) {
	return (long) (sw->stop - sw->start);
}