

#include "custopwatch.h"

void stopwatch_start(stopwatch * sw) {
	/* 時間計測用タイマーのセットアップと計時開始 */
	sw->timer = NULL;
	sdkCreateTimer(&sw->timer);
	sdkResetTimer(&sw->timer);
	sdkStartTimer(&sw->timer);
}

void stopwatch_stop(stopwatch * sw) {
	sdkStopTimer(&sw->timer);
	elapsed = sdkGetTimerValue(&sw->timer);
	sdkDeleteTimer(&sw->timer);
}

long stopwatch_secs(stopwatch * sw) {
	return elapsed;
}

long stopwatch_millis(stopwatch * sw) {
	return ((long) (elapsed*1000)) % 1000;
}

long stopwatch_micros(stopwatch * sw) {
	return (long) 0;
}