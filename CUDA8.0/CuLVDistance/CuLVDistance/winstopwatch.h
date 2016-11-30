#ifndef __WINSTOPWATCH_H__
#define __WINSTOPWATCH_H__

/* winstopwatch.h */

#include <windows.h>

struct stopwatch {
	DWORD start, stop;
};
typedef struct stopwatch stopwatch;

void stopwatch_start(stopwatch *);
void stopwatch_stop(stopwatch *);
long stopwatch_secs(stopwatch *);
long stopwatch_millis(stopwatch *);
long stopwatch_micros(stopwatch *);

#endif
