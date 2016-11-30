#ifndef __WINSTOPWATCH_H__
#define __WINSTOPWATCH_H__

/* custopwatch.h */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>


#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_timer.h>


struct stopwatch {
	StopWatchInterface * timer;
	double elapsed;
};
typedef struct stopwatch stopwatch;

void stopwatch_start(stopwatch *);
void stopwatch_stop(stopwatch *);
long stopwatch_secs(stopwatch *);
long stopwatch_millis(stopwatch *);
long stopwatch_micros(stopwatch *);

#endif
