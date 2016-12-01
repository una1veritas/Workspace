
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_timer.h>

#include "textfromfile.h"
#include "editdistance.h"


#define MEGA_B 1048576UL
#define KILO_B 1024UL
#define STR_MAXLENGTH (32 * KILO_B)

long pow2log(long base, long val) {
	long result = base;
	if (base < 2)
		return 0;
	for (result = base; result < val; result <<= 1);
	return result;
}

int main(int argc, const char * argv[]) {
	char * text, *patt;
	long m, n;
	long dist;
	long * table;

	//	stopwatch sw;

	text = (char *)malloc(sizeof(char)*STR_MAXLENGTH);
	patt = (char *)malloc(sizeof(char)*STR_MAXLENGTH);
	if (text == NULL || patt == NULL) {
		fprintf(stderr, "malloc error on the string arrays.\n");
		fflush(stderr);
		goto exit_error;
	}

	if (argc != 3) {
		fprintf(stderr, "Incorrect arguments.\n");
		fprintf(stderr, "Exit the program.\n\n");
		fflush(stderr);
		return EXIT_FAILURE;
	}

	//	getcwd(text, STR_MAXLENGTH);
	//	fprintf(stderr,"Current working directory: \n%s\n", text);
	//	fflush(stderr);

	n = textfromfile(argv[1], STR_MAXLENGTH, text);
	m = textfromfile(argv[2], STR_MAXLENGTH, patt);
	if (n == 0 || m == 0) {
		goto exit_error;
	}
	if (n < m) {
		long t = n;
		n = m;
		m = t;
		char * ptr = text;
		text = patt;
		patt = ptr;
	}
	dist = n + m + 1;

	if (n < 1000 && m < 1000)
		fprintf(stdout, "Input: \n%s (%lu),\n\n%s (%lu)\n\n", text, n, patt, m);
	else
		fprintf(stdout, "Input: (%lu), (%lu)\n\n", n, m);
	fflush(stdout);

	fprintf(stdout, "computing edit distance by DP.\n");
	fflush(stdout);
	//	stopwatch_start(&sw);

	const long N = pow2log(4, n + 1);
	const long M = pow2log(4, m + 1);
	fprintf(stdout, "N = %lu, M = %lu\n\n", N, M);

	table = (long *)malloc(sizeof(long)*N*M);
	if (table == NULL) {
		fprintf(stderr, "malloc failed for DP table.\n");
		fflush(stderr);
		goto exit_error;
	}

	/* Create timer and start the measurment */
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	dist = cu_lvdist(table, text, n, patt, m);

	/* Stop timer and report the duration, delete timer */
	sdkStopTimer(&timer);
	printf("Elapsed time %f(ms)\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

	free(table);

	//	stopwatch_stop(&sw);
	printf("Edit distance (by DP): %l\n", dist);
	//	printf("%lu sec %lu milli %lu micros.\n", stopwatch_secs(&sw), stopwatch_millis(&sw), stopwatch_micros(&sw));

exit_error:
	free(text);
	free(patt);

	return 0;
}

