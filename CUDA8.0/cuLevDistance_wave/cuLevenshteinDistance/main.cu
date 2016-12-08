
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_timer.h>

#include "textfromfile.h"
#include "culevdist.h"
#include "editdist.h"


#define MEGA_B 1048576UL
#define KILO_B 1024UL
#define STR_MAXLENGTH (500)


long pow2log(long base, long val) {
	long result = base;
	if (base < 2)
		return 0;
	for (result = base; result <= val; result <<= 1);
	return result;
}

int main(int argc, const char * argv[]) {
	char textbuf[STR_MAXLENGTH+1], pattbuf[STR_MAXLENGTH+1];
	char * text, *patt;
	long m, n;
	long dist;
	long inbound[2*STR_MAXLENGTH+1], outbound[2*STR_MAXLENGTH+1];

	if (argc != 3) {
		fprintf(stderr, "Incorrect arguments.\n");
		fprintf(stderr, "Exit the program.\n\n");
		fflush(stderr);
		return EXIT_FAILURE;
	}

	n = textfromfile(argv[1], STR_MAXLENGTH, textbuf);
	textbuf[STR_MAXLENGTH] = 0;
	m = textfromfile(argv[2], STR_MAXLENGTH, pattbuf);
	pattbuf[STR_MAXLENGTH] = 0;
	if (n == 0 || m == 0) {
		goto exit_error;
	}
	if (n < m) {
		long t = n;
		n = m;
		m = t;
		text = pattbuf;
		patt = textbuf;
	}
	else {
		text = textbuf;
		patt = pattbuf;
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

	fprintf(stdout, "n = %lu, m = %lu\n\n", n, m);
	fflush(stdout);

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);

	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	dist = dp_edist(text,n,patt,m);

	//	stopwatch_stop(&sw);
	sdkStopTimer(&timer);
	printf("Elapsed %f msec.\n", sdkGetTimerValue(&timer));

	fprintf(stdout, "Edit distance (by DP/CPU): %ld\n", dist);
	fflush(stdout);

	/* Create timer and start the measurment */
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < n + m + 1; ++i) {
		if (i < n + 1) {
			inbound[i] = i;
		} else {
			inbound[i] = i - n;
		}
	}
	fprintf(stdout, "Input: \n");
	for (int c = 0; c < (n + m + 1 > 32 ? 32 : n + m + 1); ++c) {
		fprintf(stdout, "%3ld ", inbound[c]);
		if (c == n)
			fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");

	dist = cu_lvdist(inbound, outbound, text, n, patt, m);
	 
	/* Stop timer and report the duration, delete timer */
	sdkStopTimer(&timer);
	printf("Elapsed %f msec.\n", sdkGetTimerValue(&timer));

	sdkDeleteTimer(&timer);
	
	fprintf(stdout, "\nOutput:\n");

	for (int c = 0; c < (n + m + 1 > 32 ? 32 : n + m + 1); ++c) {
		fprintf(stdout, "%3ld ", outbound[c]);
		if (c == n)
			fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");

	//	stopwatch_stop(&sw);
	fprintf(stdout, "Edit distance (by DP/GPU): %ld\n", dist);
	fflush(stdout);

exit_error:
	fprintf(stderr, "task finished.\n");
	fflush(stderr);

	return 0;
}

