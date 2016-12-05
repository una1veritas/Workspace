
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
#define STR_MAXLENGTH (1024 - 1)


long pow2log(long base, long val) {
	long result = base;
	if (base < 2)
		return 0;
	for (result = base; result <= val; result <<= 1);
	return result;
}

int main(int argc, const char * argv[]) {
	char textbuf[STR_MAXLENGTH], pattbuf[STR_MAXLENGTH];
	char * text, *patt;
	long m, n;
	long dist;
	long frame[4*(STR_MAXLENGTH+1)];

	if (argc != 3) {
		fprintf(stderr, "Incorrect arguments.\n");
		fprintf(stderr, "Exit the program.\n\n");
		fflush(stderr);
		return EXIT_FAILURE;
	}

	n = textfromfile(argv[1], STR_MAXLENGTH, textbuf);
	m = textfromfile(argv[2], STR_MAXLENGTH, pattbuf);
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

	/* Create timer and start the measurment */
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < n + m + 1; ++i) {
		if (i < n + 1) {
			frame[i] = i;
		} else {
			frame[i] = i - n;
		}
	}
	fprintf(stdout, "Input: \n");
	for (int c = 0; c < n + m + 1; ++c) {
		fprintf(stdout, "%3ld ", frame[c]);
		if (c == n)
			fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");

	dist = cu_lvdist(text, n, patt, m, frame);
	 
	/* Stop timer and report the duration, delete timer */
	sdkStopTimer(&timer);
	printf("Elapsed %f msec.\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);
	
	fprintf(stdout, "\nOutput:\n");

	for (int c = 0; c < n + m + 1; ++c) {
		fprintf(stdout, "%3ld ", frame[(n+m+1)+c]);
		if (c == n)
			fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");

	//	stopwatch_stop(&sw);
	fprintf(stdout, "Edit distance (by DP): %ld\n", dist);
	fflush(stdout);

exit_error:
	fprintf(stderr, "task finished.\n");
	fflush(stderr);

	return 0;
}

