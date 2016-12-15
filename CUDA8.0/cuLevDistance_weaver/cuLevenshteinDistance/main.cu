
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_timer.h>

#include "textfromfile.h"
#include "culevdist.h"
#include "dpedist.h"

#include "debug_table.h"
#include "cudautils.h"

#define MEGA_B 1048576UL
#define KILO_B 1024UL
#define STR_MAXLENGTH (64*KILO_B)

long * debug_table;

int main(int argc, const char * argv[]) {
	char textbuf[STR_MAXLENGTH+1], pattbuf[STR_MAXLENGTH+1];
	char * text, *patt;
	long m, n;
	long dist;

	long * frame = (long*) malloc(sizeof(long)*pow2(2 * STR_MAXLENGTH + 1));

	long * dp_table, *cu_table;

	if (frame == NULL ) {
		fprintf(stderr, "Allocation for inbound/outbound failed.\n");
		fprintf(stderr, "Exit.\n\n");
		fflush(stderr);
		return EXIT_FAILURE;
	}

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
		fprintf(stderr, "Empty input.\n\n");
		fflush(stderr);
		return EXIT_FAILURE;
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
		fprintf(stdout, "Input: \n%s \n(%lu),\n%s \n(%lu)\n", text, n, patt, m);
	else
		fprintf(stdout, "Input: (%lu), (%lu)\n\n", n, m);
	fflush(stdout);

	fprintf(stdout, "Computing edit distance by DP.\n");
	fflush(stdout);
	//	stopwatch_start(&sw);

#ifdef DEBUG_TABLE
	debug_table = (long*)malloc(sizeof(long)*(n + 1)*(m + 1));
	if (debug_table == NULL) {
		fprintf(stdout, "debug_table allocation failed.\n");
		fflush(stdout);
		return EXIT_FAILURE;
	}
	cu_table = (long*)malloc(sizeof(long)*(n + 1)*(m + 1));
	if (debug_table == NULL) {
		fprintf(stdout, "cu_table allocation failed.\n");
		fflush(stdout);
		return EXIT_FAILURE;
	}
	dp_table = (long*)malloc(sizeof(long)*(n + 1)*(m + 1));
	if (debug_table == NULL) {
		fprintf(stdout, "dp_table allocation failed.\n");
		fflush(stdout);
		return EXIT_FAILURE;
	}
#else
	debug_table = NULL;
#endif


	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	dist = dp_edist(text,n,patt,m);

	//	stopwatch_stop(&sw);
	sdkStopTimer(&timer);
	printf("\nElapsed %f msec.\n", sdkGetTimerValue(&timer));

	fprintf(stdout, "Edit distance (by DP/CPU): %ld\n\n", dist);
	fflush(stdout);

#ifdef DEBUG_TABLE
	for (int r = 0; r < m + 1; r++) {
		for (int c = 0; c < n + 1; c++) {
			dp_table[(m + 1)*c + r] = debug_table[(m + 1)*c + r];
			debug_table[(m + 1)*c + r] = -1;
			if (r < 20 || r > m + 1 - 8) {
				if ((c < 18) || (c > n + 1 - 8)) {
					fprintf(stdout, "%4ld ", dp_table[(m + 1)*c + r]);
				}
				else if (c == 18) {
					fprintf(stdout, "... ");
				}
			}
		}
		if (r < 20 || r > m + 1 - 8) {
			fprintf(stdout, "\n");
		}
		else if (r == 20) {
			fprintf(stdout, "... \n");
		}
	}
	fprintf(stdout, "\n");
	fflush(stdout);
#endif

	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	// setup input frame
	for (int i = 0; i < n + m + 1; ++i) {
		if (i < m + 1) {
			inbound[i] = m - i;
		} else {
			inbound[i] = i - m;
		}
	}

	fprintf(stdout, "Inframe: \n");
	for (int c = 0; c < (n + m + 1 > 32 ? 32 : n + m + 1); ++c) {
		fprintf(stdout, "%3ld ", inbound[c]);
		if (c == m - 1)
			fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");

	dist = cu_lvdist(frame, text, n, patt, m);
	 
	/* Stop timer and report the duration, delete timer */
	sdkStopTimer(&timer);
	printf("\nElapsed %f msec.\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

	fprintf(stdout, "\nOutframe:\n");
	for (int c = 0; c < (n + m + 1 > 32 ? 32 : n + m + 1); ++c) {
		fprintf(stdout, "%3ld ", outbound[c]);
		if (c == n)
			fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");

	fprintf(stdout, "\nEdit distance (by DP/GPU): %ld\n\n", dist);
	fflush(stdout);

#ifdef DEBUG_TABLE
	// debug table 
	for (int r = 0; r < m + 1; r++) {
		for (int c = 0; c < n + 1; c++) {
			cu_table[(m + 1)*c + r] = debug_table[(m + 1)*c + r];
			if (r < 20 || r > m + 1 - 8) {
				if ((c < 18) || (c > n + 1 - 8)) {
					fprintf(stdout, "%4ld ", cu_table[(m + 1)*c + r]);
				}
				else if (c == 18) {
					fprintf(stdout, "... ");
				}
			}
		}
		if (r < 20 || r > m + 1 - 8) {
			fprintf(stdout, "\n");
		}
		else if (r == 20) {
			fprintf(stdout, "... \n");
		}
	}
	fprintf(stdout, "\n");
	fflush(stdout);

	fprintf(stdout, "Check table...");
	int errflag = 0;
	for (int r = 0; r < m + 1 && errflag < 8; r++) {
		for (int c = 0; c < n + 1; c++) {
			if (cu_table[(m + 1)*c + r] != dp_table[(m + 1)*c + r] && errflag < 8) {
				fprintf(stdout, "!!!diff at col = %d row = %d, %d and %d!!!\n",c,r, dp_table[(m + 1)*c + r], cu_table[(m + 1)*c + r]);
				errflag++;
			}
		}
	}
	fprintf(stdout, "done.\n\n");
	fflush(stdout);

	free(dp_table);
	free(cu_table);
	free(debug_table);
#endif
	free(inbound);
	free(outbound);
	return 0;
}

