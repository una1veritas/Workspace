#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
//#include <unistd.h>

#include "cu_utils.h"
#include "levdist.h"
#include "cu_levdist.h"
//#include "stopwatch.h"
#include "textfromfile.h"

#include "debug_table.h"

#define MEGA_B 1048576UL
#define KILO_B 1024UL
#define STR_MAXLENGTH (1 * KILO_B)

int getargs(const int argc, const char * argv[], char * text, char * patt, long * n, long *m) {
	if ( argc != 3 )
		return EXIT_FAILURE;

	text[STR_MAXLENGTH - 1] = 0;
	patt[STR_MAXLENGTH - 1] = 0;

	if ( textfromfile(argv[1], STR_MAXLENGTH, text) != 0
		|| textfromfile(argv[2], STR_MAXLENGTH, patt) != 0 ) {
		return EXIT_FAILURE;
	}
	*n = (text[STR_MAXLENGTH-1] == 0? strlen(text) : STR_MAXLENGTH);
	*m = (patt[STR_MAXLENGTH-1] == 0? strlen(patt) : STR_MAXLENGTH);
	if ( *n < *m ) {
		char * tmp = text;
		text = patt;
		patt = tmp;
		long t = *n;
		*n = *m;
		*m = t;
	}

	printf("%s\n", patt);
	if ( *n < 1000 && *m < 1000 )
		fprintf(stdout, "Input: %s \n(%lu), \n%s \n(%lu)\n\n", text, *n, patt, *m);
	else
		fprintf(stdout, "Input: (%lu), (%lu)\n\n", *n, *m);
	fflush(stdout);

	return 0;
}

int main (int argc, const char * argv[]) {
	char * text, *patt;
	long * table;
	long m, n;
	long d;

	cudaSetDevice(0);
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 2);

	text = (char*) malloc(sizeof(char)*STR_MAXLENGTH);
	patt = (char*) malloc(sizeof(char)*STR_MAXLENGTH);
	if ( text == NULL || patt == NULL ) {
		fprintf(stderr, "malloc error.\n");
		fflush(stderr);
		goto exit_error;
	}


	if ( getargs(argc, argv, text, patt, &n, &m) != 0 )
		goto exit_error;

/*
	stopwatch sw;
	stopwatch_start(&sw);
	*/
	table = (long*) malloc(sizeof(long)*m*n);

	d = dp_edist(table, text, n, patt, m);

#ifndef DEBUG_TABLE
	free(table);
#endif
	/*
	stopwatch_stop(&sw);

	printf("%lu sec %lu milli %lu micros.\n", stopwatch_secs(&sw), stopwatch_millis(&sw), stopwatch_micros(&sw));
	printf("\n");
*/
	printf("Edit distance (by Pure DP): %lu\n", d);
#ifdef DEBUG_TABLE
	show_table(table, n, m);

	debug_table = (long*) malloc(sizeof(long)*m*n);
#endif

	fprintf(stdout, "computing edit distance by Waving DP.\n");
	fflush(stdout);

	long * frame = (long*)malloc(sizeof(long)*pow2(m + n + 1));
	wv_setframe(frame, n, m);
	printf("frame input: \n");
	for (int i = 0; i < pow2(n + m + 1); i++) {
		printf("%d, ", frame[i]);
	}
	printf("\n");
	fflush(stdout);

		/*
	stopwatch_start(&sw);
	*/
	d = cu_levdist(frame, text, n, patt, m);

	/*
	stopwatch_stop(&sw);
	printf("%lu sec %lu milli %lu micros.\n", stopwatch_secs(&sw), stopwatch_millis(&sw), stopwatch_micros(&sw));
	*/
	printf("Edit distance (by Weaving DP): %lu\n\n", d);
	printf("frame output: \n");
	for (int i = 0; i < pow2(n + m + 1); i++) {
		printf("%d, ", frame[i]);
	}
	printf("\n");
	fflush(stdout);
	free(frame);

#ifdef DEBUG_TABLE
	show_table(debug_table, n, m);

	if ( compare_table(debug_table, table, n, m) != 0) {
		printf("table compare failed.\n");
	} else {
		printf("two tables are identical.\n");
	}
	free(debug_table);
#endif

	free(table);

exit_error:
	free(text);
	free(patt);

    return 0;
}

