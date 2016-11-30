#include <stdio.h>
#include <string.h>

#include "winstopwatch.h"
#include "textfromfile.h"
#include "editdistance.h"

#define MEGA_B 1048576UL
#define KILO_B 1024UL
#define STR_MAXLENGTH (128 * KILO_B)

long pow2log(long base, long val) {
	long result = base;
	if ( base < 2 )
		return 0;
	for (result = base; result < val; result <<= 1 ) ;
	return result;
}

int main (int argc, const char * argv[]) {
	char * text, *patt;
	long m, n;
	long dist;
	long * table;
	
	stopwatch sw;

	text = (char *) malloc(sizeof(char)*STR_MAXLENGTH);
	patt = (char *) malloc(sizeof(char)*STR_MAXLENGTH);
	if ( text == NULL || patt == NULL ) {
		fprintf(stderr, "malloc error.\n");
		fflush(stderr);
		goto exit_error;
	}

	if ( argc != 3 )
		return EXIT_FAILURE;

//	getcwd(text, STR_MAXLENGTH);
//	fprintf(stderr,"Current working directory: \n%s\n", text);
//	fflush(stderr);

	n = textfromfile(argv[1], STR_MAXLENGTH, text);
	m = textfromfile(argv[2], STR_MAXLENGTH, patt);
	if ( n == 0 || m == 0 ) {
		goto exit_error;
	}
	if ( n < m ) {
		long t = n;
		n = m;
		m = t;
		char * ptr = text;
		text = patt;
		patt = ptr;
	}
	dist = n + m + 1;

	if ( n < 1000 && m < 1000 )
		fprintf(stdout, "Input: \n%s (%lu),\n\n%s (%lu)\n\n", text, n, patt, m);
	else
		fprintf(stdout, "Input: (%lu), (%lu)\n\n", n, m);
	fflush(stdout);
	
	fprintf(stdout, "computing edit distance by DP.\n");
	fflush(stdout);
	stopwatch_start(&sw);

	const long N = pow2log(4, n+1);
	const long M = pow2log(4, m+1);
	fprintf(stdout, "N = %lu, M = %lu\n\n", N, M);

	table = (long *) malloc(sizeof(long)*N*M);
	if ( table != NULL) {
		dist = dptable(table, N, M, text, n, patt, m);
		free(table);
	} else {
		fprintf(stderr, "DP table malloc has failed.\n");
		fflush(stderr);
	}
	stopwatch_stop(&sw);
	printf("Edit distance (by DP): %lu\n", dist);
	printf("%lu sec %lu milli %lu micros.\n", stopwatch_secs(&sw), stopwatch_millis(&sw), stopwatch_micros(&sw));
	
exit_error:
	free(text);
	free(patt);

    return 0;
}

