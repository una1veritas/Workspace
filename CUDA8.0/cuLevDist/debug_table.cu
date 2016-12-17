#include <stdio.h>
#include <stdlib.h>

#include "debug_table.h"

long * debug_table;

#ifndef max
#define min(x, y)   ((x) > (y)? (y) : (x))
#define max(x, y)   ((x) < (y)? (y) : (x))
#endif

void show_table(long * table, long n, long m) {
	static const char grays[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
	static const int grayscale = 62;

	if (n > 1024 || m > 1024) {
		printf("\ntable seems being too big.\n");
		return;
	}
	// show DP table
	printf("\ntable contents:\n");
	for (long r = 0; r < m; r++) {
		for (long c = 0; c < n; c++) {
			//printf("%c", grays[max(0, 61 - (int)((table[m*c + r] / (float)(n))*grayscale))]);
			printf("%3ld ", table[m*c+r]);
		}
		printf("\n");
		fflush(stdout);
	}
	printf("\n\n");
	fflush(stdout);
}

long compare_table(long * t0, long * t1, long n, long m) {
	long c, r;
	long count = 0;
	for (c = 0; c < n; c++) {
		for (r = 0; r < m; r++) {
			if (t0[m*c + r] != t1[m*c + r]) {
				count++;
				printf("different @ %ld, %ld\n", c, r);
			}
		}
	}
	if (count > 0)
		printf("%ld different elements in table.\n", count);
	printf("\n");
	return count;
}
