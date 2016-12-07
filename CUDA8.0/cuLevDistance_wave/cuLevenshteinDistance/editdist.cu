#include <stdio.h>
#include <stdlib.h>

#define DEBUG_TABLE

long dp_edist(char t[], long n, char p[], long m) {
	long * dist;
	long result = 0;
	long ins, del, repl;
	
	dist = (long *) malloc(sizeof(long)*m*n);
	if ( dist == NULL )
		return 0;
	
	// initialize cells in the top row or in the left-most column
	// n -- the number of columns, m -- the number of rows
	long col = 0;
	long row = 0;
	ins = col + 1;
	repl = (p[0] == t[col] ? 0 : 1);
	dist[0] = (ins < repl ? ins : repl);

	for(col = 1; col < n; ++col) {
		// row == 0
		ins = col + 1;
		repl = col - 1 + (p[0] == t[col] ? 0 : 1);
		dist[0 + m * col] = (ins < repl ? ins : repl);
	}
	for(row = 1; row < m; ++row) {
		// col == 0
		del = row + 1;
		repl = row - 1 + (p[row] == t[0] ? 0 : 1);
		dist[row + 0]  = (del < repl ? del : repl);
	}

	//table calcuration
	for(long c = 1; c < n; c++) { // column, text axis
		for (long r = 1; r < m; r++) {  // row, pattern axis
			ins = dist[(r-1) + m*c]+1;
			del = dist[r + m*(c-1)]+1;
			repl = dist[(r-1) + m*(c-1)] + (t[c] == p[r] ? 0 : 1);
			dist[r + m*c] = ins < del ? (ins < repl ? ins : repl) : (del < repl ? del : repl);
		}
	}

#ifdef DEBUG_TABLE
	// show DP table
	for(long r = 0; r < m; r++) {
		for (long c = 0; c < n; c++) {
			fprintf(stdout, "%3ld ", dist[m*c+r]);
		}
		fprintf(stdout, "\n");
	}
#endif
	fprintf(stdout, "\n");

	result = dist[n * m - 1];
	free(dist);
	return result;
}
