/*
 * editdistance.c
 *
 *  Created on: 2016/11/26
 *      Author: sin
 */

#include <stdio.h>
#include <stdlib.h>

#include "editdistance.h"

//#define DEBUG_DPTABLE

#define min(x, y)  ((x) <= (y)? (x) : (y))

// assuming the table array size (n+1) x (m+1)
long dptable(long * table, const long colsize, const long rowsize, const char t[], const long n, const char p[], const long m) {
	long dix, dcol, drow; // diagonal column index, row index
	long col; // inner diagonal index
	long result = n+m+1; // an impossible large value
	long ins, del, repl;

#ifdef DEBUG_DPTABLE
	// clear table elements
	for(long r = 0; r < rowsize; ++r)
		for(long c = 0; c < colsize; ++c)
			table[rowsize*c + r] = -1;
#endif

	// initialize
	// do in parallel for each dcol
	for(dcol = 0; dcol <= n; ++dcol) {
		if ( dcol <= m ) {
			dix = dcol*(dcol+1)/2;
		} else {
			dix = m*(m+1)/2+(m+1)*(dcol-m);
		}
		table[dix] = dcol;
	}
	// do in parallel for each drow
	for(drow = 1; drow <= m; ++drow) {
		// m <= n
		dix = (drow+1)*(drow+2)/2 - 1;
		table[dix] = drow;
	}

	// upper-left triangle
	dix = 1;
	for(dcol = 2; dcol <= m + 1; ++dcol) {
		dix += dcol;
		for(drow = 1; drow < dcol; ++drow) {
			col = dcol - drow;
#ifdef DEBUG_DPTABLE
			fprintf(stdout, "%3ld:%3ld [%2ld, %2ld] ", dcol, dix+drow, col, drow);
#endif
			ins = table[ dix + drow - dcol - 1 ] + 1;
			del = table[ dix + drow - dcol ] + 1;
			repl = table[ dix + drow - 2*dcol]
						  + (t[col-1] == p[drow-1] ? 0 : 1);
			//fprintf(stdout, " %c=%c? ",t[col-1],p[drow-1] );
			ins = ((ins <= del) ? ins : del);
			table[dix+drow]  = (ins < repl ? ins : repl);
		}
#ifdef DEBUG_DPTABLE
		fprintf(stdout, "\n");
#endif
	}
#ifdef DEBUG_DPTABLE
	fprintf(stdout, "\n");
#endif

	// skewed rectangle
	for(dcol = m+2; dcol < n + 1;++dcol) {
		dix += m + 1;
		for(drow = 1; drow < m + 1; ++drow) {
			col = dcol - drow;
#ifdef DEBUG_DPTABLE
			fprintf(stdout, "%3ld:%3ld [%2ld, %2ld] ", dcol, dix+drow, col, drow);
			//fprintf(stdout, " %c=%c? ",t[col-1],p[drow-1] );
#endif
			ins = table[ dix + drow - (m+1) - 1] + 1;
			del = table[ dix + drow - (m+1) ] + 1;
			repl = table[ dix + drow - 2 *(m+1) - 1 ]
						  + (t[col-1] == p[drow-1] ? 0 : 1);
			ins = ((ins <= del) ? ins : del);
			table[ dix + drow ]  = (ins < repl ? ins : repl);
#ifdef DEBUG_DPTABLE
			//table[(dcol + 2)*rowsize + (drow + 1)] = table[ ((dcol + 2)% (n+2))*rowsize + (drow + 1)];
#endif
			}
#ifdef DEBUG_DPTABLE
		fprintf(stdout, "\n");
#endif
	}
#ifdef DEBUG_DPTABLE
	fprintf(stdout, "\n");
#endif

	// bottom-right triangle
	for(dcol = n+1; dcol < n + m + 2; ++dcol) {
		dix += n + m + 1 - dcol;
		for(drow = dcol - n; drow < m+1; ++drow) {
			col = dcol - drow;
#ifdef DEBUG_DPTABLE
			fprintf(stdout, "%3ld:%3ld [%2ld, %2ld] ", dcol, dix+drow, col, drow);
#endif
			ins = table[ dix + drow - (n + m + 2 - dcol) ] + 1;
			del = table[ dix + drow - (n + m + 2 - dcol) + 1 ] + 1;
			repl = table[ dix + drow - 2*(n + m + 2 - dcol) ]
						  + (t[col-1] == p[drow-1] ? 0 : 1);
			//fprintf(stdout, "(%3ld)", n + m + 2 - dcol);
			ins = ((ins <= del) ? ins : del);
			table[ dix + drow ]  = (ins < repl ? ins : repl);
#ifdef DEBUG_DPTABLE
			//table[(dcol + 2)*rowsize + (drow + 1)] = table[ ((dcol + 2)% (n+2))*rowsize + (drow + 1)];
#endif
			}
#ifdef DEBUG_DPTABLE
		fprintf(stdout, "\n");
#endif
	}
#ifdef DEBUG_DPTABLE
	fprintf(stdout, "\n");
#endif

#ifdef DEBUG_DPTABLE
	// show DP table
	long r, c;
	for(r = 0; r <= m; r++) {
		for (c = 0; c <= n; c++) {
			if ( c + r <= m ) {
				dix = (c + r)*(c + r + 1)/2 + r;
			} else if (c + r <= n) {
				dix = (m+1)*(m+2)/2 + (m+1)*(c - m - 1 + r) + r;
			} else {
				dix = (m+1)*(m+2)/2 + (m+1)*(c - m - 1 + r) + r - (c + r - n)*(c + r - n + 1)/2;
			}
			fprintf(stdout, "%3ld\t", table[dix]);
		}
		fprintf(stdout, "\n");
	}

	fprintf(stdout, "[%3ld], ", (n+1)*(m+1) - 1 );
#endif
	result = table[ (n+1)*(m+1) - 1 ];
	return result;
}

