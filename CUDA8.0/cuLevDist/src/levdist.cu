/*
 * levdist.c
 *
 *  Created on: 2016/12/12
 *      Author: sin
 */
#include <stdio.h>
#include <stdlib.h>

#include "cu_utils.h"
#include "levdist.h"

long pow2(const long val) {
	long result = 1;
	for (result = 1; result < val; result <<= 1);
	return result;
}

long r_edist(char s[], int m, char t[], int n) {
	long a, b, c;
	if (m == 0 && n == 0)
		return 0;
	if (m == 0)
		return n;
	if (n == 0)
		return m;
	a = r_edist(s, m, t, n-1) + 1;
	b = r_edist(s, m-1, t, n) + 1;
	c = r_edist(s, m-1, t, n-1) + ((s[m-1] == t[n-1])? 0 : 1);
	return (a < b ? (a < c ? a: c): (b < c ? b : c));
}

long dp_edist(long * dist, char t[], long n, char p[], long m) {
	long ins, del, repl;

	if ( dist == NULL )
		return n+m+1;

	// initialize cells in the top row or in the left-most column
	// n -- the number of columns, m -- the number of rows
	long col = 0;
	long row = 0;

	//[0, 0]
	ins = col + 1; // always == del
	repl = (p[0] == t[col] ? 0 : 1);
	dist[0] = (ins < repl ? ins : repl);

	//[col, 0]
	for(col = 1; col < n; ++col) {
		// row == 0
		del = (col + 1) + 1;
		ins = dist[m*(col-1)] + 1;
		repl = col + (p[0] == t[col] ? 0 : 1);
		ins = (ins < del ? ins : del);
		dist[m * col + 0] = (ins < repl ? ins : repl);
	}
	//[0, row]
	for(row = 1; row < m; ++row) {
		// col == 0
		del = dist[row - 1] + 1;
		ins = row + 1 + 1;
		repl = row + (p[row] == t[0] ? 0 : 1);
		ins = (ins < del ? ins : del);
		dist[m*0 + row]  = (ins < repl ? ins : repl);
	}

	//table calcuration
	for(long c = 1; c < n; c++) { // column, text axis
		for (long r = 1; r < m; r++) {  // row, pattern axis
			del = dist[(r-1) + m*c]+1;
			ins = dist[r + m*(c-1)]+1;
			repl = dist[(r-1) + m*(c-1)] + (t[c] == p[r] ? 0 : 1);
			dist[r + m*c] = ins < del ? (ins < repl ? ins : repl) : (del < repl ? del : repl);
		}
	}

	return dist[n * m - 1];
}

void weaving_setframe(long * frame, const long n, const long m) {
	for (long i = 0; i < n + m + 1; i++) {
		if (i < m) {
			frame[i] = m - i;
		}
		else {
			frame[i] = i - m;  // will be untouched.
		}
	}
}

long weaving_edist(long * frame, const char t[], const long n, const char p[], const long m
#ifdef DEBUG_TABLE
	, long * table
#endif
) {
	long col, row;
	long del, ins, repl, cellval; // del = delete from pattern, downward; ins = insert to pattern, rightward
	long warpix, warp_start, warp_last;

	if (frame == NULL)
		return n+m+1;

	for (long depth = 0; depth <= (n - 1) + (m - 1); depth++) {
		warp_start = abs((m - 1) - depth);
		if (depth < n) {
			warp_last = depth + (m - 1);
		}
		else {
			warp_last = ((n - 1) << 1) + (m - 1) - depth;
		}
		// mywarpix = (thix<<1) + (depth & 1);
		//printf("depth %ld [%ld, %ld]: warpix ", depth, warp_start, warp_last);
		for (long warpix = warp_start; warpix <= warp_last; warpix += 2) {
			if (warpix < 0 || warpix > n + m + 1) {
				printf("warp value error: %ld\n", warpix);
				//fflush(stdout);
			}
			col = (depth + warpix - (m - 1)) >> 1;
			row = (depth - warpix + (m - 1)) >> 1;

			//printf("%ld = (%ld, %ld), ", warpix, col, row);
			//
			del = frame[warpix + 1 + 1] + 1;
			ins = frame[warpix - 1 + 1] + 1;
			repl = frame[warpix + 1] + (t[col] != p[row]);
			//printf("%ld: %ld [%ld,%ld] %c|%c : %ld/%ld/%ld+%ld,\n",depth, warpix, col,row,t[col],p[row], del,ins, frame[warpix], (t[col] != p[row]));
			//
			if (del < ins) {
				ins = del;
			}
			if (ins < repl) {
				repl = ins;
			}
			//
			frame[warpix + 1] = repl;
#ifdef DEBUG_TABLE
			table[m*col + row] = repl;
#endif

		}
		//printf("\n");
	}
	return frame[n];
}

