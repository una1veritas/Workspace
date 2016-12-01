
/*
 * editdistance.c
 *
 *  Created on: 2016/11/26
 *      Author: sin
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "editdistance.h"

#define DEBUG_DPTABLE

#define min(x, y)  ((x) <= (y)? (x) : (y))
#define max(x, y)  ((x) >= (y)? (x) : (y))

long pow2(long val) {
	long result;
	for (result = 1; result < val; result <<= 1);
	return result;
}

int cuStatCheck(const cudaError_t stat, const char * msg) {
	if (stat != cudaSuccess) {
		fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(stat));
		return -1;
	}
	return 0;
}

long cu_lvdist(long * table, const char t[], const long n, const char p[], const long m) {
	long result = n + m + 1; // an impossible value

	cudaError_t cuStat;

#ifdef DEBUG_DPTABLE
	long r, c, dix;
	// clear table elements
	for (long r = 0; r < (m + 1); ++r)
		for (long c = 0; c < (n + 1); ++c)
			table[(m + 1)*c + r] = -1;
#endif

	char * devt, *devp;
	cuStat = cudaMalloc((void**) &devt, n);
	cuStatCheck(cuStat, "cudaMalloc devt");

	cuStat = cudaMalloc((void**) &devp, m);
	cuStatCheck(cuStat, "cudaMalloc devp");
	cudaMemcpy(devt, t, n, cudaMemcpyHostToDevice);
	cudaMemcpy(devp, p, m, cudaMemcpyHostToDevice);

	const long tablesize = sizeof(long) * (n + 1)*(m + 1);
	long * devtable;
	cuStat = cudaMalloc((void**)&devtable, tablesize);
	cuStatCheck(cuStat, "cudaMalloc devtable failed.\n");
	cudaMemcpy(devtable, table, tablesize , cudaMemcpyHostToDevice);

	fprintf(stdout, "copied input, calling kernel...\n");

	long nthreads = pow2(n+1+m+1);
	fprintf(stdout,"num threads %d, %d grids.\n",nthreads, max(1, nthreads / 1024));
	cu_dptable <<< max(1, nthreads / 1024), 1024 >>> (devtable, devt, n, devp, m);

	// Check for any errors launching the kernel
	cuStat = cudaGetLastError();
	if (cuStat != cudaSuccess) {
		fprintf(stderr, "kernel cu_dptable() failed: %s\n", cudaGetErrorString(cuStat));
	}
	else {
		fprintf(stdout, "returned from kernel.\n");
	}

	cudaMemcpy(table, devtable, tablesize, cudaMemcpyDeviceToHost);
	cudaFree(devtable);

	result = table[(n + 1)*(m + 1) - 1];

#ifdef DEBUG_DPTABLE
	// show DP table
	for (r = 0; r <= m; r++) {
		for (c = 0; c <= min(n, 40); c++) {
			if (c + r <= m) {
				dix = (c + r)*(c + r + 1) / 2 + r;
			}
			else if (c + r <= n) {
				dix = (m + 1)*(m + 2) / 2 + (m + 1)*(c - m - 1 + r) + r;
			}
			else {
				dix = (m + 1)*(m + 2) / 2 + (m + 1)*(c - m - 1 + r) + r - (c + r - n)*(c + r - n + 1) / 2;
			}
			fprintf(stdout, "%3ld ", table[dix]);
		}
		fprintf(stdout, "\n");
	}

#endif

	return result;
}

// assuming the table array size (n+1) x (m+1)
__global__ void cu_dptable(long * table, const char t[], const long n, const char p[], const long m) {
	long dix, dcol, drow; // diagonal column index, row index
	long col; // inner diagonal index
	long ins, del, repl;

	long thix = blockDim.x * blockIdx.x + threadIdx.x ;

	
#ifdef DEBUG_DPTABLE
	if (thix < (n+1) * (m + 1))
		table[thix] = blockDim.x;
	__syncthreads();
#endif

	// initialize
	// do in parallel for each dcol
	// for (dcol = 0; dcol <= n; ++dcol) {
	dcol = thix;
	if (dcol <= m) {
		dix = dcol*(dcol + 1) / 2;
	}
	else if (dcol <= n ) {
		dix = m*(m + 1) / 2 + (m + 1)*(dcol - m);
	}
	if ( dcol <= n )
		table[dix] = dcol;
	__syncthreads();

	
	// do in parallel for each drow
	// for (drow = 1; drow <= m; ++drow) {
	drow = thix;
	if ( drow > 0 && drow <= m ) {
		// m <= n
		dix = (drow + 1)*(drow + 2) / 2 - 1;
		table[dix] = drow;
	}
	__syncthreads();

	return;

	// upper-left triangle
	dix = 1;
	for (dcol = 2; dcol <= m + 1; ++dcol) {
		dix += dcol;
//		for (drow = 1; drow < dcol; ++drow) {
		drow = thix;
		
		if ( drow >= 1 && drow < dcol ) {
			col = dcol - drow;
			ins = table[dix + drow - dcol - 1] + 1;
			del = table[dix + drow - dcol] + 1;
			repl = table[dix + drow - 2 * dcol];
				+ (t[col - 1] == p[drow - 1] ? 0 : 1);
			//fprintf(stdout, " %c=%c? ",t[col-1],p[drow-1] );
			ins = ((ins <= del) ? ins : del); 
			table[dix + drow] = (ins < repl ? ins : repl);
		}	
		__syncthreads();
	}
	
	
	// skewed rectangle
	for (dcol = m + 2; dcol < n + 1; ++dcol) {
		dix += m + 1;
//		for (drow = 1; drow < m + 1; ++drow) {
		drow = thix;
		if ( drow >= 1 && drow < m+1 ) {
			col = dcol - drow;
			ins = table[dix + drow - (m + 1) - 1] + 1;
			del = table[dix + drow - (m + 1)] + 1;
			repl = table[dix + drow - 2 * (m + 1) - 1]
				+ (t[col - 1] == p[drow - 1] ? 0 : 1);
			ins = ((ins <= del) ? ins : del);
			table[dix + drow] = (ins < repl ? ins : repl);
		}
	__syncthreads();
	}

	
	// bottom-right triangle
	for (dcol = n + 1; dcol < n + m + 2; ++dcol) {
		dix += n + m + 1 - dcol;
//		for (drow = dcol - n; drow < m + 1; ++drow) {
		drow = thix;
		if ( drow >= dcol - n && drow < m + 1 ) {
			col = dcol - drow;
			ins = table[dix + drow - (n + m + 2 - dcol)] + 1;
			del = table[dix + drow - (n + m + 2 - dcol) + 1] + 1;
			repl = table[dix + drow - 2 * (n + m + 2 - dcol)]
				+ (t[col - 1] == p[drow - 1] ? 0 : 1);
			//fprintf(stdout, "(%3ld)", n + m + 2 - dcol);
			ins = ((ins <= del) ? ins : del);
			table[dix + drow] = (ins < repl ? ins : repl);
		}
		__syncthreads();
	}
	
	return;
}


long lvdist(long * table, const char t[], const long n, const char p[], const long m) {
	long dix, dcol, drow; // diagonal column index, row index
	long col; // inner diagonal index
	long result = n+m+1; // an impossible large value
	long ins, del, repl;

#ifdef DEBUG_DPTABLE
	// clear table elements
	for(long r = 0; r < (m+1); ++r)
		for(long c = 0; c < (n+1); ++c)
			table[(m+1)*c + r] = -1;
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

