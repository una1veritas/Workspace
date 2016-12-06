
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
#include <string.h>

#define DEBUG_DPTABLE

#include "editdistance.h"

#define MAX_THREADSPERBLOCK 1024

#define min(x, y)  ((x) <= (y)? (x) : (y))
#define max(x, y)  ((x) >= (y)? (x) : (y))
#define align(base, val)    ((((val)+(base)-1)/(base))*(base))

static char grayscale[] = "@#$B%8&WM*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>ilI;!:,\"^`'. ";

int cuStatCheck(const cudaError_t stat, const char * msg) {
	if (stat != cudaSuccess) {
		fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(stat));
		fflush(stderr);
		return -1;
	}
	return 0;
}


long cu_lvdist(long * inbound, long * outbound, const char t[], const long n, const char p[], const long m) {
	long result = n + m + 1; // an impossible value

	cudaError_t cuStat;

	char * devt, *devp;
	long * devwavebuff, *devframe;

	cuStat = cudaMalloc((void**) &devt, n);
	cudaMemcpy(devt, t, n, cudaMemcpyHostToDevice);
	cuStatCheck(cuStat, "cudaMalloc devt");
	cuStat = cudaMalloc((void**) &devp, m);
	cudaMemcpy(devp, p, m, cudaMemcpyHostToDevice);
	cuStatCheck(cuStat, "cudaMalloc devp");


	long * devweftbuff, * devinframe, *devoutframe;
	const long table_height = m + 1;
	const long table_width = n + m + 1;

	cuStat = cudaMalloc((void**)&devweftbuff, sizeof(long)*table_height*4);
	cuStatCheck(cuStat, "cudaMalloc devtable failed.\n");

	long * dptable;
	long * devtable;
#ifdef DEBUG_DPTABLE
	cuStat = cudaMalloc((void**)&devtable, sizeof(long)*table_height*table_width);
	cuStatCheck(cuStat, "cudaMalloc devtable failed.\n");
#endif
	cuStat = cudaMalloc((void**)&devinframe, sizeof(long)*(n + m + 1));
	cuStatCheck(cuStat, "cudaMalloc devinframe failed.\n");
	cuStat = cudaMalloc((void**)&devoutframe, sizeof(long)*(n + 1 + m));
	cuStatCheck(cuStat, "cudaMalloc devoutframe failed.\n");
	cudaMemcpy(devinframe, inbound, sizeof(long)*(n + m + 1), cudaMemcpyHostToDevice);

	fprintf(stdout, "copied input, calling kernel...\n");
	fflush(stdout);

	long nthreads = align(32, m+1);
	dim3 grids(max(1, nthreads / MAX_THREADSPERBLOCK), 1), blocks(MAX_THREADSPERBLOCK);
	fprintf(stdout,"num threads %d, %d blocks.\n",nthreads, max(1, nthreads / MAX_THREADSPERBLOCK));

	cu_dptable<<< grids, blocks >>>(devweftbuff, devinframe, devoutframe, devt, n, devp, m, devtable);

	// Check for any errors launching the kernel
	cuStat = cudaGetLastError();
	if (cuStat != cudaSuccess) {
		fprintf(stderr, "kernel function(s) failed: %s\n", cudaGetErrorString(cuStat));
	}
	fprintf(stdout,"Finished kernel functions.\n");
	fflush(stdout);

	cudaMemcpy(outbound, devoutframe, sizeof(long)*(n + m + 1), cudaMemcpyDeviceToHost);

#ifdef DEBUG_DPTABLE
	long * table;
	table = (long*)malloc(sizeof(long)*table_height*table_width);
	cudaMemcpy(table, devtable, sizeof(long)*table_height*table_width, cudaMemcpyDeviceToHost);
	// show DP table
	long c, r, dix;
	const long scales = strlen(grayscale) - 1;
	for (r = 0; r < m + 1; r++) {
		for (c = 0; c < n + 1; c++) {
			long gray = m - table[c*(m+1)+r];
			gray = (gray > 0 ? gray : 0);
			gray = (gray < 0 ? 0 : gray)*scales / m;
			//fprintf(stdout, "%3ld ", table[c*(m+1)+r]);
			fprintf(stdout, "%c ", grayscale[gray]);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");
	free(table);
	cudaFree(devtable);
#endif

	cudaFree(devweftbuff);
	cudaFree(devinframe);
	cudaFree(devoutframe);

	result = outbound[n];

	return result;
}

__global__ void cu_init_row(long * row, const long n, const long m) {
	long thix = blockDim.x * blockIdx.x + threadIdx.x;

	for (int rep = 0; rep < n / m; ++rep) {
		if (rep*(m + 1) + thix < n + 1) {
			row[thix] = thix;
		}
	}
	if (thix < m) {
		row[n+2+thix] = thix+1;
	}
	__syncthreads();
}

// assuming the table array size (n+1) x (m+1)
__global__ void cu_dptable(long * weftbuff, const long * inframe, long * outframe, 
	const char t[], const long n, const char p[], const long m
#ifdef DEBUG_DPTABLE
	,long * table
#endif
) {
	long dix, dcol; // , drow; // diagonal column index, row index
	long col; // inner diagonal index
	long ins, del, repl, cellval;

	const long mperiod = 4;
	long *w0, *w1, *w2;

	// thread id
	long drow = blockDim.x * blockIdx.x + threadIdx.x ;

	long col0val, raw0val;
	if (drow == 0) {
		col0val = inframe[0];
	}
	else if (drow < m + 1) {
		col0val = inframe[n+1+drow-1];
	}
	__syncthreads();

	// skewed rectangle
	for (dcol = 0, dix = 0; dcol < n + m + 1; ++dcol, dix += m+1) {
		//		for (drow = 1; drow < m + 1; ++drow) {
		//drow = thix;
		col = dcol - drow;
		raw0val = inframe[col];
		w0 = weftbuff + (dcol % 4)*(m + 1); // % mperiod)*(m + 1); // the current front line of waves
		w1 = weftbuff + ((dcol - 1 + 4) % 4)*(m + 1); // % mperiod)*(m + 1); // the last passed line of waves
		w2 = weftbuff + ((dcol - 2 + 4) % 4)*(m + 1); // % mperiod)*(m * 1); // the 2nd last line of waves
		if (drow == 0) {
			// load the value of the top row from the initial boundary
			cellval = raw0val;
		} else if (col == 0) {
			// load the value of the left-most column from the initial boundary
			cellval = col0val;
		} else if ((col > 0) && (1 <= drow && drow < m + 1)) {
			ins = w1[drow - 1] + 1;
			del = w1[drow] + 1;
			repl = 0;
			if (t[col - 1] != p[drow - 1])
				repl = 1;
			repl = w2[drow - 1] + repl;
			if (ins > del)
				ins = del;
			if (repl > ins)
				repl = ins;
			cellval = repl;
		}
		if (drow <= m) {
			w0[drow] = cellval;
			table[(m + 1)*col + drow] = w0[drow];
			if (drow == m && col <= n)
				outframe[col] = cellval;
			if ( drow < m && col == n ) 
				outframe[n + 1 + drow] = cellval;
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

