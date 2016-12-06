
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

	cuStat = cudaMalloc((void**)&devweftbuff, sizeof(long)*table_height*2);
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
	for (r = 1; r < m + 1; r++) {
		for (c = 1; c < n + 1; c++) {
			long gray = m - table[c*(m+1)+r];
			gray = (gray > 0 ? gray : 0);
			gray = (gray < 0 ? 0 : gray)*scales / m;
			fprintf(stdout, "%3ld ", table[c*(m+1)+r]);
			//fprintf(stdout, "%c ", grayscale[gray]);
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
	long dcol; // , drow; // diagonal column index
	long col; // inner diagonal index
	long ins, del, repl, nextrepl, w2, cellval;

	const long mperiod = 2;
	long *w0, *w1;

	// thread id = row index
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
	for (dcol = 0; dcol < n + m + 1; ++dcol) {
		//		for (drow = 1; drow < m + 1; ++drow) {
		//drow = thix;
		col = dcol - drow;
		raw0val = inframe[col];
		w0 = weftbuff + (dcol % 2)*(m + 1); // % mperiod)*(m + 1); // the current front line of waves
		w1 = weftbuff + ((dcol - 1 + 2) % 2)*(m + 1); // % mperiod)*(m + 1); // the last passed line of waves
		if ((col > 0) && (1 <= drow && drow < m + 1)) {
			if (drow == 1) {
				w2 = inframe[col];
			} else {
				w2 = w1[drow - 1];
			}
			ins = w2 + 1;
			if (col == 1) {
				del = col0val;
			} else {
				//del = w1[drow] + 1;
				del = cellval + 1;
			}
			if (col == 1 && drow == 1) {
				repl = 0;
			} else {
				repl = nextrepl; // w2[drow - 1];
			}
			repl += (t[col - 1] != p[drow - 1]);
			if (ins > del)
				ins = del;
			if (repl > ins)
				repl = ins;
			cellval = repl;
			//
			del = cellval;
			nextrepl = w2;
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
