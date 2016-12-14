
/*
 * culevdist.c
 *
 *  Created on: 2016/11/26
 *      Author: sin
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "culevdist.h"

#include "debug_table.h"

#define MAX_THREADSPERBLOCK (192)

#define min(x, y)  ((x) <= (y)? (x) : (y))
#define max(x, y)  ((x) >= (y)? (x) : (y))
#define align(base, val)    ((((val)+(base)-1)/(base))*(base))
#define ceildiv(n,d)		( (n)%(d) > 0 ? (n)/(d)+1 : (n)/(d) )
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
	const long theight = align(MAX_THREADSPERBLOCK, m + 1);
	const long twidth = n + 1;

	cuStat = cudaMalloc((void**)&devweftbuff, sizeof(long)*theight* 4);
	cuStatCheck(cuStat, "cudaMalloc devtable failed.\n");

	long * devtable = NULL;
#ifdef DEBUG_TABLE
	cuStat = cudaMalloc((void**)&devtable, sizeof(long)*table_height*table_width);
	cuStatCheck(cuStat, "cudaMalloc devtable failed.\n");
#endif
	cuStat = cudaMalloc((void**)&devinframe, sizeof(long)*(n + m + 1));
	cuStatCheck(cuStat, "cudaMalloc devinframe failed.\n");
	cuStat = cudaMalloc((void**)&devoutframe, sizeof(long)*(n + 1 + m));
	cuStatCheck(cuStat, "cudaMalloc devoutframe failed.\n");
	cudaMemcpy(devinframe, inbound, sizeof(long)*(n + m + 1), cudaMemcpyHostToDevice);

	long nthreads = align(32, m + 1), threadsperblock = min(nthreads, MAX_THREADSPERBLOCK);
	dim3 grids(ceildiv(nthreads, threadsperblock)), blocks(threadsperblock);
	fprintf(stdout,"\nnum threads %d, %d blocks, %d threads per block.\n",nthreads, ceildiv(nthreads,threadsperblock), threadsperblock);

	// dp table computation
	long dcol; // , drow; // diagonal column index
	long *w0, *w1, *w2;
	for (dcol = 0; dcol < n + m + 1; ++dcol) {
		w0 = devweftbuff + (dcol & 0x03)*theight; // % mperiod)*(m + 1); // the current front line of waves
		w1 = devweftbuff + ((dcol - 1 + 4) & 0x03)*theight; // % mperiod)*(m + 1); // the last passed line of waves
		w2 = devweftbuff + ((dcol - 2 + 4) & 0x03)*theight; // % mperiod)*(m + 1); // the second last line of waves

		cu_dptable_kernel << < grids, blocks >> > (w2, w1, w0, dcol, devinframe, devoutframe, devt, n, devp, m, devtable);
	}

	// Check for any errors launching the kernel
	cuStat = cudaGetLastError();
	if (cuStat != cudaSuccess) {
		fprintf(stderr, "kernel function(s) failed: %s\n", cudaGetErrorString(cuStat));
		fflush(stdout);
	}

	cudaMemcpy(outbound, devoutframe, sizeof(long)*(n + m + 1), cudaMemcpyDeviceToHost);

#ifdef DEBUG_TABLE
	cudaMemcpy(debug_table, devtable, sizeof(long)*table_height*table_width, cudaMemcpyDeviceToHost);
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
	const char t[], const long n, const char p[], const long m, long * table
) {
	long dcol; // , drow; // diagonal column index
	long col; // inner diagonal index
	long ins, del, repl, cellval; // nextrepl, above, prevval;

	long *w0, *w1, *w2;

	// thread id = row index
	long drow = blockDim.x * blockIdx.x + threadIdx.x ;

	//long col0val;
	//__syncthreads();

	// skewed rectangle
	for (dcol = 0; dcol < n + m + 1; ++dcol) {
		//		for (drow = 1; drow < m + 1; ++drow) {
		//drow = thix;
		col = dcol - drow;
		w0 = weftbuff + (dcol & 0x03)*(m + 1); // % mperiod)*(m + 1); // the current front line of waves
		w1 = weftbuff + ((dcol - 1 + 4) & 0x03)*(m + 1); // % mperiod)*(m + 1); // the last passed line of waves
		w2 = weftbuff + ((dcol - 2 + 4) & 0x03)*(m + 1); // % mperiod)*(m + 1); // the second last line of waves
		if (drow == 0 && drow < m + 1) {
			// load the value of the top row from the initial boundary 
			cellval = inframe[m+col];
		}
		else if (col == 0 && drow < m + 1) {
			// load the value of the left-most column from the initial boundary 
			//cellval = col0val;
			cellval = inframe[m-drow];
		}
		else if ((col > 0) && (1 <= drow && drow < m + 1)) {
			ins = w1[drow - 1] + 1;
			del = w1[drow] + 1;
			repl = w2[drow - 1] + (t[col - 1] != p[drow - 1]);
			cellval = ins;
			if (del < cellval)
				cellval = del;
			if (repl < cellval)
				cellval = repl;
		}
		if ( (0 <= col && col < n+1) && drow < m + 1) {
			//if (drow == 2) printf("(%d, 1) = %d\n", col, cellval);
			w0[drow] = cellval;
#ifdef DEBUG_TABLE
			table[(m + 1)*col + drow] = w0[drow];
#endif
			if (drow == m && col <= n)
				outframe[col] = cellval;
			if ( drow < m && col == n ) 
				outframe[n + 1 + drow] = cellval;
		}

		__syncthreads();
	}

	return;
}

// assuming the table array size (n+1) x (m+1)
__global__ void cu_dptable_kernel(long * w2, long *w1, long *w0, const long dcol, const long * inframe, long * outframe,
	const char t[], const long n, const char p[], const long m, long * table
) {
	long col; // inner diagonal index
	long ins, del, repl, cellval; // nextrepl, above, prevval;

	// thread id = row index
	long drow = blockDim.x * blockIdx.x + threadIdx.x;

	//__syncthreads();

	//for (dcol = 0; dcol < n + m + 1; ++dcol) {
		//for (drow = 1; drow < m + 1; ++drow) {
		//drow = thix;
		col = dcol - drow;
		//w0 = weftbuff + (dcol % 4)*(m + 1); // % mperiod)*(m + 1); // the current front line of waves
		//w1 = weftbuff + ((dcol - 1 + 4) % 4)*(m + 1); // % mperiod)*(m + 1); // the last passed line of waves
		//w2 = weftbuff + ((dcol - 2 + 4) % 4)*(m + 1); // % mperiod)*(m + 1); // the second last line of waves
		if (drow == 0 && drow < m + 1) {
			// load the value of the top row from the initial boundary 
			cellval = inframe[m + col];
		}
		else if (col == 0 && drow < m + 1) {
			// load the value of the left-most column from the initial boundary 
			//cellval = col0val;
			cellval = inframe[m - drow];
		}
		else if ((col > 0) && (1 <= drow && drow < m + 1)) {
			ins = w1[drow - 1] + 1;
			del = w1[drow] + 1;
			repl = w2[drow - 1] + (t[col - 1] != p[drow - 1]);
			cellval = ins;
			if (del < cellval)
				cellval = del;
			if (repl < cellval)
				cellval = repl;
		}
		if ((0 <= col && col < n + 1) && drow < m + 1) {
			//if (drow == 2) printf("(%d, 1) = %d\n", col, cellval);
			w0[drow] = cellval;
#ifdef DEBUG_TABLE
			table[(m + 1)*col + drow] = w0[drow];
#endif
			if (drow == m && col <= n)
				outframe[col] = cellval;
			if (drow < m && col == n)
				outframe[n + 1 + drow] = cellval;
		}
		//__syncthreads();

	return;
}
