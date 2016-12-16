
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

#include "cudautils.h"

#define MAX_THREADSPERBLOCK (192)

#define min(x, y)  ((x) <= (y)? (x) : (y))
#define max(x, y)  ((x) >= (y)? (x) : (y))

static char grayscale[] = "@#$B%8&WM*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>ilI;!:,\"^`'. ";


long cu_levdist(long * frame, const char t[], const long n, const char p[], const long m) {
	long result = n + m + 1; // an impossible value
	long weftlen = pow2(n+m+1);
	char * devt, *devp;
	long *devframe;

	CUCHECK(cudaMalloc((void**) &devt, n));
	CUCHECK(cudaMemcpy(devt, t, n, cudaMemcpyHostToDevice));
	CUCHECK(cudaMalloc((void**)&devp, m));
	CUCHECK(cudaMemcpy(devp, p, m, cudaMemcpyHostToDevice));
	CUCHECK(cudaMalloc((void**)&devframe, sizeof(long)*weftlen));

#ifdef DEBUG_TABLE
	long *devtable;
	CUCHECK(cudaMalloc((void**)&devtable, sizeof(long)*(n+1)*(m+1)));
#endif
	CUCHECK(cudaMemcpy(devframe, frame, sizeof(long)*weftlen, cudaMemcpyHostToDevice));

	long numthds = pow2(n+m + 1), thdsperblock = min(numthds, MAX_THREADSPERBLOCK);
	dim3 grids(ceildiv(numthds, thdsperblock)), blocks(thdsperblock);
	fprintf(stdout,"\nnum threads %d, %d blocks, %d threads per block.\n",numthds, ceildiv(numthds,thdsperblock), thdsperblock);

	// dp table computation
	wv_edist(devframe, t, n, p, n);

	// Check for any errors launching the kernel
	CUCHECK(cudaGetLastError());

	CUCHECK(cudaMemcpy(frame, devframe, sizeof(long)*weftlen, cudaMemcpyDeviceToHost));

#ifdef DEBUG_TABLE
	cudaMemcpy(debug_table, devtable, sizeof(long)*(m+1)*(n+1), cudaMemcpyDeviceToHost);
	cudaFree(devtable);
#endif

	cudaFree(devframe);

	result = frame[(n - m) & (weftlen -1)];

	return result;
}

void wv_setframe(long * frame, const char t[], const long n, const char p[], const long m) {
	for (long i = 0; i < m + n + 1; i++) {
		if (i < n + 1) {
			frame[i] = i;
		}
		else {
			frame[i] = n + m + 1 - i;
		}
	}
}

long pow2(const long val) {
	long result = 1;

	while (result < val) {
		result <<= 1;
	}
	return result;
}

long wv_edist(long * frame, const char t[], const long n, const char p[], const long m) {
	long result = n + m + 1;
	long col, row;
	long del, ins, repl; // del = delete from pattern, downward; ins = insert to pattern, rightward
	long thix, lthix, rthix;
	long thread_min, thread_max;
	long weftlen = pow2(n + m + 1);

	if (frame == NULL)
		return -1;

	for (long depth = 0; depth < n + m; depth++) {
		thread_min = -depth;
		if (!(depth < m))
			thread_min += (depth + 1 - m) << 1;

		thread_max = depth;
		if (!(depth < n))
			thread_max -= (depth + 1 - n) << 1;

		for (long thread = thread_min; thread <= thread_max; thread += 2) {
			col = (depth + thread) >> 1;
			row = (depth - thread) >> 1;

			thix = (thread + weftlen) & (weftlen - 1);
			lthix = (thix - 1 + weftlen) & (weftlen - 1);
			rthix = (thix + 1) & (weftlen - 1);
			//
			del = frame[rthix] + 1;
			ins = frame[lthix] + 1;
			repl = frame[thix] + (t[col] != p[row]);
			//
			if (del < ins)
				ins = del;
			if (ins < repl)
				repl = ins;
			//
			frame[thix] = repl;
#ifdef DEBUG_TABLE
			debug_table[m*col + row] = repl;
#endif
		}
	}

	result = frame[(n - m) & (weftlen - 1)];

	return result;
}
