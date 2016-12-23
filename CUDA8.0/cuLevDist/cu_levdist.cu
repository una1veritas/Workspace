#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cu_levdist.h"
#include "cu_utils.h"

#include "debug_table.h"

long cu_levdist(long * frame, const char text[], const long n, const char patt[], const long m) {
	long result = n + m + 1;
	long weftlen = cu::pow2(n+m+1);

	long * devframe;
	char *devtext, *devpatt;
	cudaMalloc((void **)&devframe, sizeof(long)*weftlen);
	cudaMemcpy(devframe, frame, sizeof(long)*weftlen, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&devtext, sizeof(char)*n);
	cudaMemcpy(devtext, text, sizeof(char)*n, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&devpatt, sizeof(char)*m);
	cudaMemcpy(devpatt, patt, sizeof(char)*m, cudaMemcpyHostToDevice);

#ifdef DEBUG_TABLE
	long * dev_debug_table;
	cudaMalloc((void**)&dev_debug_table, sizeof(long)*(n*m));
#endif

	weaving_kernel <<<1, 1>> > (devframe, weftlen, devtext, n, devpatt, m
#ifdef DEBUG_TABLE
		, dev_debug_table
#endif
		);
		cuCheckErrors(cudaGetLastError());

#ifdef DEBUG_TABLE
	cudaMemcpy(debug_table, dev_debug_table, sizeof(long)*(n*m),cudaMemcpyDeviceToHost);
	cudaFree(dev_debug_table);
#endif
	cudaMemcpy(frame, devframe, sizeof(long)*weftlen, cudaMemcpyDeviceToHost);
	cudaFree(devframe);

	cuCheckErrors(cudaDeviceSynchronize());

	result = frame[n];

	return result;
}


__global__ void weaving_kernel(long * frame, const long weftlen, const char t[], const long n, const char p[], const long m
#ifdef DEBUG_TABLE
, long * table
#endif
) {
	long result = n + m + 1;
	long col, row;
	long del, ins, repl, cellval; // del = delete from pattern, downward; ins = insert to pattern, rightward
	long warpix, warp_start, warp_last;

	if (frame == NULL)
		return;

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
			col = (depth + warpix - (m - 1))>>1;
			row = (depth - warpix + (m - 1))>>1;

			//printf("%ld = (%ld, %ld), ", warpix, col, row);
			//
			del = frame[warpix+1+1] + 1;
			ins = frame[warpix-1+1] + 1;
			repl = frame[warpix+1] + (t[col] != p[row]);
			//printf("%ld: %ld [%ld,%ld] %c|%c : %ld/%ld/%ld+%ld,\n",depth, warpix, col,row,t[col],p[row], del,ins, frame[warpix], (t[col] != p[row]));
			//
			if (del < ins) {
				ins = del;
			}
			if (ins < repl) {
				repl = ins;
			}
			//
			frame[warpix+1] = repl;
#ifdef DEBUG_TABLE
			table[m*col + row] = repl;
#endif

		}
		//printf("\n");
		
	}
}

