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

	cu_levdist_kernel <<<1, 1>> > (devframe, weftlen, devtext, n, devpatt, m
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

	result = frame[(n - m) & (weftlen - 1)];

	return result;
}

__global__ void cu_setframe(long * frame, const long weftlen, const long n, const long m) {
	const long thix = blockDim.x * blockIdx.x + threadIdx.x;

	if (thix < n + 1) {
		frame[thix] = thix;
	}
	else if (thix < weftlen && thix > weftlen - m - 1) {
		frame[thix] = weftlen - thix;
	}
	else {
		frame[thix] = 0;  // will be untouched.
	}
}


__global__ void cu_levdist_kernel(long * frame, const long weftlen, const char t[], const long n, const char p[], const long m
#ifdef DEBUG_TABLE
, long * table
#endif
) {
	long result = n + m + 1;
	long col, row;
	long del, ins, repl, cellval; // del = delete from pattern, downward; ins = insert to pattern, rightward
	long thix, lthix, rthix;
	long thread_min, thread_max;

	if (frame == NULL)

		return;

	for (long depth = 0; depth < n + m - 1; depth++) {
		thread_min = -depth;
		if (!(depth < m))
			thread_min += (depth + 1 - m) << 1;

		thread_max = depth;
		if (!(depth < n))
			thread_max -= (depth + 1 - n) << 1;

		//printf("depth = %ld, %ld <= thread < %ld\n", depth, thread_min, thread_max);
		//printf("thread_min = %d\n", (thread_min + weftlen) & (weftlen - 1) );
		//cu_weft_kernel <<< ceildiv(weftlen, 512), 512 >> > (frame, weftlen, t, n, p, m, depth, thread_min, thread_max, table);
		
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
			// printf("[%ld,%ld] %c|%c : %ld/%ld/%ld+%ld,\n",col,row,t[col],p[row], del,ins, frame[thix], (t[col] != p[row]));
			//
			if (del < ins) {
				ins = del;
			}
			if (ins < repl) {
				repl = ins;
			}
			//
			frame[thix] = repl;
#ifdef DEBUG_TABLE
			table[m*col + row] = repl;
#endif

		}
		
	}
	/*
	result = frame[(n - m) & (weftlen - 1)];
	return result;
	*/
}


__global__ void cu_weft_kernel(long * frame, const long weftlen, const char * t, const long n, const char * p, const long m, const long depth, const long thread_start, const long thread_last
#ifdef DEBUG_TABLE
, long * table
#endif
) {
	const long thread = blockDim.x * blockIdx.x + threadIdx.x;
	long col, row;
	long thix, lthix, rthix, del, ins, repl;

//	for (long thread = thread_min; thread <= thread_max; thread += 2) {
	if ( 0 <= thread && thread <= thread_last ) { 
		if ((thread & 1) == (((thread_start + weftlen)&(weftlen-1)) & 1)) {
			col = (depth + thread) >> 1;
			row = (depth - thread) >> 1;

			thix = (thread + weftlen) & (weftlen - 1);
			lthix = (thix - 1 + weftlen) & (weftlen - 1);
			rthix = (thix + 1) & (weftlen - 1);
			//
			del = frame[rthix] + 1;
			ins = frame[lthix] + 1;
			repl = frame[thix] + (t[col] != p[row]);
			// printf("[%ld,%ld] %c|%c : %ld/%ld/%ld+%ld,\n",col,row,t[col],p[row], del,ins, frame[thix], (t[col] != p[row]));
			//
			if (del < ins) {
				ins = del;
			}
			if (ins < repl) {
				repl = ins;
			}
			//
			frame[thix] = repl;
#ifdef DEBUG_TABLE
			table[m*col + row] = thix;
#endif
		}
	}

	return;
}