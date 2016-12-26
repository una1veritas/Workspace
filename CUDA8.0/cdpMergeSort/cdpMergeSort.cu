/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#include <iostream>
#include <cstdio>

#include <helper_timer.h>
#include <helper_cuda.h>
#include <helper_string.h>

#define MAX_DEPTH       16
#define INSERTION_SORT  32

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(long *data, long left, long right)
{
    for (long i = left ; i <= right ; ++i)
    {
        long min_val = data[i];
        long min_idx = i;

        // Find the smallest value in the range [left, right].
        for (long j = i+1 ; j <= right ; ++j)
        {
            long val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

__global__ 
void cdp_merge(long array[], long len, long n, long dst[]) {
	const long thix = blockIdx.x * blockDim.x + threadIdx.x;
	long start, end;
	long cl, cr, ct;
	start = thix * (len << 1);
	if (start < n) {
		end = min(start + (len << 1), n);
		//printf("cdp_merge %d, %d [%d, %d)\n", len, thix, start, end);
		for (cl = start, cr = start + len, ct = cl; ct < end; ct++) {
			if (cl < start + len && cr < end) {
				if (array[cl] < array[cr]) {
					dst[ct] = array[cl];
					cl++;
				} else {
					dst[ct] = array[cr];
					cr++;
				}
			} else {
				if (cl < start + len) {
					dst[ct] = array[cl];
					cl++;
				} else {
					dst[ct] = array[cr];
					cr++;
				}
			}
		}
	}
	__syncthreads();
}


__global__
void copy_back(long array[], long buf[], long n) {
	const long thix = blockDim.x*blockIdx.x + threadIdx.x;
	if ( thix < n)
		array[thix] = buf[thix];
	__syncthreads();
}

//__global__ 
void cdp_mergeSort(long array[], long buf[], int n) {
	long *a, *b, *t, len;
	long threadnum;

	a = array;
	b = buf;
	for (len = 1; len < n; len = len <<= 1 ) {
		threadnum = (n + (len << 1) - 1) / (len<<1);
		//printf("cdp_mergeSort: len = %d, blocks %d, total threads %d\n", len, (threadnum + 191) / 192, threadnum);
		//for (start = 0; start < n; start += (len<<1)) {
		cdp_merge << < (threadnum + 191) / 192, 192 >> > (a, len, n, b);
		checkCudaErrors(cudaDeviceSynchronize());
		
		if ((len << 1) < n ) {
			t = a;
			a = b; 
			b = t;
		}
		else if (b == buf) {
			copy_back << <(n + 191) / 192, 192 >> > (array, buf, n);
			checkCudaErrors(cudaDeviceSynchronize());
		}
	}
	return;
}

void mergeSort(long array[], long n) {
	long * buf;
	long i, len, start, end, cleft, cright, ctemp;

	buf = (long*)malloc(sizeof(long)*n);
	if (buf == NULL) {
		printf("error.\n");
		return;
	}

	for (len = 1; len < n; len = len << 1) {
		for (start = 0; start < n; start = start + (len << 1)) {
			end = min(start + (len << 1), n);
			//printf("%d: %d, %d\n", len, start, end);
			for (cleft = start, cright = start + len, ctemp = cleft; ctemp < end; ctemp++) {
				if (cleft < start + len && cright < end) {
					if (array[cleft] < array[cright]) {
						buf[ctemp] = array[cleft];
						cleft++;
					}
					else {
						buf[ctemp] = array[cright];
						cright++;
					}
				}
				else {
					if (cleft < start + len) {
						buf[ctemp] = array[cleft];
						cleft++;
					}
					else {
						buf[ctemp] = array[cright];
						cright++;
					}
				}
			}
		}
		//copy back
		for (i = 0; i < n; i++)
			array[i] = buf[i];
	}

	free(buf);
	return;
}


void run_mergeSort(long *data, long nitems)
{
	long * devbuf;
	// Allocate GPU memory.
	checkCudaErrors(cudaMalloc((void **)&devbuf, nitems * sizeof(long)));

	// Prepare CDP for the max depth 'MAX_DEPTH'.
	//checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 2));

	// Launch on device
	std::cout << "Launching mergeSort kernel on the GPU" << std::endl;
	
	cdp_mergeSort(data, devbuf, nitems);

	cudaFree(devbuf);
}


////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(long *data, long left, long right, int depth)
{
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT)
    {
        selection_sort(data, left, right);
        return;
    }

    long *lptr = data+left;
    long *rptr = data+right;
    long  pivot = data[(left+right)/2];

    // Do the partitioning.
    while (lptr <= rptr)
    {
        // Find the next left- and right-hand values to swap
        long lval = *lptr;
        long rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot)
        {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot)
        {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr)
        {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}


////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(long *data, long nitems)
{
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    // Launch on device
    long left = 0;
    long right = nitems-1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_simple_quicksort<<< 1, 1 >>>(data, left, right, 0);
    checkCudaErrors(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
void initialize_data(long *dst, long nitems)
{
    // Fixed seed for illustration
    srand(2047);

    // Fill dst with random values
    for (long i = 0 ; i < nitems ; i++)
        dst[i] = rand() % nitems ;
}

////////////////////////////////////////////////////////////////////////////////
// Verify the results.
////////////////////////////////////////////////////////////////////////////////
void check_results(long n, long *results_h)
{
    for (long i = 1 ; i < n ; ++i)
        if (results_h[i-1] > results_h[i])
        {
            std::cout << "Invalid item[" << i-1 << "]: " << results_h[i-1] << " greater than " << results_h[i] << std::endl;
            exit(EXIT_FAILURE);
        }

    std::cout << "OK" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    long num_items = 128;
    bool verbose = false;

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h"))
    {
        std::cerr << "Usage: " << argv[0] << " num_items=<num_items>\twhere num_items is the number of items to sort" << std::endl;
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "v"))
    {
        verbose = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "num_items"))
    {
        num_items = getCmdLineArgumentInt(argc, (const char **)argv, "num_items");

        if (num_items < 1)
        {
            std::cerr << "ERROR: num_items has to be greater than 1" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Get device properties
    int device_count = 0, device = -1;
    
    if(checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        
        cudaDeviceProp properties;
        checkCudaErrors(cudaGetDeviceProperties(&properties, device));
        
        if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
        {
            std::cout << "Running on GPU " << device << " (" << properties.name << ")" << std::endl;
        }
        else
        {
            std::cout << "ERROR: cdpsimpleQuicksort requires GPU devices with compute SM 3.5 or higher."<< std::endl;
            std::cout << "Current GPU device has compute SM" << properties.major <<"."<< properties.minor <<". Exiting..." << std::endl;
            exit(EXIT_FAILURE);
        }

    }
    else
    {
        checkCudaErrors(cudaGetDeviceCount(&device_count));
    
        for (int i = 0 ; i < device_count ; ++i)
        {
            cudaDeviceProp properties;
            checkCudaErrors(cudaGetDeviceProperties(&properties, i));

            if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
            {
                device = i;
                std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
                break;
            }

            std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
         }
     }

    if (device == -1)
    {
        std::cerr << "cdpSimpleQuicksort requires GPU devices with compute SM 3.5 or higher.  Exiting..." << std::endl;
        exit(EXIT_WAIVED);
    }

    cudaSetDevice(device);

    // Create input data
    long *h_data = 0;
    long *d_data = 0;

    // Allocate CPU memory and initialize data.
    std::cout << "Initializing data:" << std::endl;
    h_data =(long *)malloc(num_items*sizeof(long));
    initialize_data(h_data, num_items);

    if (verbose)
    {
        for (long i=0 ; i<num_items ; i++)
            std::cout << "Data [" << i << "]: " << h_data[i] << std::endl;
    }

    // Allocate GPU memory.
    checkCudaErrors(cudaMalloc((void **)&d_data, num_items * sizeof(long)));
    checkCudaErrors(cudaMemcpy(d_data, h_data, num_items * sizeof(long), cudaMemcpyHostToDevice));

    // Execute
    std::cout << "Running sort algorithm on " << num_items << " elements" << std::endl;

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

    run_qsort(d_data, num_items);

	sdkStopTimer(&timer);
	printf("\nGPU qsort elapsed %.3f msec.\n", sdkGetTimerValue(&timer));

	std::cout << "Validating results: ";
	checkCudaErrors(cudaMemcpy(h_data, d_data, num_items * sizeof(long), cudaMemcpyDeviceToHost));
	for (long i = 0; i < (num_items < 128 ? num_items : 128); i++) {
		printf("%d, ", h_data[i]);
	}
	printf("\n");
	check_results(num_items, h_data);


	initialize_data(h_data, num_items);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	mergeSort(h_data, num_items);

	sdkStopTimer(&timer);
	printf("\nCPU mergeSort elapsed %.3f msec.\n", sdkGetTimerValue(&timer));
	std::cout << "Validating results: ";
	for (long i = 0; i < (num_items < 128 ? num_items : 128); i++) {
		printf("%d, ", h_data[i]);
	}
	printf("\n");
	check_results(num_items, h_data);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	run_mergeSort(d_data, num_items);

	sdkStopTimer(&timer);
	printf("\nGPU mergeSort elapsed %.3f msec.\n", sdkGetTimerValue(&timer));

	checkCudaErrors(cudaMemcpy(h_data, d_data, num_items * sizeof(long), cudaMemcpyDeviceToHost));
	// Check result
    std::cout << "Validating results: ";
	for (long i = 0; i < (num_items < 128 ? num_items : 128); i++) {
		printf("%d, ", h_data[i]);
	}
	printf("\n");
	check_results(num_items, h_data);


    free(h_data);
    checkCudaErrors(cudaFree(d_data));

    exit(EXIT_SUCCESS);
}

