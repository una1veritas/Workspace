/*
============================================================================
Name        : fft_main.c
Author      : Sin Shimozono
Version     :
Copyright   : reserved.
Description : Factored discrete Fourier transform, or FFT, and its inverse iFFT
============================================================================
* Reference:
* http://www.math.wustl.edu/~victor/mfmm/fourier/fft.c
* http://rosettacode.org/wiki/Fast_Fourier_transform
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_timer.h>

#define DEBUG_VECTOR
#define DEBUG_OCCURRENCES

#define VECTOR_MAXSIZE 1024

struct compvect {
	cuComplex * elem;
	unsigned int dim;
};
typedef struct compvect compvect;

struct text {
	char * str;
	unsigned int size;
	unsigned int length;
};
typedef struct text text;

__global__ void cuCvecmul_into(cuComplex * x, cuComplex * y, const int dimsize);
__global__ void cuCfindpos(cuComplex * v, unsigned int * occurrences, const int dim, const int pattlen);
__global__ void make_signal(const char * str, const unsigned int length, cuComplex * vec, const int dim, const int flag);

#define min(x,y)  ( ((x) < (y) ? (x) : (y)) )
#define max(x,y)  ( ((x) < (y) ? (y) : (x)) )
#define abs(x)  ( (x) < 0 ? (-(x)) : (x) )

long smallestpow2(const long n) {
	long t = 1;
	while (t < n) {
		t <<= 1;
	}
	return t;
}

void print_vector(const char *title, compvect *x);

__host__ __device__ static __inline__ cuComplex cuCexpf(cuComplex x)
{
	float factor = expf(x.x);
	return make_cuComplex(factor * cosf(x.y), factor * sinf(x.y));
}

int main(int argc, char * argv[]) {
	text text1, text2;
	compvect vec1, vec2;
	int vecsize;
	int pattlen;

	if (argc != 3)
		exit(EXIT_FAILURE);

	text1.size = VECTOR_MAXSIZE;
	text1.str = (char*)malloc(sizeof(char)*text1.size);
	strncpy(text1.str, argv[1], text1.size);
	text1.length = min(strlen(argv[1]), text1.size);
	text2.size = VECTOR_MAXSIZE;
	text2.str = (char*)malloc(sizeof(char)*text2.size);
	strncpy(text2.str, argv[2], text2.size);
	text2.length = min(strlen(argv[2]), text2.size);

	char buf[VECTOR_MAXSIZE+1];
	strncpy(buf, text1.str, VECTOR_MAXSIZE);
	buf[VECTOR_MAXSIZE] = 0;
	printf("inputs: \"%s\" (%d), ", buf, text1.length);
	strncpy(buf, text2.str, VECTOR_MAXSIZE);
	buf[VECTOR_MAXSIZE] = 0;
	printf("\"%s\" (%d)\n", buf, text2.length);
	vecsize = smallestpow2(min(max(text1.length, text2.length), VECTOR_MAXSIZE));
	pattlen = min(text1.length, text2.length);

	vec1.dim = vecsize;
	vec1.elem = (cuComplex*)malloc(sizeof(cuComplex) * vec1.dim);
	vec2.dim = vecsize;
	vec2.elem = (cuComplex*)malloc(sizeof(cuComplex) * vec2.dim);


	/* 時間計測用タイマーのセットアップと計時開始 */
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	/* GPU用メモリ割り当て */
	cufftComplex *devmemptr;
	char * devstrptr;

	
	/* バッチ数 2 (vec1.elem, vec2.elem)　*/
	cudaError_t error_id;
	error_id = cudaMalloc((void**)&devmemptr, sizeof(cufftComplex) * vecsize * 2);
	if ( error_id != cudaSuccess ) {
		printf("cudaMalloc returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}
	error_id = cudaMalloc((void**)&devstrptr, sizeof(int) * vecsize);
	if ( error_id != cudaSuccess ) {
		printf("cudaMalloc returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}
	/* GPU thread allocation */
	dim3 grid(32, 1);
	dim3 block(VECTOR_MAXSIZE / 32, 1);

	/* GPU用メモリに転送 */
	cudaMemcpy(devstrptr, text1.str, sizeof(char)*text1.length, cudaMemcpyHostToDevice);
	make_signal<<<grid, block>>>(devstrptr, text1.length, devmemptr, vecsize, 0);
	cudaMemcpy(devstrptr, text2.str, sizeof(char)*text1.length, cudaMemcpyHostToDevice);
	make_signal<<<grid, block>>>(devstrptr, text2.length, devmemptr+vecsize, vecsize, 1);

#ifdef DEBUG_VECTOR
	cudaMemcpy(vec1.elem, devmemptr, sizeof(cuComplex)*vecsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(vec2.elem, devmemptr+vecsize, sizeof(cuComplex)*vecsize, cudaMemcpyDeviceToHost);
	print_vector("text1 ", &vec1);
	print_vector("text2 ", &vec2);
#endif

	//cudaMemcpy(devmemptr, vec1.elem, sizeof(cufftComplex)* vecsize, cudaMemcpyHostToDevice);
	//cudaMemcpy(devmemptr + vecsize, vec2.elem, sizeof(cufftComplex)* vecsize, cudaMemcpyHostToDevice);

	/* 1D FFT plan作成 */
	cufftHandle cufftplan;
	cufftPlan1d(&cufftplan, vecsize, CUFFT_C2C, 2);
	cufftExecC2C(cufftplan, devmemptr, devmemptr, CUFFT_FORWARD);
	/* CUFFT plan削除 */
	cufftDestroy(cufftplan);

#ifdef DEBUG_VECTOR
	/*  計算結果をGPUメモリから転送して表示 */
	cudaMemcpy(vec1.elem, devmemptr, sizeof(cufftComplex)*vec1.dim, cudaMemcpyDeviceToHost);
	cudaMemcpy(vec2.elem, devmemptr + vecsize, sizeof(cufftComplex)*vec2.dim, cudaMemcpyDeviceToHost);
	print_vector("fft1: 2 ", &vec1);
	print_vector("fft2 ", &vec2);
#endif

	/* ベクトルの積をとる */
	cuCvecmul_into << < grid, block >> > (devmemptr, devmemptr + vecsize, vecsize);

#ifdef DEBUG_VECTOR
	/*  計算結果をGPUメモリから転送して表示 */
	cudaMemcpy(vec1.elem, devmemptr, sizeof(cufftComplex)* vecsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(vec2.elem, devmemptr + vecsize, sizeof(cufftComplex)*vecsize, cudaMemcpyDeviceToHost);
	print_vector("prod ", &vec1);
#endif

	cufftPlan1d(&cufftplan, vecsize, CUFFT_C2C, 1);
	cufftExecC2C(cufftplan, devmemptr, devmemptr, CUFFT_INVERSE);
	/* CUFFT plan削除 */
	cufftDestroy(cufftplan);

#ifdef DEBUG_VECTOR
	cudaMemcpy(vec1.elem, devmemptr, sizeof(cufftComplex)* vecsize, cudaMemcpyDeviceToHost);
	print_vector("iFFT ", &vec1);
#endif

	cuCfindpos << <grid, block >> >(devmemptr, (unsigned int *)(devmemptr + vecsize), vecsize, pattlen);

	/*
	int pos = vec1.dimsize;
	for (int i = pattlen - 1; i < vec1.dimsize + pattlen - 1; i++) {
	if (abs(cuCrealf(vec1.elem[(vec1.dimsize + i - 1) % vec1.dimsize])/vecsize - (double)pattlen) < (1.0F/4096.0F))
	pos = min(pos, i);
	}
	*/
	/*
	printf("Occurring positions: ");
	for(int i = 0; i < vec1.dimsize; i++) {
	if ( abs(creal(vec1.elem[(vec1.dimsize - pattlen + i) % vec1.dimsize]) - (double) pattlen) < 0.0001220703125 )
	printf("%d, ", i - pattlen + 1);
	}
	printf(".\n");
	*/
#ifdef DEBUG_OCCURRENCES
	unsigned int pos[VECTOR_MAXSIZE];
	cudaMemcpy(pos, devmemptr + vecsize, sizeof(int) * vecsize, cudaMemcpyDeviceToHost);
#else
	unsigned int pos[1];
	cudaMemcpy(pos, devmemptr + vecsize, sizeof(int) * 1, cudaMemcpyDeviceToHost);
#endif
	/* タイマーを停止しかかった時間を表示 */
	sdkStopTimer(&timer);
	printf("computation time %f (ms)\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

#ifdef DEBUG_OCCURRENCES
	printf("\nResult: \n");
	for (int i = 0; i < vecsize; i++) {
		printf("[%d] %d, ", i, pos[i]);
	}
#endif
	printf("\n");
	if (pos[0] < vecsize)
		printf("First occurrence is at %d.\n", pos[0] - pattlen + 1);
	else
		printf("Could not find.\n");

	/* GPU用メモリ開放*/
	cudaFree(devmemptr);
	cudaFree(devstrptr);

	free(vec1.elem);
	free(vec2.elem);
	free(text1.str);
	free(text2.str);

	exit(EXIT_SUCCESS);
}

__global__ void make_signal(const char * str, const unsigned int strlen, 
							cuComplex * vec, const int dim, const int flag) {
	const int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int dst;
	float factor;

	// the first as normal
	if ( idx < dim ) {
		if (!flag) {
			dst = idx;
			factor = 2 * 3.14159265358979323846264338327950288F;
		}
		else {
			dst = dim - idx - 1;
			factor = -2 * 3.14159265358979323846264338327950288F;
		}
		if (idx < strlen)
			vec[dst] = cuCexpf(make_cuComplex(0, factor * (float)(str[idx]) / 256.0f));  // by rotated unit vector
																								  // (*array)[i] = (float)(str[i]) / 128.0f  ;  // by char value
		else
			vec[dst] = make_cuComplex(0, 0);
	}
	__syncthreads();
}

/* Print a vector of complexes as ordered pairs. */
void print_vector(const char *title, compvect *v) {
	unsigned int i;
	printf("%s (dim=%d):\n", title, v->dim);
	for (i = 0; i < min(v->dim, 28); i++)
		printf("%6d    ", i);
	putchar('\n');
	for (i = 0; i < min(v->dim, 28); i++)
		printf(" %8.3f,", cuCrealf(v->elem[i]));
	putchar('\n');
	for (i = 0; i < min(v->dim, 28); i++)
		printf(" %8.3f,", cuCimagf(v->elem[i]));
	putchar('\n');
	for (i = 0; i < min(v->dim, 28); i++)
		printf(" %8.3f,", cuCabsf(v->elem[i]));
	printf("\n\n");
	return;
}

__global__ void cuCvecmul_into(cuComplex * v, cuComplex * w, const int dimsize) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < dimsize)
		v[idx] = cuCmulf(v[idx], w[idx]);
	__syncthreads();
}

__global__ void cuCfindpos(cuComplex * v, unsigned int * occ, const int dim, const int pattlen) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float val;
	int width;

	//(vec1.dimsize - pattlen + i) % vec1.dimsize

	if (idx < dim) {
		val = (cuCrealf(v[idx]) / dim) - (float) pattlen;
		val = ((val < 0) ? -val : val);
	}
	__syncthreads();

	if (idx < dim) {
		if (val < 0.000122)
			if (idx + pattlen >= dim) {
				occ[idx] = idx + pattlen - dim;
			}
			else {
				occ[idx] = idx + pattlen;
			}
		else
			occ[idx] = dim;
	}
	__syncthreads();


	int t1, t2;
	for (width = (dim >> 1); width > 0; width >>= 1) {
		if (idx < width) {
			t1 = occ[idx];
			t2 = occ[idx + width];
			occ[idx] = ( (t1 < t2) ? t1 : t2);
		}
		__syncthreads();
	}

}

/* 2567
So Satan spake, and him Beelzebub
Thus answer'd. Leader of those Armies bright,
Which but th' Onmipotent none could have foyld,
If once they hear that voyce, thir liveliest pledge
Of hope in fears and dangers, heard so oft
In worst extreams, and on the perilous edge
Of battel when it rag'd, in all assaults
Thir surest signal, they will soon resume
New courage and revive, though now they lye
Groveling and prostrate on yon Lake of Fire,
As we erewhile, astounded and amaz'd,
No wonder, fall'n such a pernicious highth.
He scarce had ceas't when the superiour Fiend
Was moving toward the shoar; his ponderous shield
Ethereal temper, massy, large and round,
Behind him cast; the broad circumference
Hung on his shoulders like the Moon, whose Orb
Through Optic Glass the Tuscan Artist views
At Ev'ning from the top of Fesole,
Or in Valdarno, to descry new Lands,
Rivers or Mountains in her spotty Globe.
His Spear, to equal which the tallest Pine
Hewn on Norwegian hills, to be the Mast
Of some great Ammiral, were but a wand,
He walkt with to support uneasie steps
Over the burning Marle, not like those steps
On Heavens Azure, and the torrid Clime
Smote on him sore besides, vaulted with Fire;
Nathless he so endur'd, till on the Beach
Of that inflamed Sea, he stood and call'd
His Legions, Angel Forms, who lay intrans't
Thick as Autumnal Leaves that strow the Brooks
In Vallombrosa, where th' Etrurian shades
High overarch't imbowr; or scatterd sedge
Afloat, when with fierce Winds Orion arm'd
Hath vext the Red-Sea Coast, whose waves orethrew
Busiris and his Memphian Chivalry,
While with perfidious hatred they pursu'd
The Sojourners of Goshen, who beheld
From the safe shore thir floating Carkases
And broken Chariot Wheels, so thick bestrown
Abject and lost lay these, covering the Flood,
Under amazement of thir hideous change.
He call'd so loud, that all the hollow Deep
Of Hell resounded. Princes, Potentates,
Warriers, the Flowr of Heav'n, once yours, now lost,
If such astonishment as this can sieze
Eternal spirits; or have ye chos'n this place
After the toyl of Battel to repose
Your wearied vertue, for the ease you find
To slumber here, as in the Vales of Heav'n?
Or in this abject posture have ye sworn
To adore the Conquerour? who now beholds
Cherube and Seraph rowling in the Flood
With scatter'd Arms and Ensigns, till anon
His swift pursuers from Heav'n Gates discern
Th' advantage, and descending tread us down
Thus drooping, or with linked Thunderbolts
Transfix us to the bottom of this Gulfe.
Awake, arise, or be for ever fall'n.
*/