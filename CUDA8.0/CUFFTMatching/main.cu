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


#define TEXTBUFFER_LENGTH 512
#define VECTOR_MAXSIZE 512

struct dcompvec {
	cufftComplex * elem;
	int dimsize;
};
typedef struct dcompvec dcompvec;
const int CONJ_REVERSE = 1;

struct text {
	char str[TEXTBUFFER_LENGTH];
	int length;
};
typedef struct text text;

__global__ void cuCvecmul_into(cuComplex * x, cuComplex * y, const int dimsize);
__global__ void cuCfindpos(cuComplex * v, const int dimsize, const int pattlen);
long smallestpow2(const long n) {
	long t = 1;
	while (t < n) {
		t <<= 1;
	}
	return t;
}

#define min(x,y)  ( ((x) < (y) ? (x) : (y)) )
#define max(x,y)  ( ((x) < (y) ? (y) : (x)) )
#define abs(x)  ( (x) < 0 ? (-(x)) : (x) )

int make_signal(text * text1, int dimsize, dcompvec * vec, const int flag);
void print_vector(const char *title, cufftComplex *x, int n);

void cufft(cufftComplex * vec, int n, int inverseflag);

__host__ __device__ static __inline__ cuComplex cuCexpf(cuComplex x)
{
	float factor = expf(x.x);
	return make_cuComplex(factor * cosf(x.y), factor * sinf(x.y));
}

int main(int argc, char * argv[]) {
	text text1, text2;
	dcompvec vec1, vec2;
	int vecsize;
	int pattlen;

	if ( argc != 3)
		exit(EXIT_FAILURE);

	strncpy(text1.str, argv[1], TEXTBUFFER_LENGTH);
	text1.length = min(strlen(text1.str), TEXTBUFFER_LENGTH);
	strncpy(text2.str, argv[2], TEXTBUFFER_LENGTH);
	text2.length = min(strlen(text2.str), TEXTBUFFER_LENGTH);

	printf("inputs: \"%s\", \"%s\" \n", text1.str, text2.str);
	vecsize = smallestpow2(min(VECTOR_MAXSIZE, max(text1.length, text2.length)));
	pattlen = min(text1.length, text2.length);

	make_signal(&text1, vecsize, &vec1, !CONJ_REVERSE);
	make_signal(&text2, vecsize, &vec2, CONJ_REVERSE);
	/* FFT, iFFT of v[]: */
	print_vector("text1 ", vec1.elem, vec1.dimsize);
	print_vector("text2 ", vec2.elem, vec2.dimsize);

	/*タイマーを作成して計測開始*/
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	/* GPU用メモリ割り当て */
	cufftComplex *devmemptr;
	/* バッチ数 2 (vec1.elem, vec2.elem)*/
	cudaMalloc((void**)&devmemptr, sizeof(cufftComplex) * vecsize * 2 );

	/* GPU用メモリに転送 */
	cudaMemcpy(devmemptr, vec1.elem, sizeof(cufftComplex)* vecsize, cudaMemcpyHostToDevice);
	cudaMemcpy(devmemptr+vecsize, vec2.elem, sizeof(cufftComplex)* vecsize, cudaMemcpyHostToDevice);

	/* 1D FFT plan作成 */
	cufftHandle plan2way, plan1inv;
	cufftPlan1d(&plan2way, vecsize, CUFFT_C2C, 2);
	cufftPlan1d(&plan1inv, vecsize, CUFFT_C2C, 1);

	cufftExecC2C(plan2way, devmemptr, devmemptr, CUFFT_FORWARD);

#ifdef DEBUG
	/* 計算結果をGPUメモリから転送して表示 */
	cudaMemcpy(vec1.elem, devmemptr, sizeof(cufftComplex)*vec1.dimsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(vec2.elem, devmemptr+vecsize, sizeof(cufftComplex)*vec2.dimsize, cudaMemcpyDeviceToHost);
	print_vector("fft1: 2 ", vec1.elem, vec1.dimsize);
	print_vector("fft2 ", vec2.elem, vec2.dimsize);
#endif

	/* ベクトルの積をとる */
	dim3 grid(16, 1);
	dim3 block(VECTOR_MAXSIZE/16, 1);
	cuCvecmul_into <<< grid, block >>> (devmemptr, devmemptr+vecsize, vecsize);

#ifdef DEBUG
	/* 計算結果をGPUメモリから転送して表示 */
	cudaMemcpy(vec1.elem, devmemptr, sizeof(cufftComplex)* vecsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(vec2.elem, devmemptr + vecsize, sizeof(cufftComplex)*vecsize, cudaMemcpyDeviceToHost);
	print_vector("prod ", vec1.elem, vec1.dimsize);
#endif

	cufftExecC2C(plan1inv, devmemptr, devmemptr, CUFFT_INVERSE);


	cudaMemcpy(vec1.elem, devmemptr, sizeof(cufftComplex)* vecsize, cudaMemcpyDeviceToHost);
	print_vector("iFFT ", vec1.elem, vecsize);

	cuCfindpos<<<grid, block>>>(devmemptr, vecsize, pattlen);

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
	cudaMemcpy(vec1.elem, devmemptr, sizeof(cufftComplex)* vecsize, cudaMemcpyDeviceToHost);
	int pos = (int) cuCrealf(vec1.elem[0]);

	/*タイマーを停止しかかった時間を表示*/
	sdkStopTimer(&timer);
	printf("計算時間 =%f(ms)\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

	print_vector(" pos ", vec1.elem, vecsize);
	if (pos < vecsize) {
		printf("The 1st occurrence at: %d.\n", pos);
	}
	else {
		printf("None.\n");
	}

	/* GPU用メモリ開放 */
	cudaFree(devmemptr);

	/* CUFFT plan削除 */
	cufftDestroy(plan2way);
	cufftDestroy(plan2way);

	free(vec1.elem);
	free(vec2.elem);

	exit(EXIT_SUCCESS);
}


int make_signal(text * str, const int dimsize, dcompvec * vec, const int flag) {
	int len;
	int dst;
	float factor;

	// the first as normal
	vec->dimsize = dimsize;
	vec->elem = (cuComplex*)malloc(sizeof(cuComplex)*dimsize);
	len = str->length;
	for (int i = 0; i < vec->dimsize; ++i) {
		if (!flag) {
			dst = i;
			factor = 2 * 3.14159265358979323846264338327950288F;
		}
		else {
			dst = vec->dimsize - i - 1;
			factor = -2 * 3.14159265358979323846264338327950288F;
		}
		if (i < len)
			vec->elem[dst] = cuCexpf(make_cuComplex(0, factor * (float)(str->str[i]) / 256.0f));  // by rotated unit vector
																								  // (*array)[i] = (float)(str[i]) / 128.0f  ;  // by char value
		else
			vec->elem[dst] = make_cuComplex(0, 0);
	}
	return 1;
}

/* Print a vector of complexes as ordered pairs. */
void print_vector(const char *title, cufftComplex *x, int n) {
	int i;
	printf("%s (dim=%d):\n", title, n);
	for (i = 0; i < n; i++)
		printf("%5d    ", i);
	putchar('\n');
	for (i = 0; i < n; i++)
		printf(" %7.3f,", cuCrealf(x[i]));
	putchar('\n');
	for (i = 0; i < n; i++)
		printf(" %7.3f,", cuCimagf(x[i]));
	putchar('\n');
	for (i = 0; i < n; i++)
		printf(" %7.3f,", cuCabsf(x[i]));
	printf("\n\n");
	return;
}

__global__ void cuCvecmul_into(cuComplex * v, cuComplex * w, const int dimsize) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if ( idx < dimsize )
		v[idx] = cuCmulf(v[idx], w[idx]);
	__syncthreads();
}

__global__ void cuCfindpos(cuComplex * v, const int dimsize, const int pattlen) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float val;
	int width;

	//(vec1.dimsize - pattlen + i) % vec1.dimsize

	if (idx < dimsize) {
		int i = dimsize - pattlen + idx;
		if (i >= dimsize)
			i %= dimsize;
		val = cuCrealf(v[i]) / dimsize - (float)pattlen;
		if (val < 0)
			val = -val;
	}
	__syncthreads();
	if (idx < dimsize) {
		if (val < 0.000122)
			v[idx] = make_cuComplex(idx, 0.0);
		else
			v[idx] = make_cuComplex(dimsize, 0.0);
	}
	__syncthreads();

	float pos1, pos2;
	for (width = (dimsize >> 1); width > 0; width >>= 1) {
		if ( idx < width ) {
			pos1 = cuCrealf(v[idx]);
			pos2 = cuCrealf(v[idx + width]);
			v[idx] = make_cuComplex( (pos1 < pos2 ? pos1 : pos2), 0.0);
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