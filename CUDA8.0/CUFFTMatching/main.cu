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

#define VECTOR_MAXSIZE 16384

struct compvect {
	cuComplex * elem; // = cuFloatComplex (8 bytes)
	unsigned int dim;
};
typedef struct compvect compvect;

struct text {
	char * str;
	unsigned int size;
	unsigned int length;
};
typedef struct text text;

__global__ void cuCvecmul_into(cuComplex * x, cuComplex * y, const int pattlen, const int dim);
__global__ void cuCfindpos(cuComplex * v, unsigned long * occurrences, const int dim, const int pattlen);
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

	text1.length = min(strlen(argv[1]), VECTOR_MAXSIZE);
	text2.length = min(strlen(argv[2]), VECTOR_MAXSIZE);
	vecsize = smallestpow2(max(text1.length, text2.length));
	pattlen = min(text1.length, text2.length);

	text1.size = vecsize;
	text1.str = (char *)malloc(sizeof(char)*text1.size);
	strncpy(text1.str, argv[1], text1.size);  // possibly has no ternimator NULL
	text2.size = VECTOR_MAXSIZE;
	text2.str = (char *)malloc(sizeof(char)*text2.size);
	strncpy(text2.str, argv[2], text2.size); // possibly has no ternimator NULL


	printf("inputs: \"");
	for (int i = 0; i < text1.length; ++i)
		printf("%c", text1.str[i]);
	printf("\" (%d), \"", text1.length);
	for (int i = 0; i < text2.length; ++i)
		printf("%c", text2.str[i]);
	printf("\" (%d).\n", text2.length);


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
	cudaMalloc((void**)&devmemptr, sizeof(cufftComplex) * vecsize * 2);
	
	/* GPU thread allocation */
	dim3 grid(16, 1);
	dim3 block(VECTOR_MAXSIZE / 16, 1);

	/* GPU用メモリに転送 */
	cudaMalloc((void**)&devstrptr, sizeof(char) * vecsize);

	cudaMemcpy(devstrptr, text1.str, sizeof(char)*text1.length, cudaMemcpyHostToDevice);
	make_signal<<<grid, block>>>(devstrptr, text1.length, devmemptr, vecsize, 0);
	cudaMemcpy(devstrptr, text2.str, sizeof(char)*text1.length, cudaMemcpyHostToDevice);
	make_signal<<<grid, block>>>(devstrptr, text2.length, devmemptr+vecsize, vecsize, 1);

	cudaFree(devstrptr);

#ifdef DEBUG_VECTOR
	cudaMemcpy(vec1.elem, devmemptr, sizeof(cuComplex)*vecsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(vec2.elem, devmemptr+vecsize, sizeof(cuComplex)*vecsize, cudaMemcpyDeviceToHost);
	print_vector("text1 ", &vec1);
	print_vector("text2 ", &vec2);
#endif

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
	cuCvecmul_into << < grid, block >> > (devmemptr, devmemptr + vecsize, pattlen, vecsize);

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

//	cudaMalloc((void**)&devoccptr, sizeof(int) * vecsize);
	cuCfindpos <<< grid, block >>>(devmemptr, (unsigned long *) (devmemptr + vecsize), vecsize, pattlen);

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
	unsigned long pos[VECTOR_MAXSIZE];
	cudaMemcpy(pos, devmemptr+vecsize, sizeof(unsigned long) * vecsize, cudaMemcpyDeviceToHost);
#else
	unsigned long pos[1];
	cudaMemcpy(pos, devmemptr + vecsize, sizeof(unsigned long) * 1, cudaMemcpyDeviceToHost);
#endif

	/* タイマーを停止しかかった時間を表示 */
	sdkStopTimer(&timer);
	printf("\n------------\ncomputation time %f (ms)\n------------\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

#ifdef DEBUG_OCCURRENCES
	printf("\nResult: \n");
	for (int i = 0; i < min(vecsize, 32); i++)
		printf("%4d  ", i);
	putchar('\n');
	for (int i = 0; i < min(vecsize, 32); i++)
		printf(" %4lu,", pos[i]);
	putchar('\n');
#endif
	printf("\n");
	if (pos[0] < vecsize)
		printf("First occurrence is at %lu.\n", pos[0] - pattlen + 1);
	else
		printf("Could not find.\n");

	/* GPU用メモリ開放*/
	cudaFree(devmemptr);

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
			vec[dst] = cuCexpf(make_cuComplex(0, factor * (float)(str[idx]) / 256.0f));  
			// char value by rotated unit vector
		else
			vec[dst] = make_cuComplex(0, 0);
	}
	__syncthreads();
}

/* Print a vector of cuComplex as ordered pairs. */
void print_vector(const char *title, compvect *v) {
	unsigned int i;
	printf("%s (dim=%d):\n", title, v->dim);
	for (i = 0; i < min(v->dim, 27); i++)
		printf("%6d   ", i);
	putchar('\n');
	for (i = 0; i < min(v->dim, 27); i++)
		printf(" %7.2f,", cuCrealf(v->elem[i]));
	putchar('\n');
	for (i = 0; i < min(v->dim, 27); i++)
		printf(" %7.2f,", cuCimagf(v->elem[i]));
	putchar('\n');
/*
	for (i = 0; i < min(v->dim, 27); i++)
		printf(" %7.2f,", cuCabsf(v->elem[i]));
	printf("\n\n");
	*/
	return;
}

__global__ void cuCvecmul_into(cuComplex * v, cuComplex * w, const int pattlen, const int dimsize) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < dimsize)
		v[idx] = cuCdivf(cuCmulf(v[idx], w[idx]), make_cuComplex((float) dimsize, 0) );
	__syncthreads();
}

__global__ void cuCfindpos(cuComplex * v, unsigned long * occ, const int dim, const int pattlen) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float val;
	int width;

	//(vec1.dimsize - pattlen + i) % vec1.dimsize

	if (idx < dim) {
		val = cuCrealf(v[idx]) - (float) pattlen;
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


	unsigned long t1, t2;
	for (width = (dim >> 1); width > 0; width >>= 1) {
		if (idx < width) {
			t1 = occ[idx];
			t2 = occ[idx + width];
			occ[idx] = ( (t1 < t2) ? t1 : t2);
		}
		__syncthreads();
	}

}

/* 10561
If thou beest he; But O how fall'n! how chang'd
From him, who in the happy Realms of Light
Cloth'd with transcendent brightness didst out-shine
Myriads though bright: If he Whom mutual league,
United thoughts and counsels, equal hope
And hazard in the Glorious Enterprize,
Joynd with me once, now misery hath joynd
In equal ruin: into what Pit thou seest
From what highth fall'n, so much the stronger prov'd
He with his Thunder: and till then who knew
The force of those dire Arms? yet not for those,
Nor what the Potent Victor in his rage
Can else inflict, do I repent or change,
Though chang'd in outward lustre; that fixt mind
And high disdain, from sence of injur'd merit,
That with the mightiest rais'd me to contend,
And to the fierce contention brought along
Innumerable force of Spirits arm'd
That durst dislike his reign, and me preferring,
His utmost power with adverse power oppos'd
In dubious Battel on the Plains of Heav'n,
And shook his throne. What though the field be lost?
All is not lost; the unconquerable Will,
And study of revenge, immortal hate,
And courage never to submit or yield:
And what is else not to be overcome?
That Glory never shall his wrath or might
Extort from me. To bow and sue for grace
With suppliant knee, and deifie his power,
Who from the terrour of this Arm so late
Doubted his Empire, that were low indeed,
That were an ignominy and shame beneath
This downfall; since by Fate the strength of Gods
And this Empyreal substance cannot fail,
Since through experience of this great event
In Arms not worse, in foresight much advanc't,
We may with more successful hope resolve
To wage by force or guile eternal Warr
Irreconcileable, to our grand Foe,
Who now triumphs, and in th' excess of joy
Sole reigning holds the Tyranny of Heav'n.
So spake th' Apostate Angel, though in pain,
Vaunting aloud, but rackt with deep despare:
And him thus answer'd soon his bold Compeer.
O Prince, O Chief of many Throned Powers,
That led th' imbattelld Seraphim to Warr
Under thy conduct, and in dreadful deeds
Fearless, endanger'd Heav'ns perpetual King;
And put to proof his high Supremacy,
Whether upheld by strength, or Chance, or Fate,
Too well I see and rue the dire event,
That with sad overthrow and foul defeat
Hath lost us Heav'n, and all this mighty Host
In horrible destruction laid thus low,
As far as Gods and Heav'nly Essences
Can perish: for the mind and spirit remains
Invincible, and vigour soon returns,
Though all our Glory extinct, and happy state
Here swallow'd up in endless misery.
But what if he our Conquerour, (whom I now
Of force believe Almighty, since no less
Then such could hav orepow'rd such force as ours)
Have left us this our spirit and strength intire
Strongly to suffer and support our pains,
That we may so suffice his vengeful ire,
Or do him mightier service as his thralls
By right of Warr, what e're his business be
Here in the heart of Hell to work in Fire,
Or do his Errands in the gloomy Deep;
What can it then avail though yet we feel
Strength undiminisht, or eternal being
To undergo eternal punishment?
Whereto with speedy words th' Arch-fiend reply'd.
Fall'n Cherube, to be weak is miserable
Doing or Suffering: but of this be sure,
To do ought good never will be our task,
But ever to do ill our sole delight,
As being the contrary to his high will
Whom we resist. If then his Providence
Out of our evil seek to bring forth good,
Our labour must be to pervert that end,
And out of good still to find means of evil;
Which oft times may succeed, so as perhaps
Shall grieve him, if I fail not, and disturb
His inmost counsels from thir destind aim.
But see the angry Victor hath recall'd
His Ministers of vengeance and pursuit
Back to the Gates of Heav'n: The Sulphurous Hail
Shot after us in storm, oreblown hath laid
The fiery Surge, that from the Precipice
Of Heav'n receiv'd us falling, and the Thunder,
Wing'd with red Lightning and impetuous rage,
Perhaps hath spent his shafts, and ceases now
To bellow through the vast and boundless Deep.
Let us not slip th' occasion, whether scorn,
Or satiate fury yield it from our Foe.
Seest thou yon dreary Plain, forlorn and wilde,
The seat of desolation, voyd of light,
Save what the glimmering of these livid flames
Casts pale and dreadful? Thither let us tend
From off the tossing of these fiery waves,
There rest, if any rest can harbour there,
And reassembling our afflicted Powers,
Consult how we may henceforth most offend
Our Enemy, our own loss how repair,
How overcome this dire Calamity,
What reinforcement we may gain from Hope,
If not what resolution from despare.
Thus Satan talking to his neerest Mate
With Head up-lift above the wave, and Eyes
That sparkling blaz'd, his other Parts besides
Prone on the Flood, extended long and large
Lay floating many a rood, in bulk as huge
As whom the Fables name of monstrous size,
Titanian, or Earth-born, that warr'd on Jove,
Briareos or Typhon, whom the Den
By ancient Tarsus held, or that Sea-beast
Leviathan, which God of all his works
Created hugest that swim th' Ocean stream:
Him haply slumbring on the Norway foam
The Pilot of some small night-founder'd Skiff,
Deeming some Island, oft, as Sea-men tell,
With fixed Anchor in his skaly rind
Moors by his side under the Lee, while Night
Invests the Sea, and wished Morn delayes:
So stretcht out huge in length the Arch-fiend lay
Chain'd on the burning Lake, nor ever thence
Had ris'n or heav'd his head, but that the will
And high permission of all-ruling Heaven
Left him at large to his own dark designs,
That with reiterated crimes he might
Heap on himself damnation, while he sought
Evil to others, and enrag'd might see
How all his malice serv'd but to bring forth
Infinite goodness, grace and mercy shewn
On Man by him seduc't, but on himself
Treble confusion, wrath and vengeance pour'd.
Forthwith upright he rears from off the Pool
His mighty Stature; on each hand the flames
Drivn backward slope thir pointing spires, and rowld
In billows, leave i'th' midst a horrid Vale.
Then with expanded wings he stears his flight
Aloft, incumbent on the dusky Air
That felt unusual weight, till on dry Land
He lights, if it were Land that ever burn'd
With solid, as the Lake with liquid fire;
And such appear'd in hue, as when the force
Of subterranean wind transports a Hill
Torn from Pelorus, or the shatter'd side
Of thundring Ætna, whose combustible
And fewel'd entrals thence conceiving Fire,
Sublim'd with Mineral fury, aid the Winds,
And leave a singed bottom all involv'd
With stench and smoak: Such resting found the sole
Of unblest feet.  Him followed his next Mate,
Both glorying to have scap't the Stygian flood
As Gods, and by thir own recover'd strength,
Not by the sufferance of supernal Power.
Is this the Region, this the Soil, the Clime,
Said then the lost Arch-Angel, this the seat
That we must change for Heav'n, this mournful gloom
For that celestial light? Be it so, since he
Who now is Sovran can dispose and bid
What shall be right: fardest from him is best
Whom reason hath equald, force hath made supream
Above his equals. Farewel happy Fields
Where Joy for ever dwells: Hail horrours, hail
Infernal world, and thou profoundest Hell
Receive thy new Possessor: One who brings
A mind not to be chang'd by Place or Time.
The mind is its own place, and in it self
Can make a Heav'n of Hell, a Hell of Heav'n.
What matter where, if I be still the same,
And what I should be, all but less then he
Whom Thunder hath made greater? Here at least
We shall be free; th' Almighty hath not built
Here for his envy, will not drive us hence:
Here we may reign secure, and in my choyce
To reign is worth ambition though in Hell:
Better to reign in Hell, then serve in Heav'n.
But wherefore let we then our faithful friends,
Th' associates and copartners of our loss
Lye thus astonisht on th' oblivious Pool,
And call them not to share with us their part
In this unhappy Mansion, or once more
With rallied Arms to try what may be yet
Regaind in Heav'n, or what more lost 

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