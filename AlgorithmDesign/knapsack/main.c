#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int bestPrice_recursive(int list[], int budget) {
	int s, ss;
	
	if ( *list == 0 ) 
		return 0;
	s = bestPrice_recursive(list+1, budget);
	if ( *list > budget) 
		return s;
	ss = *list + bestPrice_recursive(list+1, budget - *list);
	if (ss > s) 
		return ss;
	return s;
}

int bestPrice_dp(int list[], int budget) {
	int i, n;
	int b;
	
	for (n = 1; list[n] != 0; n++);
	int best[n][budget+1];

	for (b = 0; b <= budget; b++) {
		if (list[0] > b) {
			best[0][b] = 0;
		} else {
			best[0][b] = list[0];
		}
	}

	for (i = 1; i < n; i++) {
		for (b = 0; b <= budget; b++) {
			if (list[i] > b) {
				best[i][b] = best[i-1][b];
				continue;
			}
			if ( best[i-1][b] > list[i] + best[i-1][b-list[i]] ) {
				best[i][b] = best[i-1][b];
			} else {
				best[i][b] = list[i] + best[i-1][b-list[i]];
			}
		}
	}
	
	return best[n-1][budget];
}

int main (int argc, const char * argv[]) {
	int budget;
	int itemCount;
	int i, s, totalPrice;
	clock_t swatch;
	
	budget = atoi(argv[1]);
	itemCount = argc - 2;
	int priceList[itemCount + 1];
	//int * priceList; priceList = (int *) malloc(sizeof(int) * (itemCount+1) );
	for (i = 0, s = 2; i < itemCount; i++, s++) {
		priceList[i] = atoi(argv[s]);
	}
	priceList[i] = 0; // the end mark.
	
	// Show our input.
	printf("%d yen for %d items.\n", budget, itemCount);
	for (i = 0; priceList[i] != 0; i++) {
		printf("%d, ", priceList[i]);
	}
	printf("\n");
	
	// compute.
	swatch = clock();
	totalPrice = bestPrice_recursive(priceList, budget);
	swatch = clock() - swatch;
	// Show the result.
	printf("bought totally %d yen.\n", totalPrice);
	printf("By recursion: %f\n", (double) swatch / CLOCKS_PER_SEC);

	swatch = clock();
	totalPrice = bestPrice_dp(priceList, budget);
	swatch = clock() - swatch;
	// Show the result.
	printf("bought totally %d yen.\n", totalPrice);
	printf("By dp: %f\n", (double) swatch / CLOCKS_PER_SEC);
    return 0;
}
