#include <stdio.h>
#include <stdlib.h>


int Euclidean(int a, int b) {
	int c;
	
	while ( (c = a % b) != 0 ) {
		a = b;
		b = c;
	}
	return b;
}

int Naive(int a, int b) {
	int c;
	
	if (a > b) 
		c = b;
	else 
		c = a;
	for ( ; (a % c) != 0 || ( (b % c) != 0); c--);
	return c;
}

int main (int argc, const char * argv[]) {
	int a, b, gcd;
	
	a = atoi(argv[1]);
	b = atoi(argv[2]);
    printf("Input: %d, %d\n", a, b);
	
	printf("Computing by naive enumeration algorithm...\n");
	gcd = Naive(a, b);
	printf("GCD: %d\n", gcd);
		
    return 0;
}
