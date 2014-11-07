#include <stdio.h>
#include <stdlib.h>


int Euclidean(int a, int b) {
	int c;
	
	printf("a = %d, b= %d\n", a, b);
	while ( (c = a % b) != 0 ) {
		a = b;
		b = c;
		printf("a = %d, b= %d\n", a, b);
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
	int a = 0, b = 0, gcd = 0;

	if (argc >= 3 ) {
	  a = atoi(argv[1]);
	  b = atoi(argv[2]);
	}
    printf("Input: %d, %d\n", a, b);
	
    	printf("Computing by Euclidean algorithm...\n");
    	gcd = Euclidean(a, b);
    	printf("GCD: %d\n", gcd);
	
//	printf("Computing by naive enumeration algorithm...\n");
//	gcd = Naive(a, b);
//	printf("GCD: %d\n", gcd);

	if (gcd == 0 ) {
	  return 1;
	}
		
    return 0;
}
