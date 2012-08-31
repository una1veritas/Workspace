#include <stdio.h>

unsigned int modPower(unsigned int, unsigned int, unsigned int);

int main (int argc, const char * argv[]) {
	unsigned int ans = 1, p = 1, m = 1;
	unsigned char * ptr;
	
	if ( argv[1] != 0 )
		p = atoi(argv[1]);
	if ( argv[2] != 0 )
		m = atoi(argv[2]);
	ptr = (unsigned char *) argv[3];

	printf("Input: to the power of %d, modulo by %d\r\n", p, m);
	printf("Message: %s\r\n", ptr);
	
	for ( ; ptr != 0 && *ptr != 0; ptr++) {
		ans = modPower(*ptr, p, m);
		printf(" '%c' %d \t($%x) ^%d mod %d = %d \t($%x)\r\n", *ptr, *ptr, *ptr, p, m, ans, ans, ans);
	}
    return 0;
}

unsigned int modPower(unsigned int b, unsigned int p, unsigned int m) {
	//assumes m is no less than 1
	if ( p == 0 ) 
		return 1;
	if ( p == 1 ) 
		return (b % m);
	return (modPower(b,p/2,m) * modPower(b,p-(p/2),m)) % m;
}
