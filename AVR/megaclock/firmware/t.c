#include <stdio.h>

int main(int argc, char * argv[]) {
	int a[argc];
	printf("%d/%d.\n", (int)sizeof(a), (int)sizeof(a[0]));
	return 0;
}

