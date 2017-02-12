#include "cu_utils.h"


unsigned int nlz32(unsigned int x)
{
	/* Hacker's Delight 2nd by H. S. Warren Jr., 5.3, p. 104 -- */
	double d = (double)x + 0.5;
	unsigned int *p = ((unsigned int*)&d) + 1;
	return 0x41e - (*p >> 20);  // 31 - ((*(p+1)>>20) - 0x3FF)
}
