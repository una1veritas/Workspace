#include <stdio.h>
#include <string.h>

#include "des.h"

int main (int argc, const char * argv[]) {
    // insert code here...
    printf("Hello, World!\n");
	
	byte buf[8], code[8];
	byte key[8] = { 0x5C, 0x78, 0x51, 0xCB, 0x6A, 0xB5, 0x4E, 0x5E};
	int index;
	
	memcpy(buf, "12345678", 8);
	
	desDecrypt(buf, code);
	
	printf("code = ");
	for (index = 0; index < 8; index++) {
		printf("%02x", code[index]);
	}
	printf("\n");
    return 0;
}
