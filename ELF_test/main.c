/*
 * main.c
 *
 *  Created on: 2012/11/08
 *      Author: sin
 */

#include <stdio.h>
#include <stdint.h>

typedef uint8_t 	uint8;
typedef uint16_t 	uint16;
typedef uint32_t 	uint32;
typedef int8_t 		int8;
typedef int16_t 	int16;
typedef int32_t		int32;

int main(int argc, char * argv[]) {
	uint16 sizen = 0;
	if ( argc > 1 )
		sizen = argc - 1;

	for(int i = 0; i < sizen; i++) {
		fprintf(stdout, "%s, \r\n", argv[i]);
	}
	fprintf(stdout, "\r\n");
}
