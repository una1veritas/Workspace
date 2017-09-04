#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

// Necessary reset/interrrupt vectors (jump tables) 
// will be automatically generated and linked with 
// your program codes. 
// Initialization routine of the stack pointer 
// register (SPL, SPH) will be provided and placed
// just after the reset event and just before your code.

#include "Wiring.h"
// typedef unsigned char byte;
// typedef unsigned int word;

#include "String.h"

int main() {
	sbuf.print("hgf");
	for (;;) {
	}
}
