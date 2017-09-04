#include <avr/io.h>
#include <util/delay.h>


// Necessary reset/interrrupt vectors (jump tables) 
// will be automatically generated and linked with 
// your program codes. 
// Initialization routine of the stack pointer 
// register (SPL, SPH) will be provided and placed
// just after the reset event and just before your code.

int main(void) {
    DDRD = 0xff;           /* make all the pins output */
	PORTD = 0x00;
    for(;;){
		PORTD = 0xaa;
    }
    return 0;               /* never reached */
}
