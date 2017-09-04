#include <avr/io.h>
#include <util/delay.h>


// Necessary reset/interrrupt vectors (jump tables) 
// will be automatically generated and linked with 
// your program codes. 
// Initialization routine of the stack pointer 
// register (SPL, SPH) will be provided and placed
// just after the reset event and just before your code.

int main(void) {
	int count = 0;
    DDRD = 0xff;           /* make all the pins output */
	PORTD = 0x00;
    for(;;){
		int i;
        for(i = 0; i < 10; i++){
            _delay_ms(50);  /* max is 262.14 ms / F_CPU in MHz */
        }
		count = count % 8;
		PORTD ^= (0xaa << count);    /* toggle the LED */
		count++;
    }
    return 0;               /* never reached */
}
