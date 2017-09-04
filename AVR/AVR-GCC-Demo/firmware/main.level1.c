#include <avr/io.h>
#include <util/delay.h>


// Necessary reset/interrrupt vectors (jump tables) 
// will be automatically generated and linked with 
// your program codes. 
// Initialization routine of the stack pointer 
// register (SPL, SPH) will be provided and placed
// just after the reset event and just before your code.
// _delay_ms( ) is provided in util/delay.h
int main(void) {
    DDRD = 0xff;           /* make all the pins output */
	PORTD = 0x00;
    for(;;){
		int i;
        for(i = 0; i < 10; i++){
            _delay_ms(50);  /* max is 262.14 ms / F_CPU in MHz */
        }
		PORTD ^= 0xaa;    /* toggle the LED */
    }
    return 0;               /* never reached */
}
