#include <avr/io.h>

void waitamoment() {
    int m, u;
    for (m = 0; m < 750; m++) 
		for (u = 0; u < 1000; u++)
			asm("nop");
}

int main(void) {
    DDRD = 0xff;           /* make all the pins output */
    PORTD = 0x00;
    for(;;){
    	PORTD ^= 0xff;    /* toggle the LEDs */
		waitamoment ();
    }
    return 0;               /* never reached */
}
