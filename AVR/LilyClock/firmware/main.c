#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

//#include <stdio.h>

// Necessary reset/interrrupt vectors (jump tables) 
// will be automatically generated and linked with 
// your program codes. 
// Initialization routine of the stack pointer 
// register (SPL, SPH) will be provided and placed
// just after the reset event and just before your code.

#include "Wiring.h"

//USART Serial
#define BAUD 9600

// usual globals
void SPIInit(void);
void SPITransmit(char data);
void serialPrint(char * msg);

byte gsdata[48];
void gsprep(byte i, word v);

#define floor(x) ((long)(x))

long JulianDay(int y, int m, int d, int hour) {
	int a, b;
	if ( m < 3 ) {
		m += 12;
		y -= 1;
	}
	a = y/100; b = 2 - a + (a/4);
	return 365L * y + (long)(30.6001*(m+1)) + d + (long)(1720994.5)+ b;
}

void init() {
	// Timer 1 for RTC
	bitSet(TCCR1B,WGM12);
	OCR1AH = highByte(15625);
	OCR1AL = lowByte(15625);
	TCCR1B &= ~(0b111<<CS10);
	TCCR1B |= (0b011<<CS10);
	bitSet(TIMSK1,OCIE1A);
	
	//SPI setup
	SPIInit();

	// Pin Change Interrupt on PB0
	bitClear(DDRB, PB0);
	bitSet(PCICR, PCIE0);
	bitSet(PCMSK0, PCINT0);
	
	bitClear(DDRD, PD3);
	bitClear(DDRD, PD7);
	bitSet(PORTD, PD3);
	bitSet(PORTD, PD7);
	
}

// global for interrupts
volatile byte ottosec = 0;
volatile byte pc0int = 0;
volatile byte ready = 0;

ISR(TIMER1_COMPA_vect) {
	ottosec++;
}

ISR(PCINT0_vect) {
	pc0int++;
	
	if (pc0int > 3) {
		bitSet(PORTB,PB2);
		ready = !0;
		// asm("nop");
		bitClear(PORTB,PB2);
		pc0int = 0;
	}
}


int main(void) {
	byte gauss[] = {128, 127, 123, 116, 108, 98, 87, 75, 64, 53, 43, 35, 27, 21, 15, 11, 8, 6, 4, 3, 2, 1, 1, 0};
	byte prev = ottosec;
	long rtsec = (10L*3600+10*60);  // in seconds; the 86400th of the day
	long day = 0;
	init();
	
//	beginSerial(BAUD); // overides DDRD
	
	int i;
	word d, gsval;
	
	// prepare
	for (i = 0; i < 48; i++)
        gsprep(i,0);
	
	
	sei();
    for(;;){
		if ( ready ) {
			ready = 0;
			for (i = 0; i < 48; i++) 
				SPITransmit(gsdata[i]);
			//
			continue;
		}
		if ( prev == ottosec )
			continue;
		else {
			if ( ottosec > 7 ) {
				ottosec -= 8;
				rtsec++;
				if ( rtsec > 86400L-1 ) {
					rtsec -= 86400L;
					day++;
				}
			}
			prev = ottosec;

		}
		
		if ( !bitRead(PIND, PD3) ) {
			rtsec += 60;
		} else if ( !bitRead(PIND, PD7) ) {
			rtsec += 300;
		}
		
		for (i = 0; i < 32; i++) {
			gsval = 0;

			d = abs(rtsec/180-i*15);
			if ( d > 239 )
				d = 480 - d;
			gsval = gauss[min(d,23)]<<1;
			
			d = abs((rtsec/60)%60*8 - i*15);
			if ( d > 239 )
				d = 480 - d;
			if (gauss[min(d,23)] > 0) {
				gsval += gauss[min(d<<1,23)] * (2+abs(((rtsec % 4)*4 + ottosec / 2) - 8))/4;
			}

			d = abs((rtsec%60)*8 + ottosec - i*15 );
			if ( d > 239 )
				d = 480 - d;
			gsval += gauss[min(d<<1,23)];
			
			gsprep(i, gsval);
		}
		
		/*
		sprintf(strbuf, "%02d:%02d:%02d",clock.hours,clock.minutes,clock.seconds);
		serialPrint(strbuf);
		serialPrint("\r\n");
		*/
    }
    return 0;               /* never reached */
}

/*
0, 1, 
1, 2, 
3, 4, 
4, 5, 
6, 7, 
7, 8,
9, 10,
10, 11,
12, 13,
13, 14,
15, 16,
16, 17,
18, 19,
19, 20,
21, 22,
22, 23
*/
void gsprep(byte i, word v) {
	byte odd = 1 & i;
	i = i*3/2;
	v = min(v, 4095);
	if (odd) {
		gsdata[i] &= 0xf0;
		gsdata[i] |= 0x0f & (v>>8);
		gsdata[i+1] = (byte) v;
	} else {
		gsdata[i] = lowByte(v>>4);
		gsdata[i+1] &= 0x0f;
		gsdata[i+1] |= 0xf0 &(v<<4);
	}
}


void beginSerial(long baudrate) {
	unsigned int ubrr = F_CPU/16/baudrate-1;
	/*Set baud rate */
	UBRR0H = highByte(ubrr);
	UBRR0L = lowByte(ubrr);
	/* Enable receiver and transmitter */
//	UCSR0B = (1<<RXEN0)|(1<<TXEN0);
	UCSR0B = (0<<RXEN0)|(1<<TXEN0);
	/* Set frame format: 8data, 2stop bit */
	UCSR0C = (0<<USBS0)|(3<<UCSZ00);
}

void serialWrite(unsigned char data ){
	/* Wait for empty transmit buffer */
	while ( !( UCSR0A & (1<<UDRE0)) ) ;
	/* Put data into buffer, sends the data */
	UDR0 = data;
}

void serialPrint(char * msg) {
	for (;*msg != '\0';msg++) {
		serialWrite((unsigned char) *msg);
	}
}


void SPIInit(void) {
	/* Set MOSI and SCK output, all others input */
	DDRB = (1<<PB3)|(1<<PB5)|(1<<PB2);
	/* Enable SPI, Master, set clock rate fck/4 */
	SPCR = (1<<SPE)|(1<<MSTR)|(0b00<<SPR0);
}

void SPITransmit(char cData) {
	/* Start transmission */
	SPDR = cData;
	/* Wait for transmission complete */
	while(!(SPSR & (1<<SPIF))) ;
}
