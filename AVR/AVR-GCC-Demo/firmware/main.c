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
/*
//USART Serial
#define BAUD 9600

void SerialBegin(unsigned int brate);
void SerialWrite(unsigned char data );
void SerialPrint(char * msg);
char strbuf[32];
*/

volatile word gscount = 0;
volatile byte sccount = 0;

#define XLAT  PB4
#define SCLK  PB7
#define SIN  PB6
#define GSCLK PB2
#define BLANK PB3
/*
volatile word t0ovf = 0;
volatile byte bp = 0;

ISR(TIMER0_OVF_vect) {
	bitWrite(PORTD,PD2, !bitRead(PORTD,PD2));
	t0ovf++;
	if ( t0ovf > 8191) {
		t0ovf = 0;
	}
	bp++;
}
*/
ISR(TIMER1_COMPA_vect) {
	bitSet(PORTB,BLANK);
	bitSet(PORTB,XLAT);
	asm("nop");
	bitClear(PORTB,XLAT);
}

int main(void) {

	bitClear(TCCR0A, COM0A1); bitSet(TCCR0A,COM0A0);
	bitSet(TCCR0A, WGM01); bitClear(TCCR1B, WGM00);
	OCR0A = 4;
	TCCR0B |= 0b001 << CS00;

	
	//bitClear(TCCR1A, COM1A1); bitSet(TCCR1A,COM1A0);
	bitClear(TCCR1B, WGM13); bitSet(TCCR1B, WGM12);
	OCR1AH = highByte(4096);
	OCR1AL = lowByte(4096);
	TCCR1B |= (0b010<<CS10);
	bitSet(TIMSK,OCIE1A);
//	SerialBegin(BAUD);
	
	//
    DDRD = 0xff;           /* make all the pins output */
	PORTD = 0x00;
	DDRB = 0xff;
	PORTB = 0x00;
//	bitSet(DDRB,PB2);
	
	//
	word gsval[16];
	byte data;
	word wdata;
	
	int i;
	for (i = 0; i < 16; i++) {
		gsval[i] = i<<0;
	}	
	bitSet(PORTB,GSCLK);
	bitClear(PORTB,GSCLK);

	sei();
    for(;;){
		if ( bitRead(PORTB, BLANK) /* && (TCNT1 < 512) */) {
			bitClear(PORTB,BLANK);
			for (sccount = 0; sccount < 24 /* 192/8 */; sccount++) {
				i = sccount/3*2;
				if (sccount % 3 == 0) {
					wdata = gsval[i];
					data = (wdata >> 4);
				} else if (sccount % 3 == 1) {
					data = 0x0f & wdata;
					data <<= 4;
					data |= 0x0f & highByte(gsval[i+1]);
				} else {
					data = lowByte(gsval[i+1]);
				}
				USIDR = data;
				USISR = (1<<USIOIF);
				//data = 
				while ( !(USISR & (1<<USIOIF)) ) {
					USICR = (0b01<<USIWM0) | (1<<USICS1) | (0<<USICS0) | (1<<USICLK) | (1<<USITC);;
					/*
					if ( bitRead(PINB,SCLK) ) // sync to the SPI clk
						bitSet(PORTB,GSCLK);
					else
						bitClear(PORTB,GSCLK);
					 */
				}
				data = USIDR;
			}
		}
		/*
		bitSet(PORTB,BLANK);
		bitSet(PORTB,XLAT);
		asm("nop");
		bitClear(PORTB,XLAT);
		 */
	}
    return 0;               /* never reached */
}

/*
void SerialBegin(unsigned int baudrate) {
	unsigned int ubrr = F_CPU/16/baudrate-1;
	// Set baud rate
	UBRR0H = highByte(ubrr);
	UBRR0L = lowByte(ubrr);
	// Enable receiver and transmitter
	UCSR0B = (1<<RXEN0)|(1<<TXEN0);
	// Set frame format: 8data, 2stop bit 
	UCSR0C = (1<<USBS0)|(3<<UCSZ00);
}

void SerialWrite(unsigned char data ){
	// Wait for empty transmit buffer 
	while ( !( UCSR0A & (1<<UDRE0)) ) ;
	// Put data into buffer, sends the data 
	UDR0 = data;
}

void SerialPrint(char * msg) {
	for (;*msg != '\0';msg++) {
		SerialWrite((unsigned char) *msg);
	}
}
*/