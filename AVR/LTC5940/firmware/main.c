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

//USART Serial
#define BAUDRATE 19200
#define UBRRVAL ((word)(F_CPU/16/BAUDRATE-1))

void SerialBegin();
void SerialWrite(byte data );
void SerialPrint(char * msg);
byte SerialReceive();
byte SerialRxComplete();
void SerialRxEnable();
void SerialRxDisable();
void SerialFlush();

byte USIshiftOut(byte data);

#define XLAT  PB4
#define SCLK  PB7
#define SIN  PB6
#define GSCLK PB2
#define BLANK PB3
/*

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
	char rxbuf[8];
	byte rxp = 0;

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
	bitSet(DDRB,BLANK);

	//
	byte gsval[24];
	byte data;
//	word wdata;
	word val[16] = {
		1, 3, 8, 13, 
		31, 51, 77, 91,
		134, 190, 221, 255,
		380, 260, 190, 66
	};
	
	int i;
	for (i = 0; i < 24; i++) {
		if (i % 3 == 0) {
			data = (val[i/3*2] >> 4);
		} else if (i % 3 == 1) {
			data = 0x0f & val[i/3*2];
			data <<= 4;
			data |= 0x0f & highByte(val[i/3*2+1]);
		} else {
			data = lowByte(val[i/3*2+1]);
		}
		gsval[i] = data;
	}
	
	byte sccount;

	SerialBegin();
	
	sei();
    for(;;){
		if ( bitRead(PORTB, BLANK) /* && (TCNT1 < 512) */) {
			//
			bitClear(PORTB,BLANK);
			for (sccount = 0; sccount < 24 /* 192/8 */; sccount++) {
				USIshiftOut(gsval[sccount]);
			}
			//
			SerialWrite('?');
			rxp = 0;
		}
		if (SerialRxComplete()) {
			rxbuf[rxp] = SerialReceive();
//			SerialWrite(rxbuf[rxp]);
			rxp++;
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

byte USIshiftOut(byte data) {
	USIDR = data;
	USISR = (1<<USIOIF);
	while ( !(USISR & (1<<USIOIF)) ) {
		USICR = (0b01<<USIWM0) | (1<<USICS1) | (0<<USICS0) | (1<<USICLK) | (1<<USITC);
	}
	return USIDR;
}


void SerialBegin() {
	// Set baud rate
	UBRRH = highByte(UBRRVAL);
	UBRRL = lowByte(UBRRVAL);
	// Enable receiver and transmitter
	UCSRB = (1<<RXEN)|(1<<TXEN);
	// Set frame format: 8data, 2stop bit 
	UCSRC = (0<<USBS)|(3<<UCSZ0);
	bitClear(DDRD,PD0);
}
 

void SerialWrite(unsigned char data ){
	// Wait for empty transmit buffer 
	while ( !( UCSRA & (1<<UDRE)) ) ;
	// Put data into buffer, sends the data 
	UDR = data;
}

void SerialPrint(char * msg) {
	for (;*msg != '\0';msg++) {
		SerialWrite((unsigned char) *msg);
	}
}

byte SerialRxComplete() {
	return bitRead(UCSRA, RXC);
}

void SerialRxEnable() {
	bitSet(UCSRB, RXEN);
}

void SerialRxDisable() {
	bitClear(UCSRB, RXEN);
}

byte SerialReceive( void ){
	/* Wait for data to be received */
	while ( !(UCSRA & (1<<RXC)) ) ;
	/* Get and return received data from buffer */
	return UDR;
}

void SerialFlush( void ) {
	byte dummy;
	while ( UCSRA & (1<<RXC) ) 
		dummy = UDR;
}
